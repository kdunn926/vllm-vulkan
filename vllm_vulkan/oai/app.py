# SPDX-License-Identifier: Apache-2.0
"""Assemble base vLLM's OpenAI server on top of RustEngineClient.

Wires the REAL vLLM renderer (`HfRenderer` + `OnlineRenderer`) and the REAL
OpenAI routers (chat + completions) to our `RustEngineClient`, bypassing the
torch-heavy `AsyncLLM` (`build_async_engine_client`). Proven end-to-end torch-
free against vLLM 0.26.0 (see oai/README.md + oai/tests/).

Two entry points:
  * build_app(model, backend_factory, ...) -> FastAPI  (used by the tests with a
    DummyRustBackend; used in prod with a VulkanRustBackend factory)
  * serve(...) -> runs uvicorn.

STEP 5 / TODO(you): provide a backend_factory that loads the Vulkan model and
returns a VulkanRustBackend (see backend.py + vllm_vulkan/server.py for model
loading), then validate on hardware.

TRUST_REMOTE_CODE (opt-in, default OFF)
---------------------------------------
Some models ship a *custom* tokenizer whose loader is Python code in the model
repo (e.g. Kimi's ``auto_map -> tokenization_kimi.TikTokenTokenizer``). Loading
such a tokenizer requires ``trust_remote_code=True``, which EXECUTES that repo
Python in-process. This is DISABLED BY DEFAULT. A user who trusts a specific
model can opt in, per the same pattern as upstream vLLM's ``--trust-remote-code``:

  * CLI:  ``--trust-remote-code`` on the launcher (blanket, this run only).
  * env:  ``VLLM_VULKAN_TRUST_REMOTE_CODE`` —
             ``1`` / ``true`` / ``yes`` / ``on``  -> blanket enable, OR
             a comma-list of trusted model ids/dirs -> SCOPED enable for only
             those models (matched by exact id, path, or basename).

When OFF (default) behavior is unchanged: base vLLM's safe default runs and a
custom-tokenizer model errors clearly. When ON, a prominent WARNING is logged
naming the model whose repo code will execute. ONLY enable for models you trust.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable

from .backend import DummyRustBackend, RustBackend
from .config_stub import RustModelConfig, RustVllmConfig
from .engine_client import RustEngineClient

logger = logging.getLogger("vllm_vulkan.oai")

_TRUTHY = {"1", "true", "yes", "on"}


def _warn_remote_code(model: str, source: str) -> None:
    logger.warning(
        "SECURITY: trust_remote_code is ON for model '%s' (enabled via %s). "
        "Model-repo Python WILL be executed in-process while loading the "
        "tokenizer/config. Only enable this for models you trust.",
        model, source,
    )


def _model_in_scope(model: str, trusted: list[str]) -> bool:
    """True iff `model` matches one of the trusted allow-list entries. Matches an
    exact id/path, a normalized path suffix, or an equal basename so a checkpoint
    referenced as an HF id or a local dir resolves the same trusted entry."""
    m = model.rstrip("/")
    m_base = os.path.basename(m)
    for t in trusted:
        t = t.rstrip("/")
        if not t:
            continue
        if m == t or m_base == os.path.basename(t):
            return True
        if m.endswith("/" + t) or m.endswith(t):
            return True
    return False


def resolve_trust_remote_code(model: str, cli_flag: bool = False) -> bool:
    """Decide whether model-repo Python may execute when loading `model`'s
    tokenizer/config. DEFAULT OFF — returns True ONLY on an explicit opt-in:
    the launcher `--trust-remote-code` flag (`cli_flag`) or the
    VLLM_VULKAN_TRUST_REMOTE_CODE env var (blanket, or a scoped comma-list of
    trusted model ids/dirs). Logs a prominent WARNING when it returns True."""
    if cli_flag:
        _warn_remote_code(model, "CLI --trust-remote-code")
        return True
    raw = os.environ.get("VLLM_VULKAN_TRUST_REMOTE_CODE", "").strip()
    if not raw:
        return False
    if raw.lower() in _TRUTHY:
        _warn_remote_code(model, "env VLLM_VULKAN_TRUST_REMOTE_CODE (blanket enable)")
        return True
    trusted = [t.strip() for t in raw.split(",") if t.strip()]
    if _model_in_scope(model, trusted):
        _warn_remote_code(model, "env VLLM_VULKAN_TRUST_REMOTE_CODE (scoped allow-list)")
        return True
    logger.info(
        "trust_remote_code stays OFF for model '%s': not in the "
        "VLLM_VULKAN_TRUST_REMOTE_CODE allow-list %s.", model, trusted,
    )
    return False


def build_app(
    model: str,
    backend_factory: Callable[[object], RustBackend] | None = None,
    *,
    served_model_name: str | None = None,
    max_model_len: int = 8192,
    model_type: str = "qwen2",
    response_role: str = "assistant",
    trust_remote_code: bool = False,
    clock_manager: object | None = None,
    clock_fanout: Callable[[bool], None] | None = None,
):
    """Build the FastAPI app.

    backend_factory(tokenizer) -> RustBackend. If None, a DummyRustBackend
    ("Hello world") is used so the app is runnable/testable with no GPU.

    trust_remote_code: opt-in (default OFF). When True (from the launcher's
    `--trust-remote-code`) OR when the VLLM_VULKAN_TRUST_REMOTE_CODE env var
    opts this model in, the model repo's custom tokenizer code is allowed to
    execute during tokenizer load (needed for e.g. Kimi's TikToken tokenizer).
    See the module docstring. OFF by default -> base-vLLM's safe behavior.

    clock_manager: an explicit ClockManager (distributed serve passes one wired to
    its peer-fanout). If None, one is built from env (VLLM_VULKAN_CLOCK_MANAGE=1 by
    default) with the optional single-node ``clock_fanout``. Pass MANAGE=0 to
    disable when an external owner (nrun / a benchmark) already holds the clock.
    """
    from fastapi import FastAPI
    from vllm.tokenizers.hf import CachedHfTokenizer
    from vllm.renderers.hf import HfRenderer
    from vllm.renderers.online_renderer import OnlineRenderer
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    from vllm.entrypoints.openai.models.protocol import BaseModelPath
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
    from vllm.entrypoints.openai.chat_completion.api_router import attach_router as chat_router
    from vllm.entrypoints.openai.completion.api_router import attach_router as cmpl_router

    served = served_model_name or model
    trust = resolve_trust_remote_code(model, cli_flag=trust_remote_code)
    # Only pass the kwarg when opting in, so the DEFAULT (OFF) path is byte-for-
    # byte the base-vLLM call — an unchanged safe default.
    tok_kwargs = {"trust_remote_code": True} if trust else {}
    tokenizer = CachedHfTokenizer.from_pretrained(model, **tok_kwargs)
    model_config = RustModelConfig(
        model, served_model_name=served, max_model_len=max_model_len, model_type=model_type,
        trust_remote_code=trust,
    )
    vllm_config = RustVllmConfig(model_config)

    renderer = HfRenderer(vllm_config, tokenizer)
    online = OnlineRenderer(
        model_config=model_config, renderer=renderer, request_logger=None,
        chat_template=None, chat_template_content_format="auto",
        trust_request_chat_template=False, enable_auto_tools=False,
        tool_parser=None, reasoning_parser=None, default_chat_template_kwargs=None,
    )

    backend = (backend_factory(tokenizer) if backend_factory is not None
               else DummyRustBackend(tokenizer))

    # Serving clock manager: active-pin the GPU while requests are in flight, idle-
    # revert after a timeout (the ~45% serve-latency lever). Explicit one wins;
    # else build from env (default-on). install_process_handlers() adds the atexit +
    # SIGTERM/SIGINT revert; a FastAPI shutdown hook reverts on a clean stop too.
    if clock_manager is None:
        try:
            from .clock_manager import ClockManager
        except ImportError:
            # Clock management is an optional, platform-specific add-on (it pins a
            # GPU governor). When the module is absent the serve path simply runs
            # without it — nothing else depends on a live manager.
            ClockManager = None
        if ClockManager is not None:
            clock_manager = ClockManager.from_env(fanout=clock_fanout)
    if clock_manager is not None:
        clock_manager.install_process_handlers()
    engine = RustEngineClient(vllm_config, renderer, backend, tokenizer,
                              clock_manager=clock_manager)

    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=[BaseModelPath(name=served, model_path=model)],
        lora_modules=None,
    )
    chat = OpenAIServingChat(
        engine, models, response_role, online_renderer=online, request_logger=None,
        chat_template=None, chat_template_content_format="auto",
        trust_request_chat_template=False, return_tokens_as_token_ids=False,
        reasoning_parser="", enable_auto_tools=False, tool_parser=None,
    )
    chat.warmup()
    cmpl = OpenAIServingCompletion(
        engine, models, online_renderer=online, request_logger=None,
        return_tokens_as_token_ids=False,
    )

    app = FastAPI(title="vllm-vulkan (base-vLLM OpenAI frontend)")
    chat_router(app)
    cmpl_router(app)

    # Clean-stop revert: uvicorn's graceful shutdown fires this before exit so the
    # governor is dropped back to the idle floor (belt-and-suspenders with the
    # atexit/signal revert in install_process_handlers()).
    app.state.clock_manager = clock_manager
    # newer Starlette removed App.add_event_handler → fall back to the router's
    # on_shutdown list (the atexit/SIGTERM revert in install_process_handlers()
    # already covers clean-stop, so this is belt-and-suspenders either way).
    if clock_manager is not None:
        if hasattr(app, "add_event_handler"):
            app.add_event_handler("shutdown", clock_manager.shutdown_revert)
        else:
            app.router.on_shutdown.append(clock_manager.shutdown_revert)

    app.state.openai_serving_chat = chat
    app.state.openai_serving_chat_batch = None
    app.state.openai_serving_completion = cmpl
    app.state.enable_server_load_tracking = False
    app.state.server_load_metrics = 0
    return app


def serve(
    model: str,
    backend_factory: Callable[[object], RustBackend] | None = None,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    **build_kwargs,
) -> None:
    import uvicorn
    app = build_app(model, backend_factory, **build_kwargs)
    uvicorn.run(app, host=host, port=port)
