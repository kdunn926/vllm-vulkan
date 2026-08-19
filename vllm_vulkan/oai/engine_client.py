# SPDX-License-Identifier: Apache-2.0
"""RustEngineClient — bridges vLLM's OpenAI server to the Vulkan Rust backend.

It implements vLLM's ``EngineClient`` interface but replaces the entire torch
engine: generation comes from a ``RustBackend`` (raw token ids), and this class
reuses vLLM's OWN detokenizer + stop logic to turn those ids into correct text +
finish_reason. That reuse (rather than reimplementing in Rust) is what keeps
token-level output bit-identical to what vLLM would produce — see oai/README.md
and the validation gate in oai/tests/.

`generate()` is the only substantive method; the other 23 are lifecycle/health
no-ops. The canonical driving loop is mirrored from vLLM's
`v1/engine/output_processor.py`:

    fr = <finish from check_stop: stop-token / EOS / length / max_model_len>
    stop_string = det.update(new_ids, stop_terminated = fr is FinishReason.STOP)
    if stop_string: fr, stop_reason = FinishReason.STOP, stop_string
    text = det.get_next_output_text(finished = fr is not None, delta = streaming)

`sampling_params.output_kind` (set by the serving layer from `stream=`) selects
DELTA (yield per-step deltas) vs FINAL_ONLY (yield one cumulative output).

NOTE (follow-up, not blocking): per-token logprobs are left as None. The Vulkan
sampler runs in Rust, so to serve `logprobs` the backend Step must also carry the
chosen token's logprob (and top-k), which this adapter would wrap in
vllm.logprobs.Logprob. Text/finish_reason are fully handled.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator

from vllm.engine.protocol import EngineClient
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import EngineCoreRequest, FinishReason
from vllm.v1.engine.detokenizer import IncrementalDetokenizer
from vllm.v1.request import Request, RequestStatus
from vllm.v1.core.sched.utils import check_stop

from .backend import RustBackend


def _prompt_token_ids(prompt) -> list[int]:
    """Extract the token ids the real renderer produced for this request."""
    if isinstance(prompt, dict):
        ids = prompt.get("prompt_token_ids")
        if ids is not None:
            return list(ids)
    ids = getattr(prompt, "prompt_token_ids", None)
    if ids is not None:
        return list(ids)
    raise ValueError("RustEngineClient.generate: no prompt_token_ids on engine input")


def _truthy(v) -> bool:
    return v is not None and str(v) != "0" and str(v).lower() != "false"


def _allowed_leads(bos_id):
    """First-token ids accepted as a well-formed chat lead: BOS plus any per-arch
    CONTROL-token leads declared in VLLM_VULKAN_CHAT_LEAD_TOKENS (comma-separated).

    A model whose chat_template.jinja correctly leads with a control token instead
    of literal BOS (e.g. Kimi emits 163587, not bos_token_id 163584) sets that env
    so the guard ACCEPTS the templated prompt while STAYING ACTIVE (mirrors
    scripts/chat_prompt_guard.resolve_allowed_leads). Default => {bos_id} only.
    """
    import os
    leads = set()
    if bos_id is not None:
        leads.add(int(bos_id))
    extra = os.environ.get("VLLM_VULKAN_CHAT_LEAD_TOKENS")
    if extra:
        for tok in str(extra).replace(" ", ",").split(","):
            if not tok:
                continue
            try:
                leads.add(int(tok, 0))
            except ValueError:
                continue
    return leads


def _assert_chat_prompt(prompt_ids, bos_id) -> None:
    """Defense-in-depth misprompt tripwire at the serve seam.

    The real vLLM renderer (HfRenderer/OnlineRenderer) already applies the chat
    template + BOS, so on the HTTP path this always passes. It exists so that if
    ANY future backend/wiring hands the engine raw, un-rendered ids (the class of
    bug that made a chat model silently emit [tok]xN and masquerade as a deep
    regression — docs/laguna-newimage-gpu-regression.md), it errors LOUDLY instead.

    Model-aware (tokenizer BOS). A control-token-led chat model (e.g. Kimi 163587)
    is ACCEPTED via VLLM_VULKAN_CHAT_LEAD_TOKENS (the guard stays active). Full
    opt-out for deliberate raw-token debug only: VLLM_VULKAN_ALLOW_RAW_PROMPT=1.
    """
    import os
    if _truthy(os.environ.get("VLLM_VULKAN_ALLOW_RAW_PROMPT")):
        return
    leads = _allowed_leads(bos_id)
    if not leads:
        return  # no BOS and no configured control lead -> nothing to assert
    if not prompt_ids:
        raise ValueError("RustEngineClient.generate: empty prompt_token_ids")
    if int(prompt_ids[0]) not in leads:
        leads_str = ",".join(str(x) for x in sorted(leads))
        raise ValueError(
            f"RustEngineClient.generate: MISPROMPT — chat model BOS={bos_id} but "
            f"prompt_token_ids[0]={prompt_ids[0]} is not an accepted chat lead "
            f"({leads_str}). The renderer (chat_template + BOS/control lead) was "
            "bypassed; a chat model fed a raw prompt degenerates to [tok]xN. Route "
            "through the OpenAI serving renderer; a control-token-led model declares "
            "its lead via VLLM_VULKAN_CHAT_LEAD_TOKENS; or set "
            "VLLM_VULKAN_ALLOW_RAW_PROMPT=1 for deliberate raw-token debugging.")


class RustEngineClient(EngineClient):
    def __init__(self, vllm_config, renderer, backend: RustBackend, tokenizer,
                 clock_manager=None):
        # Attributes the OpenAI serving layer / renderer read (see config_stub).
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.renderer = renderer          # the REAL HfRenderer
        self.input_processor = None        # stored, never called on these paths
        # Ours:
        self._backend = backend
        self._tokenizer = tokenizer
        # Serving clock manager (ClockManager): active-pin the GPU while a request
        # is in flight, idle-revert after a timeout. None => no clock mgmt (dev/CI).
        self._clock_manager = clock_manager
        self._max_model_len = getattr(vllm_config.model_config, "max_model_len", 8192)
        eos = getattr(tokenizer, "eos_token_id", None)
        self._eos_token_id = int(eos) if eos is not None else None
        bos = getattr(tokenizer, "bos_token_id", None)
        self._bos_token_id = int(bos) if bos is not None else None

    # ---- the one substantive method -------------------------------------
    async def generate(
        self, prompt, sampling_params, request_id, *args, **kwargs
    ) -> AsyncGenerator[RequestOutput, None]:
        # Serving clock lifecycle: acquire (pin on the 0->1 in-flight edge) around
        # the whole request, release in `finally` so completion, an early stop, an
        # exception, or a client-abort (GeneratorExit closes this generator and runs
        # the finally) all decrement the refcount and arm the idle-revert.
        cm = self._clock_manager
        if cm is not None:
            await cm.acquire()
        try:
            async for out in self._generate_impl(
                    prompt, sampling_params, request_id, *args, **kwargs):
                yield out
        finally:
            if cm is not None:
                await cm.release()

    async def _generate_impl(
        self, prompt, sampling_params, request_id, *args, **kwargs
    ) -> AsyncGenerator[RequestOutput, None]:
        prompt_ids = _prompt_token_ids(prompt)
        _assert_chat_prompt(prompt_ids, self._bos_token_id)   # anti-misprompt tripwire
        streaming = getattr(sampling_params, "output_kind", None) == RequestOutputKind.DELTA

        # Give the sampling params an EOS so check_stop halts on it (unless the
        # request set ignore_eos, in which case leave it unset).
        if not getattr(sampling_params, "ignore_eos", False) and self._eos_token_id is not None:
            try:
                if getattr(sampling_params, "eos_token_id", None) is None:
                    sampling_params.update_from_generation_config({}, eos_token_id=self._eos_token_id)
            except Exception:
                pass

        ecr = EngineCoreRequest(
            request_id=request_id, prompt_token_ids=list(prompt_ids),
            mm_features=None, sampling_params=sampling_params, pooling_params=None,
            arrival_time=0.0, lora_request=None, cache_salt=None, data_parallel_rank=None,
        )
        det = IncrementalDetokenizer.from_new_request(self._tokenizer, ecr)
        req = Request(
            request_id=request_id, prompt_token_ids=list(prompt_ids),
            sampling_params=sampling_params, pooling_params=None, block_hasher=None,
        )

        finish_reason: FinishReason | None = None
        stop_reason = None

        async for step in self._backend.stream(prompt_ids, sampling_params):
            tid = int(step.token_id)
            req.append_output_token_ids(tid)

            # 1) stop-token / EOS / length / max_model_len (vLLM's own check).
            #    check_stop sets req.stop_reason to the matched token id (for
            #    stop_token_ids); EOS/length leave it None.
            if check_stop(req, self._max_model_len):
                finish_reason = RequestStatus.get_finished_reason(req.status)
                stop_reason = req.stop_reason

            # 2) stop-STRING detection + trimming (vLLM's detokenizer)
            stop_terminated = finish_reason == FinishReason.STOP
            stop_string = det.update([tid], stop_terminated)
            if stop_string:
                finish_reason = FinishReason.STOP
                stop_reason = stop_string

            finished = finish_reason is not None
            text = det.get_next_output_text(finished=finished, delta=streaming)

            if streaming:
                # DELTA: emit the incremental piece (+ this step's token id).
                yield self._request_output(
                    request_id, prompt_ids, text, [tid],
                    finish_reason, stop_reason, finished, det,
                )
            if finished or step.done:
                if not streaming:
                    # FINAL_ONLY: one cumulative output at the end.
                    full = det.get_next_output_text(finished=True, delta=False)
                    yield self._request_output(
                        request_id, prompt_ids, full, list(det.output_token_ids),
                        finish_reason, stop_reason, True, det,
                    )
                return

        # Backend stream ended without a stop condition (e.g. it hit its own cap).
        if not streaming:
            full = det.get_next_output_text(finished=True, delta=False)
            yield self._request_output(
                request_id, prompt_ids, full, list(det.output_token_ids),
                finish_reason, stop_reason, True, det,
            )

    def _request_output(self, request_id, prompt_ids, text, token_ids,
                        finish_reason, stop_reason, finished, det) -> RequestOutput:
        co = CompletionOutput(
            index=0, text=text, token_ids=token_ids,
            cumulative_logprob=None, logprobs=None,   # see module docstring (logprobs TODO)
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            stop_reason=stop_reason,
        )
        return RequestOutput(
            request_id=request_id, prompt=None, prompt_token_ids=list(prompt_ids),
            prompt_logprobs=None, outputs=[co], finished=finished, num_cached_tokens=0,
        )

    # ---- lifecycle / health: trivial (23 methods) -----------------------
    async def encode(self, *a, **k):
        if False:  # make this an async generator, never yields
            yield None
        return

    async def get_supported_tasks(self):
        return ("generate",)

    @property
    def is_running(self):
        return True

    @property
    def is_stopped(self):
        return False

    @property
    def errored(self):
        return False

    @property
    def dead_error(self):
        return RuntimeError("engine dead")

    async def abort(self, request_id):
        return None

    async def notify_kv_transfer_request_rejected(self, *a, **k):
        return None

    async def is_tracing_enabled(self):
        return False

    async def do_log_stats(self, *a, **k):
        return None

    async def check_health(self):
        return None

    async def start_profile(self):
        return None

    async def stop_profile(self):
        return None

    async def reset_mm_cache(self):
        return None

    async def reset_encoder_cache(self):
        return None

    async def reset_prefix_cache(self, *a, **k):
        return True

    async def sleep(self, *a, **k):
        return None

    async def wake_up(self, *a, **k):
        return None

    async def is_sleeping(self):
        return False

    async def add_lora(self, lora_request):
        return True

    async def pause_generation(self, *a, **k):
        return None

    async def resume_generation(self):
        return None

    async def is_paused(self):
        return False

    def shutdown(self, timeout=None):
        return None
