# SPDX-License-Identifier: Apache-2.0
"""Standalone OpenAI-compatible API server backed by the Rust VulkanModel.

This server bypasses vLLM's internal model runner entirely and uses our
Rust VulkanModel directly for generation.  This gives ~3 tok/s on GB10
vs 1.7 tok/s from vLLM's CPU backend.

Usage:
    python -m vllm_vulkan.server google/gemma-4-E2B-it --port 8000

API:
    POST /v1/chat/completions  (OpenAI-compatible)
    GET  /v1/models
    GET  /health
"""

import argparse
import glob
import hashlib
import logging
import os
import random
import time
import uuid
from pathlib import Path

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def _fold_u64(data: bytes) -> int:
    """Stable 64-bit fold of `data` (blake2b truncated) for the KV content
    fingerprint identity. Deterministic across processes/nodes so a
    content-addressed KV tile warmed on one node is a hit on all."""
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def _ensure_kv_identity(model, tokenizer, model_dir: "str | None") -> None:
    """Fold the tokenizer + checkpoint identity into the NAS KV content
    fingerprint ONCE per process (NAS prefix-cache Phase 1, scope §3.3): a
    tokenizer bump or re-quant then MISSES cleanly instead of serving stale KV.

    Complements the `SessionKvManager`, which OWNS the resident/NAS restore
    policy (`prepare` continues a warm session or `kv_cache_load`s a cold prefix;
    `persist` `kv_cache_store`s). This only ARMS the identity the Rust
    `content_fingerprint` folds — it never restores or resets KV itself, so it
    can't double-restore or fight the session. No-op if the model lacks the
    pymethod (older `_rs` build) or the identity is already set."""
    if getattr(model, "_kv_identity_set", False):
        return
    setter = getattr(model, "kv_cache_set_identity", None)
    if setter is None:
        return
    # tokenizer identity: name + vocab size + a hash of the merges/vocab file if
    # discoverable; falls back to name_or_path + vocab_size (still catches a swap).
    tok_bytes = f"{getattr(tokenizer, 'name_or_path', '')}|{tokenizer.vocab_size}".encode()
    tok_file = None
    if model_dir:
        for cand in ("tokenizer.json", "tokenizer.model"):
            p = os.path.join(model_dir, cand)
            if os.path.exists(p):
                tok_file = p
                break
    if tok_file:
        with open(tok_file, "rb") as fh:
            tok_bytes = fh.read()
    tokenizer_hash = _fold_u64(tok_bytes)
    # weights identity: hash of the safetensors index (or the sorted shard
    # (name, size) listing) so a re-quant changes the id. 0 when undiscoverable
    # (the store stays self-consistent; the tokenizer hash still gates drift).
    weights_id = 0
    if model_dir:
        idx = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx, "rb") as fh:
                weights_id = _fold_u64(fh.read())
        else:
            shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
            listing = "".join(f"{os.path.basename(s)}:{os.path.getsize(s)}" for s in shards)
            if listing:
                weights_id = _fold_u64(listing.encode())
    setter(weights_id, tokenizer_hash)
    model._kv_identity_set = True


def find_safetensors(model_name_or_path: str) -> str:
    """Find the safetensors file for a model."""
    # Try local directory first
    if os.path.isdir(model_name_or_path):
        files = glob.glob(f"{model_name_or_path}/*.safetensors")
        if files:
            return sorted(files)[0]

    # Try HuggingFace cache
    try:
        import huggingface_hub

        local_dir = huggingface_hub.snapshot_download(
            model_name_or_path,
            local_files_only=True,
            ignore_patterns=["*.bin", "*.gguf", "*.pt"],
        )
        files = glob.glob(f"{local_dir}/*.safetensors")
        if files:
            return sorted(files)[0]
    except Exception as e:
        logger.warning("Could not find model in HF cache: %s", e)

    raise FileNotFoundError(f"No safetensors file found for {model_name_or_path}")


def greedy_sample(logits: list[float]) -> int:
    """Return the token with the highest logit.

    Superseded by ``model.forward_and_sample`` (Rust, ``src/lib.rs`` /
    ``src/model.rs``) in ``generate()`` below — kept here for any external
    callers that already depend on this exact pure-Python signature.
    """
    return max(range(len(logits)), key=lambda i: logits[i])


def temperature_sample(
    logits: list[float], temperature: float = 1.0, top_p: float = 1.0, top_k: int = 64
) -> int:
    """Sample from logits with temperature, top-p, and top-k filtering.

    Superseded by ``model.forward_and_sample`` (Rust, ``src/lib.rs`` /
    ``src/model.rs``) in ``generate()`` below: this pure-Python
    implementation does a full ``sorted()`` over the entire vocab (plus
    several more full-vocab list comprehensions) on every call, which
    measured ~82ms/call at Gemma4-E2B's 262144-token vocab — vs. ~8.6ms/call
    even through ``vllm_vulkan._rs.sample_logits`` (which still pays a
    Python-list round trip that ``forward_and_sample`` skips entirely, by
    never converting the logit vector out of Rust in the first place). Kept
    here for any external callers that already depend on this exact
    pure-Python signature, and as a readable reference for the algorithm
    `model::sample_with_temperature` (Rust) implements.
    """
    import math

    if temperature == 0.0:
        return greedy_sample(logits)

    # Apply temperature
    scaled = [v / temperature for v in logits]

    # Softmax
    max_l = max(scaled)
    exp_l = [math.exp(x - max_l) for x in scaled]
    total = sum(exp_l)
    probs = [x / total for x in exp_l]

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, len(probs))
        top_k_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[
            :top_k
        ]
        top_k_probs = [probs[i] for i in top_k_indices]
        total_k = sum(top_k_probs)
        top_k_probs = [p / total_k for p in top_k_probs]
    else:
        top_k_indices = list(range(len(probs)))
        top_k_probs = probs

    # Top-p (nucleus) filtering
    sorted_indices = sorted(
        range(len(top_k_probs)), key=lambda i: top_k_probs[i], reverse=True
    )
    cumsum = 0.0
    nucleus = []
    for idx in sorted_indices:
        cumsum += top_k_probs[idx]
        nucleus.append(idx)
        if cumsum >= top_p:
            break

    nucleus_probs = [top_k_probs[i] for i in nucleus]
    total_n = sum(nucleus_probs)
    nucleus_probs = [p / total_n for p in nucleus_probs]

    # Sample
    r = random.random()
    cumsum = 0.0
    for i, p in zip(nucleus, nucleus_probs, strict=False):
        cumsum += p
        if r <= cumsum:
            return top_k_indices[i]
    return top_k_indices[nucleus[-1]]


def generate(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 64,
    session=None,
) -> tuple[str, int, int]:
    """Generate a response using the Rust VulkanModel.

    Returns: (generated_text, num_prompt_tokens, num_completion_tokens)

    ``session`` (optional ``SessionKvManager``): when supplied and enabled, the
    model's KV cache is kept alive across requests and only the appended tail of
    a continuing conversation is prefilled (Item 3 — session-KV continuation).
    When ``None`` or disabled, behavior is byte-identical to the legacy
    reset+full-prefill.
    """
    # Format prompt using the tokenizer's chat template.
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0].tolist()

    # NAS prefix-cache (Phase 1): arm the model/tokenizer identity folded into the
    # KV content fingerprint ONCE (idempotent). This does NOT restore or reset KV
    # — the SessionKvManager below owns that (resident continuation on a warm turn,
    # `kv_cache_load` on a cold prefix). Arming here just makes a re-quant /
    # tokenizer bump MISS cleanly instead of serving stale gemma/Laguna tiles.
    _model_dir = getattr(tokenizer, "name_or_path", None)
    _model_dir = _model_dir if _model_dir and os.path.isdir(_model_dir) else None
    _ensure_kv_identity(model, tokenizer, _model_dir)

    # Reset-or-continue the KV cache. With no session (or a disabled one) this
    # resets and starts at 0 == the legacy full re-prefill; an enabled session
    # continuing a transcript returns the resident prefix length so ONLY the
    # appended tail is prefilled (bit-exact by construction — see the Rust gate
    # session_continuation_matches_full_reprefill).
    if session is not None:
        start = session.prepare(input_ids)
    else:
        model.reset_kv_cache()
        start = 0
    # Never prefill past the last token (its logits are what we sample from);
    # clamp so a full residency hit still re-forwards the final token to produce
    # a fresh logit vector.
    start = min(start, len(input_ids) - 1)

    # Prefill: run forward for each prompt token from `start` except the last
    # (whose logits are sampled below) — advances the KV cache only, discarding
    # logits nothing will read.
    for pos in range(start, len(input_ids) - 1):
        model.forward(input_ids[pos], pos)

    # Get next token from the last prefill step. forward_and_sample (Rust)
    # replaces forward() + greedy_sample()/temperature_sample(): sampling
    # happens without ever converting the 262144-element logit vector into
    # a Python object, and without CPython's interpreter overhead for the
    # temperature/top-p/top-k algorithm itself (measured ~82ms/call for the
    # old pure-Python temperature_sample vs. ~8.6ms/call even through the
    # standalone vllm_vulkan._rs.sample_logits, which still pays a
    # Python-list round trip that forward_and_sample skips entirely — see
    # temperature_sample's docstring above and model::sample_with_temperature's
    # doc comment in the Rust source).
    last_pos = len(input_ids) - 1
    next_token = model.forward_and_sample(
        input_ids[-1], last_pos, temperature, top_p, top_k, random.random()
    )
    # The whole prompt is now resident in the KV (prefill loop + the final
    # token just forwarded above); record it so the next turn can continue.
    if session is not None:
        session.mark_resident(input_ids)

    # Decode: generate new tokens.
    generated_ids: list[int] = []
    pos = len(input_ids)
    eos_token_id = tokenizer.eos_token_id
    # Stop on EOS plus any chat end-of-turn marker the tokenizer defines.
    # Covers Gemma (<end_of_turn>) and Qwen3 (<|im_end|>) without hardcoding ids.
    stop_tokens = {eos_token_id}
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for marker in ("<end_of_turn>", "<|im_end|>"):
        tid = tokenizer.convert_tokens_to_ids(marker)
        if tid is not None and tid != unk_id:
            stop_tokens.add(tid)

    while len(generated_ids) < max_new_tokens:
        generated_ids.append(next_token)
        if next_token in stop_tokens:
            break

        # forward_and_sample forwards `next_token` at `pos` (advancing the KV)
        # and returns the following token; record `next_token` as now-resident.
        if session is not None:
            session.observe(next_token)
        next_token = model.forward_and_sample(
            next_token, pos, temperature, top_p, top_k, random.random()
        )
        pos += 1

    # Persist the (exact) resident context to the NAS overflow tier, if on.
    if session is not None:
        session.persist()

    # Remove trailing EOS/end-of-turn tokens.
    while generated_ids and generated_ids[-1] in stop_tokens:
        generated_ids.pop()

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, len(input_ids), len(generated_ids)


def make_app(model_name: str, model, tokenizer):
    """Create the FastAPI application."""
    import asyncio

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(title="vllm-vulkan API server")

    # One persistent session manager per server process: it keeps the single
    # resident KV cache alive across requests and continues (rather than
    # reset+full-reprefills) whenever a request extends the resident transcript.
    # Disabled by default (unset VLLM_VULKAN_SESSION_KV and VLLM_VULKAN_KV_STORE_DIR
    # => reset+full-prefill every turn, byte-identical to the legacy path).
    from .session_kv import SessionKvManager

    session = SessionKvManager(model)
    if session.enabled:
        logger.info(
            "session-KV continuation ENABLED (resident-first%s)",
            " + NAS overflow" if session.use_nas else "",
        )
    # Serialize requests: the model owns ONE KV cache, and continuation is
    # single-stream (max_num_seqs=1) by construction.
    import threading

    session_lock = threading.Lock()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "vllm-vulkan",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict):
        try:
            messages = request.get("messages", [])
            max_tokens = request.get("max_tokens", 200)
            temperature = request.get("temperature", 1.0)
            top_p = request.get("top_p", 0.95)
            top_k = request.get("top_k", 64)

            def _run():
                # The lock makes concurrent HTTP requests serialize onto the one
                # resident KV cache (no interleaving of two sessions' prefills).
                with session_lock:
                    return generate(
                        model, tokenizer, messages, max_tokens,
                        temperature, top_p, top_k, session=session,
                    )

            t0 = time.perf_counter()
            text, n_prompt, n_gen = await asyncio.get_event_loop().run_in_executor(
                None, _run
            )
            elapsed = time.perf_counter() - t0
            tok_per_sec = n_gen / elapsed if elapsed > 0 else 0

            logger.info(
                "Generated %d tokens in %.1fs = %.1f tok/s", n_gen, elapsed, tok_per_sec
            )

            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": n_prompt,
                    "completion_tokens": n_gen,
                    "total_tokens": n_prompt + n_gen,
                },
            }
        except Exception as e:
            logger.exception("Error in chat completion")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "type": "InternalServerError"}},
            )

    return app


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(description="vllm-vulkan standalone server")
    parser.add_argument("model", help="Model name or path (e.g. google/gemma-4-E2B-it)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--device-idx", type=int, default=0)
    args = parser.parse_args()

    # Load the model.
    logger.info("Loading tokenizer for %s...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if getattr(tokenizer, "chat_template", None) is None:
        # Base models (e.g. google/gemma-4-E2B) ship without a chat template.
        # Fall back to the packaged instruction-tuned template so that
        # /v1/chat/completions still works.
        template_path = Path(__file__).parent / "template_gemma4.jinja"
        if not template_path.exists():
            raise FileNotFoundError(
                f"Tokenizer has no chat template, and the fallback template "
                f"was not found at {template_path}. Chat completions will fail."
            )
        logger.info(
            "Tokenizer has no chat template; using packaged fallback from %s",
            template_path,
        )
        tokenizer.chat_template = template_path.read_text(encoding="utf-8")

    logger.info("Finding safetensors file...")
    st_path = find_safetensors(args.model)
    logger.info("Loading VulkanModel from %s...", st_path)

    from vllm_vulkan._rs import VulkanModel

    t0 = time.perf_counter()
    model = VulkanModel(
        st_path, max_seq_len=args.max_seq_len, device_idx=args.device_idx
    )
    elapsed = time.perf_counter() - t0

    logger.info(
        "Model loaded in %.1fs: %d layers, GPU=%s",
        elapsed,
        model.num_layers(),
        model.has_gpu(),
    )

    # Quick test.
    logger.info("Running test forward pass...")
    t1 = time.perf_counter()
    model.forward(1, 0)
    model.reset_kv_cache()
    logger.info("Test forward: %.0fms", (time.perf_counter() - t1) * 1000)

    # Start server.
    import uvicorn

    app = make_app(args.model, model, tokenizer)
    logger.info("Starting server on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
