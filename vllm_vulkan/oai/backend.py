# SPDX-License-Identifier: Apache-2.0
"""The Rust-backend seam for the base-vLLM OpenAI frontend.

After the three de-risks, the Rust backend's ENTIRE responsibility is:

    (prompt_token_ids, SamplingParams) -> async stream of sampled token ids

Rendering (prompt->ids), detokenization (ids->text), stop-strings, stop-tokens/
EOS/length/min_tokens/ignore_eos and finish_reason all run torch-free in Python
(reusing vLLM's own code) inside RustEngineClient. So the backend implements a
single method, `stream()`, that yields raw token ids and a done flag.

Three implementations:
  * RustBackend       - the Protocol (the seam).
  * DummyRustBackend  - echoes a fixed reply; drives the validation gate + CI
                        with NO GPU / NO _rs. Torch-free.
  * VulkanRustBackend - the real one: wraps `vllm_vulkan._rs.VulkanModel` using
                        the proven prefill/decode loop from vllm_vulkan.server.

>>> THE RUST BINDING IS THE ONE INTENTIONALLY-UNFINISHED PIECE <<<
VulkanRustBackend takes an already-constructed `_rs.VulkanModel` (dependency
injection). Constructing/loading that model (find_safetensors + VulkanModel(...),
see vllm_vulkan/server.py) and validating the decode loop ON HARDWARE is step 5,
left to you. The decode loop below mirrors server.py's proven single-token path;
verify token-for-token against the standalone server on a real GPU before relying
on it.
"""
from __future__ import annotations

import random
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class Step:
    """One decode step from the backend."""
    token_id: int
    done: bool = False   # backend-side natural end (e.g. it hit an internal cap)


@runtime_checkable
class RustBackend(Protocol):
    async def stream(
        self, prompt_token_ids: list[int], sampling_params
    ) -> AsyncIterator[Step]:
        """Yield one Step (sampled token id) per decode position.

        The backend does NOT apply stop strings, stop tokens, EOS, min_tokens or
        detokenization — RustEngineClient does, reusing vLLM's logic. The backend
        MAY set done=True to end early (its own limits); otherwise it should keep
        yielding until the caller stops consuming (the caller enforces max_tokens
        and all stop conditions).
        """
        ...


class DummyRustBackend:
    """Deterministic, GPU-free backend for the validation gate and CI.

    Encodes a fixed reply string with the provided tokenizer so the produced ids
    round-trip through the real detokenizer to exactly `reply`. Torch-free.
    """

    def __init__(self, tokenizer, reply: str = "Hello world"):
        self._tok = tokenizer
        self._reply_ids: list[int] = list(tokenizer.encode(reply))

    async def stream(
        self, prompt_token_ids: list[int], sampling_params
    ) -> AsyncIterator[Step]:
        n = len(self._reply_ids)
        for i, tid in enumerate(self._reply_ids):
            yield Step(token_id=tid, done=(i == n - 1))


class VulkanRustBackend:
    """Real backend over `vllm_vulkan._rs.VulkanModel`.

    Mirrors the prefill + decode loop in vllm_vulkan/server.py:generate_response.
    Pass an already-loaded VulkanModel; see server.py for model loading.

    STEP 5 / TODO(you): validate this decode loop on real hardware, token-for-
    token against the standalone server. The per-call signature used here is the
    one proven in server.py:
        model.reset_kv_cache()
        model.forward(token_id, pos)                                  # prefill
        model.forward_and_sample(token_id, pos, temp, top_p, top_k, rand) -> int
    If the Rust API grows a native streaming/batched entry point, implement it
    here instead of the per-token Python loop.
    """

    def __init__(self, model, session=None):
        self._model = model  # vllm_vulkan._rs.VulkanModel
        # Session-KV continuation (Item 3): a persistent per-backend manager
        # keeps the resident KV alive across streams and prefills only the
        # appended tail of a continuing transcript. Default-off (unset
        # VLLM_VULKAN_SESSION_KV and VLLM_VULKAN_KV_STORE_DIR => reset+full
        # prefill every stream, byte-identical to the legacy path).
        if session is None:
            from ..session_kv import SessionKvManager
            session = SessionKvManager(model)
        self._session = session

    async def stream(
        self, prompt_token_ids: list[int], sampling_params
    ) -> AsyncIterator[Step]:
        temperature = float(getattr(sampling_params, "temperature", 1.0) or 0.0)
        top_p = float(getattr(sampling_params, "top_p", 1.0) or 1.0)
        top_k = int(getattr(sampling_params, "top_k", 0) or 0)

        m = self._model
        ids = list(prompt_token_ids)

        # Reset-or-continue: a disabled session resets and starts at 0 (legacy
        # full re-prefill); an enabled one continuing this transcript returns the
        # resident prefix length so only the appended tail is prefilled.
        start = self._session.prepare(ids)
        start = min(start, len(ids) - 1)

        # Prefill every prompt token from `start` except the last (sampled below).
        for pos in range(start, len(ids) - 1):
            m.forward(ids[pos], pos)
        # The whole prompt is now resident.
        self._session.mark_resident(ids)

        pos = len(ids) - 1
        cur = ids[-1]
        # Decode indefinitely; RustEngineClient enforces max_tokens + all stop
        # conditions and stops consuming. `random.random()` seeds Rust sampling
        # exactly as server.py does (note: Math.random-free constraint applies to
        # WORKFLOW scripts, not to this runtime module).
        while True:
            nxt = m.forward_and_sample(cur, pos, temperature, top_p, top_k, random.random())
            yield Step(token_id=int(nxt), done=False)
            # `cur` was just forwarded at `pos` (advancing the KV); record it as
            # resident so a follow-up same-session stream can continue past it.
            self._session.observe(cur)
            cur = nxt
            pos += 1


class DistributedVulkanRustBackend:
    """M3 — DISTRIBUTED backend: a multi-node PP model served through the SAME
    `stream()` seam as the single-node one. It wraps a `DistHead` (scripts/
    serve_head.py) — vCCL rank0 + first PP stage — that drives the persistent peer
    ring (scripts/serve_peer.py) over vCCL and returns rank0 the full `[vocab]`.

    vLLM/the oai adapter never sees the distribution: `stream()` yields sampled
    token ids exactly like `VulkanRustBackend`, but each prefill/decode is a fanned-
    out N-node forward (scope §1: the head process IS rank0). `head` is injected
    already-constructed (the plugin `load_model` hook builds it: launch peers ->
    establish comm -> load rank0 stage; scope §5). Its contract:
        head.prefill(prompt_token_ids) -> [vocab]   # cache-populating, one-shot
        head.decode(token_id)          -> [vocab]   # one fused native hop/token

    MVP = greedy (argmax); the full sampler (temperature/top-k/top-p on the ringed
    `[vocab]`) is a drop-in Python sampler or vLLM's `Sampler` — same seam, since we
    now hold the whole logit vector at rank0 (the reason the logits ring-back
    variant `pp_step_laguna_logits` exists vs the argmax-fused `pp_step_laguna`).
    """

    def __init__(self, head, sampler=None):
        self._head = head          # DistHead (duck-typed: .prefill/.decode)
        self._sampler = sampler    # optional callable([vocab], sampling_params)->int

    def _sample(self, logits, sampling_params):
        if self._sampler is not None:
            return int(self._sampler(logits, sampling_params))
        # MVP greedy argmax (== the offline pp_laguna.py argmax reference).
        bi, bv = 0, float("-inf")
        for i, v in enumerate(logits):
            if v > bv:
                bv = v; bi = i
        return bi

    async def stream(
        self, prompt_token_ids: list[int], sampling_params
    ) -> AsyncIterator[Step]:
        head = self._head
        logits = head.prefill(list(prompt_token_ids))   # distributed prefill -> [vocab]
        cur = self._sample(logits, sampling_params)
        while True:
            yield Step(token_id=int(cur), done=False)
            logits = head.decode(cur)                    # one distributed decode step
            cur = self._sample(logits, sampling_params)
