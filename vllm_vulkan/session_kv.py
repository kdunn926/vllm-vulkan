# SPDX-License-Identifier: Apache-2.0
"""Server-side session-KV continuation state manager (KV-reuse Item 3).

The serve path (`vllm_vulkan/server.py::generate`, `oai/backend.py::
VulkanRustBackend.stream`) historically did ``model.reset_kv_cache()`` followed
by a FULL re-prefill of the entire conversation on EVERY request. On a
multi-turn agentic transcript that re-computes turn 1..N-1 from scratch on turn
N — 17-35x more forward passes than necessary.

``SessionKvManager`` removes that waste with pure **server-side state
management** (no engine/kernel code): it keeps the model's KV cache alive
across requests and, when the new request's token stream is an extension of the
tokens already resident in the KV, prefills ONLY the appended tail via the
existing arch-general ``prefill_logits(tail, start_pos)`` seam. The KV for the
shared prefix is already correct in the model — nothing to recompute.

## Why this is bit-exact by construction

``model.forward(token, pos)`` / ``prefill_logits(tail, start_pos)`` are
position-addressed and deterministic (see ``src/lib.rs::prefill_logits`` and the
Rust gate ``session_continuation_matches_full_reprefill`` in
``src/qwen35.rs``): a KV cache that already holds ``[0, L)`` plus a tail fed at
``start_pos = L`` is byte-identical to a clean feed of ``[0, L+tail)``. So
turn-N-with-continuation == turn-N-full-reprefill, argmax-identical.

## Key unification (one content-addressed store, not a fork)

The resident tier keys continuation on the ACTUAL resident token content (a
longest-common-prefix check against ``resident_ids``); the cold/overflow tier
keys on a content-address of the same tokens (``kvstore.rs``'s chunk-chain
hash, via ``VulkanModel.kv_cache_load``/``kv_cache_store``). Both tiers key off
the SAME conversation content — the "session id = stable-prefix hash" design
decision — so they are two tiers of ONE store, not two independent caches:

  * resident-first (hot): same node + same session still alive in RAM/GPU →
    continue with zero I/O, prefill only the tail;
  * NAS overflow (cold): a session evicted from residency (or arriving on a
    cold node) is restored from the content-addressed ``KvStore`` (the existing
    ``kv_cache_load``/``kv_cache_store`` pymethods) then continues.

## Default-safe gate

Disabled unless ``VLLM_VULKAN_SESSION_KV`` is truthy OR
``VLLM_VULKAN_KV_STORE_DIR`` is set. When disabled, ``prepare()`` resets the KV
and returns ``start_pos = 0`` — i.e. the exact legacy reset+full-prefill
behavior, byte-identical. Setting ``VLLM_VULKAN_KV_STORE_DIR`` additionally
turns on the NAS overflow tier (restore-on-cold-turn / persist-on-commit).
"""
from __future__ import annotations

import os
from typing import Sequence


def _truthy(v) -> bool:
    return v is not None and str(v) != "0" and str(v).lower() not in ("", "false", "no")


class SessionKvManager:
    """Keeps a single-stream model's KV cache alive across serve requests and
    decides, per request, whether to CONTINUE (prefill only the appended tail)
    or RESET (legacy full re-prefill, optionally seeded from the NAS store).

    Single-stream (``max_num_seqs=1``) by construction — the model owns exactly
    one resident KV cache, so at most one session is hot at a time; a request
    for a different (or diverged) session resets residency and, if the NAS tier
    is on, restores that session's longest cached prefix from the store.

    State owned: ``resident_ids`` — the exact token sequence currently reflected
    in the model's live KV cache (empty == cache reset / nothing resident).
    """

    def __init__(self, model, *, enabled: bool | None = None, use_nas: bool | None = None):
        self.model = model
        if enabled is None:
            enabled = _truthy(os.environ.get("VLLM_VULKAN_SESSION_KV")) or bool(
                os.environ.get("VLLM_VULKAN_KV_STORE_DIR")
            )
        self.enabled = bool(enabled)
        if use_nas is None:
            use_nas = bool(os.environ.get("VLLM_VULKAN_KV_STORE_DIR"))
        # The NAS overflow tier only makes sense when continuation is enabled.
        self.use_nas = bool(use_nas) and self.enabled
        self.resident_ids: list[int] = []
        # Lightweight instrumentation for logging / the "fewer forward passes"
        # story — never affects correctness.
        self.last_saved_prefill = 0

    @staticmethod
    def _common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    def prepare(self, full_ids: Sequence[int]) -> int:
        """Ensure the model's KV holds a valid prefix of ``full_ids`` and return
        the ``start_pos`` from which the caller must prefill (``full_ids
        [start_pos:]``).

        * Disabled → ``reset_kv_cache()`` + return ``0`` (byte-identical legacy
          reset+full-prefill).
        * Enabled + the live resident KV is a prefix of this request →
          CONTINUE: no reset, return ``len(resident_ids)`` so only the appended
          tail is prefilled.
        * Enabled + divergence / cold turn → reset residency, then (NAS tier
          only) restore this session's longest cached prefix from the
          content-addressed ``KvStore`` and return the restored length.
        """
        full_ids = list(full_ids)
        self.last_saved_prefill = 0
        if not self.enabled:
            self.model.reset_kv_cache()
            self.resident_ids = []
            return 0

        lcp = self._common_prefix_len(self.resident_ids, full_ids)
        # Resident-first: the whole live KV is a prefix of this request → the
        # cheapest possible path, continue from where residency ends with no I/O.
        if self.resident_ids and lcp == len(self.resident_ids):
            self.last_saved_prefill = lcp
            # resident_ids stays as-is (== full_ids[:lcp]); the caller prefills
            # the tail and then mark_resident()s the grown prompt.
            return lcp

        # Divergence within residency, or a cold session: drop residency and
        # reset, then try the NAS overflow tier for a content-addressed prefix.
        self.model.reset_kv_cache()
        self.resident_ids = []
        loaded = 0
        if self.use_nas:
            try:
                loaded = int(self.model.kv_cache_load(full_ids))
            except Exception:
                loaded = 0
        self.resident_ids = full_ids[:loaded]
        self.last_saved_prefill = loaded
        return loaded

    def mark_resident(self, ids: Sequence[int]) -> None:
        """Record that the model's KV now reflects EXACTLY ``ids``. Call after
        prefilling a prompt so residency == live KV content. ``resident_ids``
        must never overstate the KV (``prepare`` trusts it to decide what may be
        skipped), so callers pass the exact forwarded token sequence. No-op when
        disabled."""
        if self.enabled:
            self.resident_ids = list(ids)

    def observe(self, token_id: int) -> None:
        """Append one just-forwarded decode token to ``resident_ids`` so
        residency stays in lock-step with the live KV token-for-token. Call it
        for each token actually fed through ``forward``/``forward_and_sample``
        (NOT for a sampled-but-unforwarded final/stop token). No-op when
        disabled."""
        if self.enabled:
            self.resident_ids.append(int(token_id))

    def persist(self) -> None:
        """Persist the current resident context to the content-addressed NAS
        store (idempotent — only genuinely new chunk boundaries write). No-op
        unless the NAS overflow tier is on. A store failure never breaks
        generation: the hot tier is unaffected; the cold tier just misses next
        time."""
        if not self.use_nas or not self.resident_ids:
            return
        try:
            self.model.kv_cache_store(list(self.resident_ids), len(self.resident_ids))
        except Exception:
            pass
