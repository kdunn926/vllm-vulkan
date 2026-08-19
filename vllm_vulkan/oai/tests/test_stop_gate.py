# SPDX-License-Identifier: Apache-2.0
"""Stop/detok gate driving RustEngineClient.generate() directly (no HTTP).

Feeds scripted token-id streams through the REAL adapter loop (RustEngineClient
+ vLLM's IncrementalDetokenizer + check_stop) and asserts correct trimmed text,
finish_reason and output_kind (DELTA vs FINAL_ONLY). This is the always-green
correctness gate for the token->text->stop contract; a green here means the Rust
backend can safely emit raw token ids only.

Run in the base-vLLM venv (stub torch), tokenizer in HF cache:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      /opt/vllm-vulkan-venv/bin/python -m vllm_vulkan.oai.tests.test_stop_gate
"""
import asyncio
import os
import sys
import traceback
from collections.abc import AsyncIterator

MODEL = os.environ.get("OAI_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
FAILS: list[str] = []


def expect(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


class ScriptedBackend:
    """Emits a fixed token-id list (one Step per id), regardless of prompt."""
    def __init__(self, ids):
        self._ids = list(ids)

    async def stream(self, prompt_token_ids, sampling_params) -> AsyncIterator:
        from vllm_vulkan.oai.backend import Step
        n = len(self._ids)
        for i, t in enumerate(self._ids):
            yield Step(token_id=t, done=(i == n - 1))


async def _collect(engine, sampling_params, prompt_ids):
    outs = []
    async for ro in engine.generate({"prompt_token_ids": prompt_ids}, sampling_params, "rid"):
        outs.append(ro)
    return outs


def build_engine(tokenizer, ids):
    from vllm_vulkan.oai.config_stub import RustModelConfig, RustVllmConfig
    from vllm_vulkan.oai.engine_client import RustEngineClient
    vcfg = RustVllmConfig(RustModelConfig(MODEL, max_model_len=4096))
    return RustEngineClient(vcfg, renderer=None, backend=ScriptedBackend(ids), tokenizer=tokenizer)


def mk_sp(streaming=False, **kw):
    from vllm.sampling_params import SamplingParams, RequestOutputKind
    base = dict(max_tokens=128, temperature=0.0, detokenize=True)
    base.update(kw)
    sp = SamplingParams(**base)
    sp.output_kind = RequestOutputKind.DELTA if streaming else RequestOutputKind.FINAL_ONLY
    return sp


def main():
    from vllm.tokenizers.hf import CachedHfTokenizer
    tok = CachedHfTokenizer.from_pretrained(MODEL)

    # --- stop string spanning tokens: FINAL_ONLY, exclude (default) ---
    body = tok.encode("answer here</s>trailing junk")
    eng = build_engine(tok, body)
    outs = asyncio.run(_collect(eng, mk_sp(stop=["</s>"], include_stop_str_in_output=False), [0]))
    final = outs[-1].outputs[0]
    expect(final.text.startswith("answer here") and "</s>" not in final.text
           and "trailing" not in final.text and final.finish_reason == "stop"
           and final.stop_reason == "</s>",
           f"FINAL exclude-stop: text={final.text!r} finish={final.finish_reason} stop_reason={final.stop_reason!r}")

    # --- same, include-stop ---
    eng = build_engine(tok, body)
    outs = asyncio.run(_collect(eng, mk_sp(stop=["</s>"], include_stop_str_in_output=True), [0]))
    final = outs[-1].outputs[0]
    expect(final.text.endswith("</s>") and "trailing" not in final.text,
           f"FINAL include-stop: text={final.text!r}")

    # --- streaming DELTA: concatenated deltas == the trimmed final text ---
    eng = build_engine(tok, body)
    outs = asyncio.run(_collect(eng, mk_sp(streaming=True, stop=["</s>"], include_stop_str_in_output=False), [0]))
    concat = "".join(o.outputs[0].text for o in outs)
    expect(concat.startswith("answer here") and "</s>" not in concat and "trailing" not in concat,
           f"DELTA stream concat: {concat!r}")
    expect(outs[-1].finished and outs[-1].outputs[0].finish_reason == "stop",
           "DELTA stream: last chunk finished + finish_reason=stop")

    # --- stop_token_ids ---
    gen = tok.encode("hello world extra tail")
    stop_id = gen[2]
    eng = build_engine(tok, gen)
    outs = asyncio.run(_collect(eng, mk_sp(stop_token_ids=[stop_id]), [0]))
    final = outs[-1].outputs[0]
    expect(final.finish_reason == "stop" and final.stop_reason == stop_id,
           f"stop_token_ids: finish={final.finish_reason} stop_reason={final.stop_reason}")

    # --- length cap (max_tokens) via check_stop ---
    eng = build_engine(tok, [10, 11, 12, 13, 14])
    outs = asyncio.run(_collect(eng, mk_sp(max_tokens=3), [0]))
    final = outs[-1].outputs[0]
    expect(final.finish_reason == "length" and len(final.token_ids) == 3,
           f"max_tokens=3: finish={final.finish_reason} n_out={len(final.token_ids)}")

    # --- plain completion, no stop: text round-trips through detok ---
    ids = tok.encode("The quick brown fox 🦊")
    eng = build_engine(tok, ids)
    outs = asyncio.run(_collect(eng, mk_sp(), [0]))
    expect(outs[-1].outputs[0].text == tok.decode(ids),
           f"plain: {outs[-1].outputs[0].text!r} == batch decode")

    print("\n==================== SUMMARY ====================")
    if FAILS:
        print(f"{len(FAILS)} FAILED:")
        for f in FAILS:
            print("  -", f)
    else:
        print("ALL STOP/DETOK GATE CHECKS PASSED (via RustEngineClient.generate)")


if __name__ == "__main__":
    rc = 0
    try:
        main()
        rc = 1 if FAILS else 0
    except Exception:
        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    os._exit(rc)
