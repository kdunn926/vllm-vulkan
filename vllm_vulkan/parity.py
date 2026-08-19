# SPDX-License-Identifier: Apache-2.0
"""Numerical parity harness for the Rust Qwen3 forward pass.

Cross-validates vllm-vulkan's **CPU reference** Qwen3 implementation against a
ground-truth HuggingFace ``transformers`` run on the *same* HF safetensors
(identical weights, both in float32).  This confirms the ported architecture is
numerically correct, independent of the (lossy) f16 Vulkan path.

Why transformers and not qwen3-mlx as the oracle?
  - It loads the exact same HF safetensors vllm-vulkan loads (identical weights).
  - It computes in float32, matching vllm-vulkan's CPU reference path, so any
    diff is a real bug rather than bf16-vs-f32 rounding or a 4-bit MLX quant.
  - It is the ground truth qwen3-mlx is itself validated against.
qwen3-mlx remains a useful *secondary* sanity check on Apple Silicon, but it
would introduce precision/format noise that weakens a strict parity test.

CLI:
    python -m vllm_vulkan.parity /path/to/Qwen3-model-dir --prompt "Hello"
    python -m vllm_vulkan.parity /path/to/Qwen3-model-dir --tokens 9707,11,1879
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys

# Pass criteria for the strict numeric check.
DEFAULT_LOGIT_COSINE = 0.9995
DEFAULT_LAYER_COSINE = 0.999


def _first_safetensors(model_dir: str) -> str:
    """Return any one *.safetensors in the model dir (the Rust loader merges
    all sibling shards itself, so the specific file does not matter)."""
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors found in {model_dir}")
    return files[0]


def tokenize_prompt(model_dir: str, prompt: str) -> list[int]:
    """Tokenize ``prompt`` with the model's own tokenizer (no chat template)."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(model_dir)
    return tok.encode(prompt)


def dump_vulkan(model_dir: str, token_ids: list[int]) -> dict:
    """Run the Rust CPU reference Qwen3 path; return per-layer hidden + logits."""
    from vllm_vulkan._rs import VulkanModel  # noqa: PLC0415

    st = _first_safetensors(model_dir)
    max_seq = max(len(token_ids) + 8, 64)
    model = VulkanModel(st, max_seq_len=max_seq, device_idx=0)
    layers, logits = model.debug_qwen_sequence([int(t) for t in token_ids])
    return {"layers": [list(map(float, l)) for l in layers], "logits": list(map(float, logits))}


def dump_reference_hf(model_dir: str, token_ids: list[int]) -> dict:
    """Run HF transformers (float32, CPU); return per-layer hidden + logits.

    ``hidden_states`` from transformers is ``(num_layers + 1)`` long:
    ``[embeddings, out_of_layer_0, ..., out_of_layer_{N-1}]`` where the final
    entry has the model's final norm applied.  We take entries ``1..N`` (the
    per-decoder-layer outputs) for the last position, matching vllm-vulkan's
    ``per_layer`` (pre-final-norm) capture.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()
    ids = torch.tensor([list(token_ids)], dtype=torch.long)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = out.hidden_states  # tuple, len = num_layers + 1
    # Decoder-layer outputs (skip the embedding entry at index 0).
    layers = [hs[i][0, -1].float().tolist() for i in range(1, len(hs))]
    logits = out.logits[0, -1].float().tolist()
    return {"layers": layers, "logits": logits}


def _cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _max_abs_diff(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    return max((abs(a[i] - b[i]) for i in range(n)), default=0.0)


def _argmax(xs: list[float]) -> int:
    best_i, best_v = 0, xs[0]
    for i, v in enumerate(xs):
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def compare(
    vk: dict,
    ref: dict,
    logit_cosine: float = DEFAULT_LOGIT_COSINE,
    layer_cosine: float = DEFAULT_LAYER_COSINE,
) -> tuple[bool, dict]:
    """Compare two dumps.  Pass = greedy next-token match AND logit cosine high.

    Per-layer cosines are reported for diagnostics (to localise where any
    divergence begins) but the final-logits agreement is the pass criterion,
    since the per-layer index/final-norm convention is framework-dependent.
    """
    vk_logits, ref_logits = vk["logits"], ref["logits"]
    vk_tok, ref_tok = _argmax(vk_logits), _argmax(ref_logits)
    lcos = _cosine(vk_logits, ref_logits)
    lmad = _max_abs_diff(vk_logits, ref_logits)

    # Align the overlap of per-layer outputs (vk layer i ~ ref layer i).
    n_layers = min(len(vk["layers"]), len(ref["layers"]))
    layer_cos = [_cosine(vk["layers"][i], ref["layers"][i]) for i in range(n_layers)]
    worst_layer = min(range(n_layers), key=lambda i: layer_cos[i]) if n_layers else -1

    passed = (vk_tok == ref_tok) and (lcos >= logit_cosine)
    report = {
        "passed": passed,
        "next_token_vulkan": vk_tok,
        "next_token_reference": ref_tok,
        "next_token_match": vk_tok == ref_tok,
        "logit_cosine": lcos,
        "logit_max_abs_diff": lmad,
        "n_layers_compared": n_layers,
        "min_layer_cosine": min(layer_cos) if layer_cos else None,
        "worst_layer_index": worst_layer,
        "first_layer_below_threshold": next(
            (i for i, c in enumerate(layer_cos) if c < layer_cosine), None
        ),
    }
    return passed, report


def _print_report(report: dict) -> None:
    print("── Qwen3 parity (vllm-vulkan CPU ref vs HF transformers f32) ──")
    for k, v in report.items():
        print(f"  {k}: {v}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", help="Local HF Qwen3 model directory")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", help="Prompt text (tokenized with the model tokenizer)")
    g.add_argument("--tokens", help="Comma-separated token ids, e.g. 9707,11,1879")
    ap.add_argument("--logit-cosine", type=float, default=DEFAULT_LOGIT_COSINE)
    args = ap.parse_args(argv)

    if args.tokens:
        token_ids = [int(t) for t in args.tokens.split(",") if t.strip()]
    else:
        token_ids = tokenize_prompt(args.model_dir, args.prompt)
    print(f"tokens: {token_ids}")

    vk = dump_vulkan(args.model_dir, token_ids)
    ref = dump_reference_hf(args.model_dir, token_ids)
    passed, report = compare(vk, ref, logit_cosine=args.logit_cosine)
    _print_report(report)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
