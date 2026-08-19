# SPDX-License-Identifier: Apache-2.0
"""Launch the base-vLLM OpenAI server on the real Vulkan Rust backend.

    python -m vllm_vulkan.oai <model> [--port 8000] [--host 0.0.0.0]
                              [--max-seq-len N] [--device-idx 0]

Loads the Vulkan model (reusing vllm_vulkan.server.find_safetensors +
vllm_vulkan._rs.VulkanModel) and serves it through base vLLM's OpenAI API via
RustEngineClient + VulkanRustBackend (the Rust decode bindings forward_and_sample
/ prefill_logits from feat/engineclient-serving-bindings).

STEP 5 / validate-on-hardware: everything below the `_rs.VulkanModel(...)` call
runs real GPU decode. The Python serving path (render/detok/stop/response) is
gate-validated torch-free (see oai/tests/); this launcher wires it to the real
model. Confirm token-for-token against `python -m vllm_vulkan.server` on a node
before relying on it. Without a GPU/_rs (e.g. CI), use `build_app(model)` with the
default DummyRustBackend instead.
"""
from __future__ import annotations

import argparse

from .app import serve
from .backend import VulkanRustBackend


def _load_vulkan_backend_factory(model: str, max_seq_len: int, device_idx: int):
    """Return a backend_factory(tokenizer) -> VulkanRustBackend that loads the
    Vulkan model ONCE (single-stream MVP, max_num_seqs=1)."""
    def factory(_tokenizer):
        from vllm_vulkan.server import find_safetensors
        from vllm_vulkan._rs import VulkanModel
        st_path = find_safetensors(model)
        vk_model = VulkanModel(st_path, max_seq_len=max_seq_len, device_idx=device_idx)
        return VulkanRustBackend(vk_model)
    return factory


def main() -> None:
    ap = argparse.ArgumentParser(prog="vllm_vulkan.oai")
    ap.add_argument("model", help="HF model id or local path (weights + tokenizer)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--device-idx", type=int, default=0)
    ap.add_argument("--served-model-name", default=None)
    ap.add_argument("--model-type", default="qwen2",
                    help="hf_config.model_type for the config stub (must != gpt_oss)")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="OPT-IN (default OFF): allow the model repo's custom tokenizer "
                         "code to execute during tokenizer load (needed for models like "
                         "Kimi with a TikToken tokenizer). Only for models you TRUST. Can "
                         "also be scoped per-model via VLLM_VULKAN_TRUST_REMOTE_CODE.")
    args = ap.parse_args()

    serve(
        args.model,
        backend_factory=_load_vulkan_backend_factory(args.model, args.max_seq_len, args.device_idx),
        host=args.host, port=args.port,
        served_model_name=args.served_model_name,
        max_model_len=args.max_seq_len,
        model_type=args.model_type,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
