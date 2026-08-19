# `vllm_vulkan.oai` — base-vLLM OpenAI server on the Vulkan Rust backend

Use vLLM 0.26.0's **battle-tested OpenAI API server** (request validation, chat
templating, streaming SSE, OpenAI-compatible responses, detokenization, stop
handling) while running **all generation in Rust** (`vllm_vulkan._rs`) and
**stubbing out torch entirely** — no ~200 MB CPU torch, no CUDA kernels.

This is distinct from `vllm_vulkan.server` (the existing standalone hand-rolled
server) and from the `vllm_vulkan` **platform plugin** (which runs *inside*
vLLM's torch CPU engine — the thing we deliberately bypass here).

## Architecture

After validation, the Rust backend's entire contract reduces to:

```
(prompt_token_ids, SamplingParams)  ->  async stream of sampled token ids
```

Everything else runs **torch-free in Python, reusing vLLM's own code**:

```
HTTP request
  -> vLLM OpenAI router + protocol (validate)
  -> REAL OnlineRenderer/HfRenderer      (chat template + tokenize -> prompt ids)
  -> RustEngineClient.generate()
       -> RustBackend.stream(prompt_ids, sampling_params)   # Rust: yields token ids
       -> vLLM IncrementalDetokenizer   (ids -> text, byte-BPE/UTF-8 correct)
       -> vLLM check_stop / Request      (stop-tokens, EOS, length, min_tokens)
       -> stop-STRING trimming           (in the detokenizer)
       -> RequestOutput (honors output_kind: DELTA vs FINAL_ONLY)
  -> OpenAI response assembly + SSE
```

The Rust side needs **no tokenizer, no detokenizer, no stop logic** — which
removes token-level drift risk (it's vLLM's own, tested code).

## The Rust binding (now included) + what's left

The Rust decode bindings — `forward_and_sample` (fast path) and `prefill_logits`
(logprobs path) on `_rs.VulkanModel` — are included from
`feat/engineclient-serving-bindings`. `backend.py :: VulkanRustBackend` drives
them with the documented single-stream loop (`reset_kv_cache` -> `forward`
prefill -> `forward_and_sample`), and `__main__.py` loads the model and wires it:

```bash
python -m vllm_vulkan.oai Qwen/Qwen2.5-0.5B-Instruct --port 8000
```

That is **launch-ready**; the ONE remaining task is **hardware validation** —
confirm the GPU decode loop token-for-token against `python -m vllm_vulkan.server`
on a node (the Python serving path is already gate-validated torch-free; only the
real `_rs` decode is untested off-hardware).

For GPU-free CI/tests, `build_app(model)` (default `DummyRustBackend`) returns
"Hello world" over the full real serving stack — the validation gate.

## torch stub — why it's safe, and the one condition

`_torch_stub.py` is installed as the venv's top-level `torch`. It satisfies
vLLM's *import-time* torch references (annotations, dtype constants,
`torch.library` custom-op registration) but no torch op ever *runs* on the serve
path (proven: correct responses under a fake torch = no real torch executed).

**Condition: text-only models.** `is_multimodal_model=False` (in `config_stub.py`)
keeps vLLM's renderer on its text path. A multimodal model re-enters the renderer's
MM branch, which runs real torch (image/video preprocessing). To do multimodal,
run the vision encoder in Rust and keep this flag False.

## Footprint

Into the shared `/opt/vllm-vulkan-venv`: **~+130–150 MB uncompressed / ~+70–90 MB
image** (zstd). torch is stubbed (~20 KB, saved ~200–260 MB); vLLM's 571 MB of
compiled CUDA/flash-attn kernels are pruned; only the frontend Python + a slim
no-torch dep set is installed.

## Validation gate (must stay green)

Run in the base-vLLM venv (with the stub torch), tokenizer in the HF cache:

```
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /opt/vllm-vulkan-venv/bin/python -m vllm_vulkan.oai.tests.test_e2e
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /opt/vllm-vulkan-venv/bin/python -m vllm_vulkan.oai.tests.test_stop_gate
```

`test_e2e` — real OpenAI server + real renderer + DummyRustBackend over a real
httpx ASGI round-trip (chat+completions, streaming+non-streaming).
`test_stop_gate` — drives `RustEngineClient.generate()` through scripted token
streams: stop-string across tokens (trim/keep), stop_token_ids, length cap,
DELTA==FINAL concatenation, byte-BPE/UTF-8 round-trip.

## ⚠ Version coupling

This reuses vLLM `v1.*` **internals** (`IncrementalDetokenizer`, `check_stop`,
`Request`, `EngineCoreRequest`, `RequestStatus`, the `renderers.*` + per-endpoint
`entrypoints.openai.*` packages) — NOT public API. **Pin the vLLM version** and
re-run the gate on any bump. The venv build (gentoo-pxe
`extras/build-vllm-chroot.sh`, `WITH_BASE_VLLM=1`) pins `vllm==0.26.0`.

## Open follow-up

Per-token **logprobs** are currently `None` (the sampler is in Rust). To serve
`logprobs`, have `RustBackend.Step` carry the chosen token's logprob (+ optional
top-k) and wrap it as `vllm.logprobs.Logprob` in `engine_client._request_output`.
Text and finish_reason are fully handled.
