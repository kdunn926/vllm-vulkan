# SPDX-License-Identifier: Apache-2.0
"""Duck-typed ModelConfig / VllmConfig stubs for the base-vLLM OpenAI frontend.

Constructing a REAL ``vllm.config.ModelConfig`` is the one place that can pull
real torch (dtype/device resolution) — so we deliberately do NOT. This is the
*minimal* attribute surface the REAL ``HfRenderer`` / ``OnlineRenderer`` and the
OpenAI serving handlers actually read (verified end-to-end against vLLM 0.26.0;
see oai/README.md). Nothing here executes a torch op.

CRITICAL GATE: ``is_multimodal_model = False``. That single flag keeps vLLM's
renderer on its text-only path; a True would enter the multimodal branch that
runs real torch (image/video preprocessing). For multimodal, run the vision
encoder in Rust and keep this False (see README.md).

This is version-coupled to vLLM's internal renderer/serving surface (NOT public
API). Pin the vLLM version; re-run the validation gate on any bump.
"""
from __future__ import annotations


class _HFConfig:
    """Stand-in for ``model_config.hf_config`` / ``hf_text_config``. Only
    ``model_type`` is read by the chat serving layer (must be a real str and
    != "gpt_oss")."""

    def __init__(self, model_type: str = "qwen2"):
        self.model_type = model_type


class RustModelConfig:
    """Minimal ModelConfig the real renderer + OpenAI serving layer read."""

    def __init__(
        self,
        model: str,
        *,
        served_model_name: str | None = None,
        max_model_len: int = 8192,
        model_type: str = "qwen2",
        tokenizer: str | None = None,
        trust_remote_code: bool = False,
    ):
        # identity / tokenizer resolution (vllm/tokenizers/registry.py)
        self.model = model
        self.tokenizer = tokenizer or model
        self.served_model_name = served_model_name or model
        self.tokenizer_mode = "hf"
        self.tokenizer_revision = None
        # opt-in (default OFF); build_app threads the resolved flag here so any
        # downstream renderer/registry read matches the tokenizer load.
        self.trust_remote_code = trust_remote_code
        self.runner_type = "generate"
        self.skip_tokenizer_init = False
        # gating flags that keep torch / multimodal branches OFF
        self.is_encoder_decoder = False
        self.is_multimodal_model = False        # <-- the critical torch gate
        self.enable_prompt_embeds = False
        # values read directly by renderer / serving
        self.max_model_len = max_model_len
        self.renderer_num_workers = 1           # sizes the tokenizer thread pool
        self.hf_config = _HFConfig(model_type)
        self.hf_text_config = _HFConfig(model_type)
        self.multimodal_config = None
        self.encoder_config = None              # renderer default_chat_tok_params
        self.allowed_local_media_path = None    # chat_utils media parser
        self.allowed_media_domains = None
        self.lora_config = None
        # serving layer reads these for default sampling / generation config
        self.generation_config = "auto"
        self.override_generation_config = {}

    def get_diff_sampling_param(self) -> dict:
        return {}

    def get_multimodal_config(self):
        return None


class _ParallelConfig:
    # BaseRenderer.__init__ reads _api_process_rank; _api_process_count is only
    # used in the (disabled) multimodal branch.
    _api_process_rank = 0
    _api_process_count = 1


class RustVllmConfig:
    """Minimal VllmConfig: model_config + parallel_config + kv_transfer_config."""

    def __init__(self, model_config: RustModelConfig):
        self.model_config = model_config
        self.parallel_config = _ParallelConfig()
        self.kv_transfer_config = None  # read by serving get_system_fingerprint
