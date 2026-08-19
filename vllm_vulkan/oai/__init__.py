# SPDX-License-Identifier: Apache-2.0
"""Base-vLLM OpenAI server frontend on top of the Vulkan Rust backend.

Uses vLLM 0.26.0's battle-tested OpenAI API server (request validation, chat
templating, streaming, OpenAI-compatible responses) with EVERYTHING torch-heavy
removed: generation runs in Rust, and a stub `torch` (installed into the venv,
not imported here) satisfies vLLM's import-time torch references. See README.md.

Public API:
    build_app(model, backend_factory=None, ...) -> FastAPI
    serve(model, backend_factory=None, host=..., port=...) -> None
    RustEngineClient, RustBackend, DummyRustBackend, VulkanRustBackend
    RustModelConfig, RustVllmConfig
"""
from .app import build_app, serve
from .backend import DummyRustBackend, RustBackend, Step, VulkanRustBackend
from .config_stub import RustModelConfig, RustVllmConfig
from .engine_client import RustEngineClient

__all__ = [
    "build_app", "serve",
    "RustEngineClient", "RustBackend", "DummyRustBackend", "VulkanRustBackend", "Step",
    "RustModelConfig", "RustVllmConfig",
]
