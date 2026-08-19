# SPDX-License-Identifier: Apache-2.0
"""Gate for the SCOPED, OPT-IN trust_remote_code flag (default OFF).

Covers the security-critical resolver (`resolve_trust_remote_code`) branch-by-
branch, the `build_app` signature wiring, and the config-stub threading. The
resolver is the piece that decides whether model-repo Python is allowed to run,
so its default-OFF behavior and per-model scoping are asserted exhaustively.

The `config_stub` part runs anywhere (pure stdlib). The `app` parts import the
oai package, which pulls vllm at module load -> run in the base-vLLM venv:
    /opt/vllm-vulkan-venv/bin/python -m vllm_vulkan.oai.tests.test_trust_remote_code
"""
import inspect
import os
import sys

FAILS: list[str] = []


def expect(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


KIMI = "/mnt/nas/models/Kimi-Linear-48B-A3B"


def test_config_stub_threads_flag():
    """RustModelConfig default OFF; opt-in threads through (pure stdlib)."""
    from vllm_vulkan.oai.config_stub import RustModelConfig
    expect(RustModelConfig(KIMI).trust_remote_code is False,
           "config_stub: default trust_remote_code is False")
    expect(RustModelConfig(KIMI, trust_remote_code=True).trust_remote_code is True,
           "config_stub: trust_remote_code=True threads through")


def test_resolver_branches():
    from vllm_vulkan.oai.app import resolve_trust_remote_code as R
    os.environ.pop("VLLM_VULKAN_TRUST_REMOTE_CODE", None)
    expect(R(KIMI, cli_flag=False) is False, "resolver: default (no cli/env) -> OFF")
    expect(R(KIMI, cli_flag=True) is True, "resolver: CLI flag -> ON")
    for v in ("1", "true", "TRUE", "yes", "on"):
        os.environ["VLLM_VULKAN_TRUST_REMOTE_CODE"] = v
        expect(R(KIMI, cli_flag=False) is True, f"resolver: env blanket '{v}' -> ON")
    for v in (KIMI, "Kimi-Linear-48B-A3B", "x/y,Kimi-Linear-48B-A3B"):
        os.environ["VLLM_VULKAN_TRUST_REMOTE_CODE"] = v
        expect(R(KIMI, cli_flag=False) is True, f"resolver: scoped hit '{v}' -> ON")
    for v in ("Qwen/Qwen2.5-0.5B", "other-model"):
        os.environ["VLLM_VULKAN_TRUST_REMOTE_CODE"] = v
        expect(R(KIMI, cli_flag=False) is False, f"resolver: scoped miss '{v}' -> OFF")
    # allow-list for a different model must not leak
    os.environ["VLLM_VULKAN_TRUST_REMOTE_CODE"] = "Qwen/Qwen2.5-0.5B"
    expect(R(KIMI, cli_flag=False) is False, "resolver: other-model scope does NOT enable Kimi")
    os.environ.pop("VLLM_VULKAN_TRUST_REMOTE_CODE", None)


def test_build_app_signature():
    from vllm_vulkan.oai.app import build_app
    p = inspect.signature(build_app).parameters.get("trust_remote_code")
    expect(p is not None, "build_app: has a trust_remote_code parameter")
    expect(p is not None and p.default is False, "build_app: trust_remote_code default is False")


def main():
    test_config_stub_threads_flag()
    try:
        from vllm_vulkan.oai import app  # noqa: F401
    except Exception as e:  # vllm not present (CI-lite): resolver/app tests need the venv
        print(f"  SKIP app-dependent tests (vllm import failed: {e})")
    else:
        test_resolver_branches()
        test_build_app_signature()
    print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAIL: {FAILS}"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
