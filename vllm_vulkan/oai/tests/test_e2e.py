# SPDX-License-Identifier: Apache-2.0
"""E2E validation gate: real vLLM OpenAI server + real renderer + RustEngineClient
+ DummyRustBackend, over a real httpx ASGI round-trip (TestClient). No GPU.

Run inside the BASE-vLLM venv (with the stub torch installed), NOT a plain env:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      /opt/vllm-vulkan-venv/bin/python -m vllm_vulkan.oai.tests.test_e2e

Requires the tokenizer for OAI_TEST_MODEL (default Qwen2.5-0.5B-Instruct) to be
in the HF cache (fetch once online: `AutoTokenizer.from_pretrained(model)`).
Token-count assertions are specific to the Qwen2.5 tokenizer.
"""
import json
import os
import sys
import traceback

MODEL = os.environ.get("OAI_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def main():
    from fastapi.testclient import TestClient
    from vllm_vulkan.oai import build_app

    app = build_app(MODEL, served_model_name="dummy-model", model_type="qwen2")
    c = TestClient(app)

    def sse(payload, url):
        with c.stream("POST", url, json=payload) as r:
            return [ln[6:] for ln in r.iter_lines() if ln and ln.startswith("data: ")]

    # ---- completions ----
    r = c.post("/v1/completions", json={"model": "dummy-model", "prompt": "Say hello",
                                        "max_tokens": 8, "temperature": 0.0})
    b = r.json()
    assert r.status_code == 200, b
    assert b["choices"][0]["text"] == "Hello world", b
    assert b["usage"]["prompt_tokens"] == 2, b          # "Say hello" -> [45764, 23811]
    assert b["usage"]["completion_tokens"] == 2, b
    print(">>> completion non-stream PASS")

    ch = sse({"model": "dummy-model", "prompt": "Say hello", "max_tokens": 8,
              "stream": True, "stream_options": {"include_usage": True}}, "/v1/completions")
    assert ch[-1] == "[DONE]"
    txt = "".join(json.loads(x)["choices"][0]["text"] for x in ch[:-1] if json.loads(x).get("choices"))
    assert txt == "Hello world", ch
    print(">>> completion stream PASS")

    # ---- chat (real chat template) ----
    r = c.post("/v1/chat/completions", json={"model": "dummy-model",
               "messages": [{"role": "user", "content": "Say hello"}],
               "max_tokens": 8, "temperature": 0.0})
    b = r.json()
    assert r.status_code == 200, b
    assert b["choices"][0]["message"]["content"] == "Hello world", b
    assert b["choices"][0]["message"]["role"] == "assistant", b
    assert b["usage"]["prompt_tokens"] == 31, b         # full Qwen chat template
    print(">>> chat non-stream PASS (prompt_tokens=31 from REAL chat template)")

    ch = sse({"model": "dummy-model", "messages": [{"role": "user", "content": "Say hello"}],
              "max_tokens": 8, "stream": True, "stream_options": {"include_usage": True}},
             "/v1/chat/completions")
    assert ch[-1] == "[DONE]"
    content = role = None
    content = ""
    for x in ch[:-1]:
        o = json.loads(x)
        if o.get("choices"):
            d = o["choices"][0].get("delta", {})
            role = d.get("role") or role
            content += d.get("content") or ""
    assert content == "Hello world", ch
    assert role == "assistant"
    print(">>> chat stream PASS")

    print("\nALL E2E TESTS PASSED")


if __name__ == "__main__":
    rc = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    os._exit(rc)
