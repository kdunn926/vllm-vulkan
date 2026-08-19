# SPDX-License-Identifier: Apache-2.0
"""Import-satisfaction stub for `torch`, for the base-vLLM OpenAI frontend.

This file is INSTALLED AS the top-level ``torch`` package
(``site-packages/torch/__init__.py``) inside the base-vLLM venv by the image
build recipe (gentoo-pxe ``extras/build-vllm-chroot.sh``). It is NOT imported by
vllm_vulkan at runtime — it only exists so that base vLLM 0.26.0's OpenAI server
frontend can be imported and served without the real ~200 MB CPU torch wheel.

Why this is safe (proven across three de-risks, see oai/README.md):
  * The Vulkan backend does ALL generation in Rust (`vllm_vulkan._rs`). The
    frontend's job is prompt rendering, detokenization, stop handling and OpenAI
    response assembly — all torch-free (HF `tokenizers` + Python string ops).
  * vLLM's engine/sampler (the torch-heavy part) is never instantiated: we supply
    our own `RustEngineClient` (see engine_client.py), bypassing `AsyncLLM`.
  * So real torch is only *referenced* at import time (annotations, dtype
    constants, `torch.library` custom-op registration) — never *executed* on the
    serve path. This stub satisfies those references and nothing more.

Condition: text-only models. A multimodal model would re-enter vLLM's renderer
MM branch, which runs real torch (image preprocessing) — handle vision in Rust
and keep `is_multimodal_model=False` (see config_stub.py / README.md).

Mechanism: a meta-path finder fabricates any ``torch.*`` (and guided-decoding
roots that would otherwise pull real torch) submodule as a permissive package;
every attribute resolves to a cached permissive class that is usable as a base
class / annotation / dtype, is coercible to int/float/str, and exposes a pydantic
"any" schema. Dunders are NOT fabricated (libraries probe them). dtype constants
carry a real dotted ``__str__`` (``"torch.bfloat16"``) so transformers' config
repr (`str(config.dtype).split(".")[1]`) works with no monkeypatch.
"""
import sys as _sys
import types as _types
import importlib.abc as _iabc
import importlib.machinery as _imach

__version__ = "2.11.0"
_BIG = (1 << 63) - 1


def _is_dunder(n: str) -> bool:
    return len(n) > 4 and n.startswith("__") and n.endswith("__")


class _AnyMeta(type):
    """Metaclass: any (non-dunder) attribute of a stub class is another stub
    class, cached for stable identity; instances are permissive; the class is
    int-coercible and advertises a pydantic 'any' schema."""

    def __getattr__(cls, n):
        if _is_dunder(n):
            raise AttributeError(n)
        v = _AnyMeta(n, (_Any,), {})
        setattr(cls, n, v)
        return v

    def __call__(cls, *a, **k):
        return _Any.__new__(_Any)

    def __get_pydantic_core_schema__(cls, source, handler):
        # torch-typed pydantic model fields -> Any (don't misfire as a real type)
        return {"type": "any"}

    def __int__(cls):
        return _BIG

    def __index__(cls):
        return _BIG


class _Any(metaclass=_AnyMeta):
    """Permissive instance: any attribute/call returns another permissive
    instance; coercible to int/float/str/bool; absorbs string concatenation
    (so vLLM's import-time `op_name + torch.library.infer_schema(...)` passes)."""

    def __getattr__(self, n):
        if _is_dunder(n):
            raise AttributeError(n)
        return _Any.__new__(_Any)

    def __call__(self, *a, **k):
        return _Any.__new__(_Any)

    def __add__(self, o):
        return o if isinstance(o, str) else _Any.__new__(_Any)

    def __radd__(self, o):
        return o if isinstance(o, str) else _Any.__new__(_Any)

    def __str__(self):
        return ""

    def __int__(self):
        return _BIG

    def __index__(self):
        return _BIG

    def __float__(self):
        return 0.0

    def __bool__(self):
        return False

    def __len__(self):
        return 0

    def __iter__(self):
        return iter(())


class _M(_types.ModuleType):
    """Permissive fabricated submodule."""

    def __getattr__(self, n):
        if _is_dunder(n):
            raise AttributeError(n)
        v = _AnyMeta(n, (_Any,), {})
        setattr(self, n, v)
        return v


# Roots fabricated as permissive stubs. torch.* plus guided-decoding backends
# that would otherwise pull REAL torch (xgrammar/outlines/...). vLLM feature-
# detects these via importlib.util.find_spec and resolves them lazily, so a
# permissive stub keeps `import vllm.entrypoints.openai.api_server` working while
# the actual guided-decoding path is simply unused (Rust owns generation).
_ROOTS = (
    "torch", "llguidance", "xgrammar", "outlines", "outlines_core",
    "lm_format_enforcer", "lark", "flashinfer", "gguf",
)


class _Finder(_iabc.MetaPathFinder, _iabc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _ROOTS:
            sp = _imach.ModuleSpec(name, self)
            sp.submodule_search_locations = []  # mark as package (nested imports)
            return sp
        return None

    def create_module(self, spec):
        return _M(spec.name)

    def exec_module(self, module):
        pass


_sys.meta_path.insert(0, _Finder())


# torch symbols referenced by name on the frontend chain.
class Tensor(metaclass=_AnyMeta):
    ...


class dtype:
    """Plain class (NOT _AnyMeta) so instances construct normally and carry a
    real dotted __str__ that transformers' `dict_dtype_to_str` splits on."""

    def __init__(self, name="float32"):
        self._name = name

    def __str__(self):
        return "torch." + self._name

    __repr__ = __str__


class device:
    def __init__(self, *a, **k):
        ...


float32 = dtype("float32")
float16 = dtype("float16")
bfloat16 = dtype("bfloat16")
float64 = dtype("float64")
int64 = dtype("int64")
int32 = dtype("int32")
int16 = dtype("int16")
int8 = dtype("int8")
uint8 = dtype("uint8")
bool_ = dtype("bool")

# Make the module itself permissive for any other top-level torch.<attr>.
_sys.modules[__name__].__class__ = _M
