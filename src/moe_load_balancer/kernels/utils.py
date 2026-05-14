# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""JIT-kernel build helpers (vendored subset from ``sglang.jit_kernel.utils``).

This module hosts the small set of helpers the LPLB CUDA kernels (and any
future SDK-owned JIT kernels) need to compile through ``tvm-ffi``:

- ``cache_once``: simple memoization decorator (compatible with ``torch.compile``).
- ``make_cpp_args`` + ``CPPArgList``: C++ template argument formatting.
- ``ArchInfo`` + ``get_jit_cuda_arch`` + ``override_jit_cuda_arch``: CUDA arch
  detection.
- ``load_jit``: thin wrapper around ``tvm_ffi.cpp.load`` / ``load_inline``
  that injects the SDK's default include / cflag / target-arch setup.
- ``register_dependency`` + ``get_mathdx_include_paths`` +
  ``get_sgl_kernel_include_paths``: resolution of external header libraries
  needed by LPLB's ``.cuh`` files.

The SDK does **not** import any inference framework here. ``sgl_kernel``
headers are resolved by inspecting the installed ``sglang`` package at JIT
compile time; users must install ``sglang`` for the LPLB kernel build to
succeed.
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
import pathlib
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
)

import torch

if TYPE_CHECKING:
    from tvm_ffi import Module

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def cache_once(fn: F) -> F:
    """Memoize a function once per (args, kwargs) tuple.

    ``functools.lru_cache`` is incompatible with ``torch.compile``; this
    plain dict-backed cache mirrors the behavior used by the upstream
    SGLang JIT kernel utilities.
    """
    result_map: Dict[Tuple[Any, ...], Any] = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in result_map:
            result_map[key] = fn(*args, **kwargs)
        return result_map[key]

    return wrapper  # type: ignore[return-value]


def _make_wrapper(tup: Tuple[str, str]) -> str:
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"


@cache_once
def _resolve_kernel_path() -> pathlib.Path:
    """Locate the SDK's kernel sources directory.

    Layout assumed:

        moe_load_balancer/kernels/
        ├── csrc/
        │   └── <algo>/*.cuh         # CUDA sources, included via #include "<path>"
        └── utils.py                 # this file

    Unlike SGLang's ``jit_kernel/`` we do not require a sibling ``include/``
    folder — SDK ``.cuh`` files pull framework headers (e.g. ``sgl_kernel``)
    through registered dependencies instead.

    Existence of ``csrc/`` is not validated here: it can be empty until an
    algorithm subpackage is added. ``load_jit`` surfaces the missing-file
    error at compile time.
    """
    return pathlib.Path(__file__).parent.resolve()


KERNEL_PATH = _resolve_kernel_path()
DEFAULT_CFLAGS = ["-std=c++20", "-O3"]
DEFAULT_LDFLAGS: List[str] = []
CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, str, bool, torch.dtype]


class CPPArgList(list[str]):
    def __str__(self) -> str:
        return ", ".join(self)


CPP_DTYPE_MAP = {
    torch.float: "fp32_t",
    torch.float16: "fp16_t",
    torch.float8_e4m3fn: "fp8_e4m3_t",
    torch.bfloat16: "bf16_t",
    torch.int8: "int8_t",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
}


@cache_once
def is_hip_runtime() -> bool:
    return bool(torch.version.hip)


def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CPPArgList:
    """Render Python scalars into a C++ template argument list.

    Used to key JIT-compiled kernels by their template parameters.
    """

    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, str, float)):
            return str(arg)
        if isinstance(arg, torch.dtype):
            return CPP_DTYPE_MAP[arg]
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")

    return CPPArgList(_convert(arg) for arg in args)


@dataclass
class ArchInfo:
    major: int
    minor: int
    suffix: str

    @property
    def target_name(self) -> str:
        return f"{self.major}.{self.minor}{self.suffix}"

    @property
    def jit_flag(self) -> str:
        # Matches sglang's SGL_CUDA_ARCH define so vendored .cuh files
        # depending on it keep working unchanged.
        return f"-DSGL_CUDA_ARCH={self.major * 100 + self.minor * 10}"


_CUDA_ARCH: Optional[ArchInfo] = None


@cache_once
def _init_jit_cuda_arch_once() -> None:
    global _CUDA_ARCH
    try:
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
    except Exception:
        logger.warning("Cannot detect CUDA architecture.")
        major, minor = 0, 0  # invalid value to trigger compile error if used
    _CUDA_ARCH = ArchInfo(major, minor, "")


def get_jit_cuda_arch() -> ArchInfo:
    """Return the current CUDA architecture info."""
    _init_jit_cuda_arch_once()
    assert _CUDA_ARCH is not None
    return _CUDA_ARCH


@contextmanager
def override_jit_cuda_arch(major: int, minor: int, suffix: str = ""):
    """Temporarily override the detected CUDA architecture (mostly for tests)."""
    global _CUDA_ARCH
    old_value = get_jit_cuda_arch()
    _CUDA_ARCH = ArchInfo(major, minor, suffix)
    try:
        yield
    finally:
        _CUDA_ARCH = old_value


@contextmanager
def _jit_compile_context():
    if is_hip_runtime():
        yield  # TODO: ROCm `TVM_FFI_ROCM_ARCH_LIST` support if needed.
        return
    env_key = "TVM_FFI_CUDA_ARCH_LIST"
    old_value = os.environ.get(env_key, None)
    os.environ[env_key] = get_jit_cuda_arch().target_name
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = old_value


def _get_default_target_flags() -> List[str]:
    if is_hip_runtime():
        return ["-DUSE_ROCM", "-std=c++20", "-O3"]
    return [
        get_jit_cuda_arch().jit_flag,
        "-std=c++20",
        "-O3",
        "--expt-relaxed-constexpr",
    ]


def load_jit(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    extra_dependencies: List[str] | None = None,
    build_directory: str | None = None,
    header_only: bool = True,
) -> Module:
    """Load a JIT module from C++/CUDA source files under ``kernels/csrc/``.

    Mirrors ``sglang.jit_kernel.utils.load_jit`` but resolves sources against
    the SDK's own ``KERNEL_PATH`` and emits SDK-prefixed module names. See
    that function for full parameter semantics.
    """

    from tvm_ffi.cpp import load, load_inline

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    cpp_files = [str((KERNEL_PATH / "csrc" / f).resolve()) for f in cpp_files]
    cuda_files = [str((KERNEL_PATH / "csrc" / f).resolve()) for f in cuda_files]

    for dep in set(extra_dependencies or []):
        if dep not in _REGISTERED_DEPENDENCIES:
            raise ValueError(f"Dependency {dep!r} is not registered.")
        extra_include_paths += _REGISTERED_DEPENDENCIES[dep]()

    module_name = "moe_lb_jit_" + "_".join(str(arg) for arg in args)
    if header_only:
        cpp_wrappers = cpp_wrappers or []
        cuda_wrappers = cuda_wrappers or []
        cpp_sources = [f'#include "{path}"' for path in cpp_files]
        cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]

        cuda_sources = [f'#include "{path}"' for path in cuda_files]
        cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]
        with _jit_compile_context():
            return load_inline(
                module_name,
                cpp_sources=cpp_sources,
                cuda_sources=cuda_sources,
                extra_cflags=DEFAULT_CFLAGS + extra_cflags,
                extra_cuda_cflags=_get_default_target_flags() + extra_cuda_cflags,
                extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
                extra_include_paths=extra_include_paths,
                build_directory=build_directory,
            )
    assert cpp_wrappers is None and cuda_wrappers is None
    with _jit_compile_context():
        return load(
            module_name,
            cpp_files=cpp_files,
            cuda_files=cuda_files,
            extra_cflags=DEFAULT_CFLAGS + extra_cflags,
            extra_cuda_cflags=_get_default_target_flags() + extra_cuda_cflags,
            extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
            extra_include_paths=extra_include_paths,
            build_directory=build_directory,
        )


def _find_package_root(package: str) -> Optional[pathlib.Path]:
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ModuleNotFoundError, ValueError):
        # find_spec raises (not returns None) when a parent package along
        # the dotted path is missing — e.g. find_spec("nvidia.mathdx") with
        # no "nvidia" namespace package installed. Treat any such failure
        # as "package absent" so dependency resolvers can give a clean
        # descriptive RuntimeError downstream.
        return None
    if spec is None or spec.origin is None:
        return None
    return pathlib.Path(spec.origin).resolve().parent


_REGISTERED_DEPENDENCIES: Dict[str, Callable[[], List[str]]] = {}


def register_dependency(name: str):
    """Register a build-time include-path resolver under ``name``.

    ``load_jit(..., extra_dependencies=[name])`` looks up the resolver and
    appends the returned paths to the kernel's include list. Resolvers must
    not import the framework module they describe; they should only consult
    ``importlib.util.find_spec`` and the file system.
    """

    def decorator(f: Callable[[], List[str]]) -> Callable[[], List[str]]:
        if name in _REGISTERED_DEPENDENCIES:
            raise ValueError(f"Dependency {name!r} already registered")
        _REGISTERED_DEPENDENCIES[name] = f
        return f

    return decorator


def get_mathdx_root() -> Optional[pathlib.Path]:
    """Locate the NVIDIA Math-DX install (cuBLASDx + cuSolverDx).

    Searches, in order:
      1. ``$MATHDX_HOME`` env var (extracted Math-DX archive root).
      2. The ``nvidia-mathdx`` PyPI package, if installed.
      3. The bundled ``mathdx`` resource folder under ``kernels/lplb/`` —
         used by ``download-mathdx.sh`` so cluster deploys can drop the
         archive next to the source tree.
    """
    env_home = os.environ.get("MATHDX_HOME")
    if env_home:
        candidate = pathlib.Path(env_home).expanduser().resolve()
        if (candidate / "include").exists():
            return candidate

    pkg = _find_package_root("nvidia.mathdx")
    if pkg is not None and (pkg / "include").exists():
        return pkg

    bundled = KERNEL_PATH / "lplb" / "resources" / "mathdx"
    if (bundled / "include").exists():
        return bundled.resolve()

    return None


@register_dependency("mathdx")
def get_mathdx_include_paths() -> List[str]:
    root = get_mathdx_root()
    if root is None:
        raise RuntimeError(
            "Cannot find NVIDIA Math-DX (cuBLASDx + cuSolverDx) headers. "
            "Set MATHDX_HOME to the extracted Math-DX archive, install the "
            "`nvidia-mathdx` package, or run "
            "src/moe_load_balancer/kernels/lplb/resources/download-mathdx.sh."
        )
    candidates = [root / "include"]
    cutlass = root / "external" / "cutlass" / "include"
    if cutlass.exists():
        candidates.append(cutlass)
    return [str(p) for p in candidates]


@register_dependency("sgl_kernel")
def get_sgl_kernel_include_paths() -> List[str]:
    """Resolve SGLang's ``sgl_kernel/*`` C++ header library.

    LPLB ``.cuh`` files include headers like ``<sgl_kernel/tensor.h>`` for
    ``tvm-ffi`` tensor binding helpers. These ship inside the installed
    ``sglang`` package under ``jit_kernel/include/``. The SDK looks them up
    here at JIT compile time only; no Python import of ``sglang`` happens.

    Raises descriptively if ``sglang`` is not installed in the active
    environment.
    """
    sglang_root = _find_package_root("sglang")
    if sglang_root is None:
        raise RuntimeError(
            "Cannot find the `sglang` package, which provides the "
            "sgl_kernel C++ header library required to compile LPLB "
            "kernels. Install sglang in the build environment (typically "
            "inside the lmsysorg/sglang:dev container)."
        )
    include = sglang_root / "jit_kernel" / "include"
    if not include.exists():
        raise RuntimeError(
            f"Expected sgl_kernel headers at {include}, but the directory "
            "does not exist. Was sglang installed with its jit_kernel "
            "sources included?"
        )
    return [str(include.resolve())]


__all__ = [
    "ArchInfo",
    "CPPArgList",
    "CPP_DTYPE_MAP",
    "DEFAULT_CFLAGS",
    "DEFAULT_LDFLAGS",
    "KERNEL_PATH",
    "cache_once",
    "get_jit_cuda_arch",
    "get_mathdx_include_paths",
    "get_mathdx_root",
    "get_sgl_kernel_include_paths",
    "is_hip_runtime",
    "load_jit",
    "make_cpp_args",
    "override_jit_cuda_arch",
    "register_dependency",
]
