"""Optional build hook for MLB's UltraEP-derived placement extension.

The normal MLB installation remains pure Python/JIT. Set
``MLB_BUILD_ULTRAEP_PLACEMENT=1`` to build the SM90/SM100 placement and sparse
rerouting kernels. Expert communication is provided separately by UltraEP and
is deliberately not linked into this extension.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from setuptools import setup

_ROOT = Path(__file__).resolve().parent
_ULTRAEP_ROOT = _ROOT / "src" / "moe_load_balancer" / "kernels" / "ultraep"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _enabled() -> bool:
    return (
        os.environ.get("MLB_BUILD_ULTRAEP_PLACEMENT", "").strip().lower()
        in _TRUE_VALUES
    )


def _detect_arch() -> None:
    configured = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    requested = [token for token in re.split(r"[;,\s]+", configured) if token]
    supported = [
        token
        for token in requested
        if token.removesuffix("+PTX").lower() in {"9.0", "9.0a", "10.0", "10.0a"}
    ]
    if supported:
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(supported)
        return
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        check=True,
        text=True,
    )
    compute_capability = result.stdout.strip().splitlines()[0].strip()
    arch = {"9.0": "9.0", "10.0": "10.0"}.get(compute_capability)
    if arch is None:
        raise RuntimeError(
            "UltraEP placement supports SM90 and SM100 only; detected "
            f"compute capability {compute_capability}."
        )
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch


def _extension_config():
    if not _enabled():
        return [], {}

    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError as exc:
        raise RuntimeError(
            "MLB_BUILD_ULTRAEP_PLACEMENT=1 requires PyTorch in the build "
            "environment. Use pip install --no-build-isolation."
        ) from exc

    if not torch.version.cuda:
        raise RuntimeError("MLB_BUILD_ULTRAEP_PLACEMENT=1 requires a CUDA-enabled PyTorch build.")

    _detect_arch()
    cxx_flags = [
        "-O3",
        "-Wno-deprecated-declarations",
        "-Wno-unused-variable",
        "-Wno-sign-compare",
        "-Wno-attributes",
    ]
    nvcc_flags = [
        "-O3",
        "-Xcompiler",
        "-O3",
        "--ptxas-options=--register-usage-level=10",
    ]
    if os.environ.get("DISABLE_AGGRESSIVE_PTX_INSTRS", "1") != "0":
        cxx_flags.append("-DDISABLE_AGGRESSIVE_PTX_INSTRS")
        nvcc_flags.append("-DDISABLE_AGGRESSIVE_PTX_INSTRS")

    include_dirs = [str(_ULTRAEP_ROOT / "csrc")]
    if torch.version.cuda.startswith("13."):
        include_dirs.append("/usr/local/cuda/include/cccl/")
        # cusparse.h and friends: on some CUDA 13 images the math libraries
        # ship only as the pip nvidia-cu13 packages, not under the system
        # CUDA toolkit include path torch.utils.cpp_extension searches. That
        # same package directory also ships its own cuda_runtime.h /
        # crt/host_runtime.h, which must NOT shadow the system nvcc's
        # matching versions -- an -I (searched before system paths) picking
        # those up instead breaks nvcc's stub codegen (__cudaLaunch
        # macro-arity errors). -idirafter searches it only as a fallback for
        # headers the system toolkit is genuinely missing.
        site_packages = Path(torch.__file__).resolve().parent.parent
        cu13_include = site_packages / "nvidia" / "cu13" / "include"
        if cu13_include.is_dir():
            fallback_flag = f"-idirafter{cu13_include}"
            cxx_flags.append(fallback_flag)
            nvcc_flags.extend(["-Xcompiler", fallback_flag])

    sources = [
        _ULTRAEP_ROOT / "csrc" / "python_api.cpp",
        _ULTRAEP_ROOT / "csrc" / "kernels" / "placement.cu",
        _ULTRAEP_ROOT / "csrc" / "kernels" / "sparse_reroute.cu",
    ]
    extension = CUDAExtension(
        name="moe_load_balancer.kernels.ultraep._C",
        sources=[str(path.relative_to(_ROOT)) for path in sources],
        include_dirs=include_dirs,
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
    )
    return [extension], {"build_ext": BuildExtension}


ext_modules, cmdclass = _extension_config()
setup(ext_modules=ext_modules, cmdclass=cmdclass)
