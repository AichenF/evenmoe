"""SGLang execution backend for MLB's framework-neutral full LPLB policy."""

from __future__ import annotations

from functools import lru_cache
import importlib.util
import os
from pathlib import Path

import torch
from torch import Tensor


_BLOCK_DIM = 256
_NUM_ITERATIONS = 5
_HOPPER_SHMEM_BUDGET = 223 * 1024


def _shared_memory_bytes(nc: int, nv: int) -> int:
    # Must match ipm_smem in policies/l2/csrc/lplb_ipm.cuh.
    # Four NV vectors are present (c, x, r, d); the final two words cover
    # alpha plus the aligned bool flag.
    return 4 * (2 * nc * nv + nc * nc + 4 * nv + 2 * nc + 2)


def _kernel_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "csrc"
        / "lplb_ipm_sglang.cuh"
    )


def _mathdx_include_paths() -> list[str]:
    root = os.environ.get("MATHDX_HOME")
    candidates = []
    if root:
        candidates.append(Path(root).expanduser().resolve())
    spec = importlib.util.find_spec("nvidia.mathdx")
    if spec is not None:
        candidates.extend(Path(path).resolve() for path in spec.submodule_search_locations or [])
    for candidate in candidates:
        include = candidate / "include"
        if include.is_dir():
            paths = [str(include)]
            cutlass = candidate / "external" / "cutlass" / "include"
            if cutlass.is_dir():
                paths.append(str(cutlass))
            return paths
    raise RuntimeError(
        "Cannot find NVIDIA Math-DX headers; install nvidia-mathdx or set MATHDX_HOME."
    )


@lru_cache(maxsize=None)
def _load_ipm_module(nc: int, nv: int, sm_version: int):
    try:
        from sglang.kernels.jit.utils import load_jit, make_cpp_args

        dependency_kwargs = {"extra_dependencies": ["mathdx"]}
    except ImportError:
        # Compatibility with the older SGLang JIT package used by the MLB fork.
        from sglang.jit_kernel.utils import load_jit, make_cpp_args

        dependency_kwargs = {"extra_include_paths": _mathdx_include_paths()}

    arguments = make_cpp_args(
        nc,
        nv,
        _BLOCK_DIM,
        sm_version,
        _NUM_ITERATIONS,
    )
    return load_jit(
        "mlb_lplb_ipm",
        *arguments,
        cuda_files=[str(_kernel_path())],
        cuda_wrappers=[("ipm_solve", f"ipm_solve<{arguments}>")],
        **dependency_kwargs,
    )


class SGLangFusedIPMBackend:
    """Compile MLB's IPM kernel through SGLang's generic JIT infrastructure."""

    def __init__(self) -> None:
        self._warmed_shapes: set[tuple[int, int, int, int]] = set()

    @staticmethod
    def _sm_version(device: torch.device) -> int:
        major, minor = torch.cuda.get_device_capability(device)
        if major < 9:
            raise RuntimeError(
                f"Fused LPLB requires Hopper or newer; got SM {major}.{minor}."
            )
        return major * 100 + minor * 10

    def warmup(self, nc: int, nv: int, *, device: torch.device) -> None:
        required = _shared_memory_bytes(nc, nv)
        if required > _HOPPER_SHMEM_BUDGET:
            raise ValueError(
                f"LPLB IPM NC={nc}, NV={nv} needs {required / 1024:.1f} KiB "
                f"shared memory; budget is {_HOPPER_SHMEM_BUDGET / 1024:.1f} KiB."
            )
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        sm_version = self._sm_version(device)
        key = (nc, nv, device_index, sm_version)
        if key in self._warmed_shapes:
            return
        module = _load_ipm_module(nc, nv, sm_version)
        a = torch.zeros((nc, nv), dtype=torch.float32, device=device)
        b = torch.zeros(nc, dtype=torch.float32, device=device)
        c = torch.zeros(nv, dtype=torch.float32, device=device)
        result = torch.empty(nv, dtype=torch.float32, device=device)
        module.ipm_solve(a, b, c, result)
        self._warmed_shapes.add(key)

    def solve(
        self,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        *,
        result: Tensor,
    ) -> Tensor:
        if not a.is_cuda or a.dtype != torch.float32:
            raise RuntimeError("SGLang fused LPLB requires CUDA float32 inputs.")
        nc, nv = a.shape
        module = _load_ipm_module(nc, nv, self._sm_version(a.device))
        module.ipm_solve(a, b, c, result)
        return result
