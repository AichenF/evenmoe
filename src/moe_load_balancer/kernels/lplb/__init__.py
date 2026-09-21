"""LPLB CUDA kernel pipeline — framework-neutral.

This package owns the LPLB Linear-Programming load-balancer kernels:

- ``cuda_solver``           four JIT-compiled CUDA modules (IPM + LP prep /
                             post / dispatch). Build via
                             ``moe_load_balancer.kernels.jit.utils``.
- ``protocol``              execution contract consumed by the LPLB policy.
- ``shmem_budget``           shared-memory size calculation per (NC, NV).
- ``csrc/``                  CUDA kernel sources (header-only ``.cuh``).
- ``include/moe_lb_jit/``    Vendored helper C++ headers (TensorView
                             validation, ``LaunchKernel`` wrapper, etc.).
- ``resources/download-mathdx.sh``   Math-DX (cuBLASDx) installer that
                             populates ``resources/mathdx/``.

The pipeline has no runtime or build-time dependency on any inference
framework. Framework-specific implementations of ``LPLBKernels`` live
separately under ``moe_load_balancer.adapters.<framework>``.
"""

from moe_load_balancer.kernels.lplb.backend import CUDALPLBKernels
from moe_load_balancer.kernels.lplb.protocol import LPLBKernels

__all__ = ["CUDALPLBKernels", "LPLBKernels"]
