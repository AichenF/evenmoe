"""LPLB CUDA kernel pipeline — framework-neutral.

This package owns the LPLB Linear-Programming load-balancer kernels:

- ``cuda_solver``   four JIT-compiled CUDA modules (IPM + LP prep / post /
                    dispatch); SDK-side build via ``moe_load_balancer.kernels.utils``
- ``torch_solver``  PyTorch reference IPM (test oracle, CPU-runnable)
- ``cublasdx_solver``  cuBLASDx availability probe
- ``shmem_budget``  shared-memory size calculation
- ``csrc/``         CUDA kernel source files (``.cuh``)
- ``include/moe_lb_jit/``  vendored helper C++ headers (TensorView,
                          LaunchKernel, etc.; originally ``sgl_kernel/*`` in
                          PR sgl-project/sglang#24515, renamed during
                          vendoring so the SDK carries no sgl_kernel surface)
- ``resources/download-mathdx.sh``   Math-DX (cuBLASDx) installer

The pipeline has no runtime or build-time dependency on any inference
framework. Framework-specific glue (e.g. building a ``LPLBKernels`` Protocol
instance backed by these kernels, or threading them through a specific
runtime's TopK hook) lives under ``moe_load_balancer.adapters.<framework>``.
"""
