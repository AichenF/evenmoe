"""GPU kernels owned by the SDK.

Algorithm-specific kernels live in subpackages (e.g. ``lplb/``). Shared
build / loader helpers live in :mod:`moe_load_balancer.kernels.utils`.

Nothing in this package imports an inference framework. Vendored C++ /
CUDA sources are self-contained: framework-specific glue (Protocol
implementations, runtime hooks) lives in separate adapter packages under
:mod:`moe_load_balancer.adapters`.
"""
