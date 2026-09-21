"""Framework-neutral tensor operators and accelerated kernel backends.

Callable tensor operators live in :mod:`moe_load_balancer.kernels.ops`.
Algorithm-specific backends live in subpackages such as ``lplb`` and
``ultraep``. Shared build and loader infrastructure lives in
:mod:`moe_load_balancer.kernels.jit`.

Nothing in this package imports an inference framework. Vendored C++ /
CUDA sources are self-contained: framework-specific glue (Protocol
implementations, runtime hooks) lives in separate adapter packages under
:mod:`moe_load_balancer.adapters`.
"""
