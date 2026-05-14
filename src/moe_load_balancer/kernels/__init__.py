"""GPU kernels owned by the SDK.

Algorithm-specific kernels live in subpackages (e.g. ``lplb/``). Shared
build / loader helpers live in :mod:`moe_load_balancer.kernels.utils`.

The Python-side helpers do not import any inference framework. C++ / CUDA
sources may declare a build-time dependency on framework header libraries
(e.g. ``sgl_kernel``) via :func:`moe_load_balancer.kernels.utils.register_dependency`,
keeping framework coupling to the compile stage only.
"""
