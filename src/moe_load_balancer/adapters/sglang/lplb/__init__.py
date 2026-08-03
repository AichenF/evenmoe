"""SGLang execution binding for the SDK-owned LPLB kernels.

Lifecycle, communication and request construction remain in SGLang Glue;
the shared :class:`MoELoadBalancer` orchestrator invokes the LPLB policy.
There is deliberately no per-layer or per-algorithm runtime in this package.
"""

from __future__ import annotations

from moe_load_balancer.adapters.sglang.lplb.kernels import SGLangLPLBKernels

__all__ = ["SGLangLPLBKernels"]
