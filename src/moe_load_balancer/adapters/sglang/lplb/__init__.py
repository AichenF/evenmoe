"""SGLang adapter for LPLB.

Provides:

- :class:`SGLangLPLBKernels` — :class:`moe_load_balancer.core.l2.LPLBKernels`
  implementation that delegates to ``sglang.jit_kernel.lplb.cuda_solver``.
  Under the kernel-adapter decision (see
  ``load_balancer_project/DECISION_lplb_kernel_adapter.md``), the SDK does
  not vendor LPLB CUDA sources yet; this adapter uses SGLang's pipeline
  in place.

- :class:`LPLBRuntime` — SGLang-side runtime that mirrors
  :class:`moe_load_balancer.adapters.sglang.waterfill.WaterfillRuntime`.
  It counts local logical-expert hits, drives the EP all-reduce, constructs
  the framework-neutral :class:`RoutingRequest`, and dispatches to
  :class:`LPLBL2Router`. Lives per-MoE-layer.
"""

from __future__ import annotations

from moe_load_balancer.adapters.sglang.lplb.kernels import SGLangLPLBKernels
from moe_load_balancer.adapters.sglang.lplb.runtime import LPLBRuntime

__all__ = ["LPLBRuntime", "SGLangLPLBKernels"]
