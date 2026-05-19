"""DeepSeek EPLB family — L1 placement planner.

Exports a framework-neutral :class:`DeepSeekEPLBPlanner` plus the vendored
algorithm dispatcher (``rebalance_experts``, ``compute_algorithm``,
``EplbAlgorithm``). SGLang adapter at
:mod:`moe_load_balancer.adapters.sglang.eplb` is the only framework-specific
glue around this package.
"""

from __future__ import annotations

from moe_load_balancer.policies.l1.eplb.algorithms import (
    EplbAlgorithm,
    compute_algorithm,
    rebalance_experts,
)
from moe_load_balancer.policies.l1.eplb.planner import DeepSeekEPLBPlanner

__all__ = [
    "DeepSeekEPLBPlanner",
    "EplbAlgorithm",
    "compute_algorithm",
    "rebalance_experts",
]
