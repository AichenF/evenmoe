"""L1 placement policy implementations.

Provides the DeepSeek EPLB family and UltraEP's placement solve. Future
planners (CRAFT, MoETuner, replica-budget allocators) implement the same
:class:`moe_load_balancer.core.l1.PlacementPlanner` protocol.
"""

from __future__ import annotations

from moe_load_balancer.policies.l1.eplb import (
    EPLBPlacementPlanner,
    EplbAlgorithm,
    SUPPORTED_PLACEMENT_POLICIES,
    compute_algorithm,
    rebalance_experts,
)
from moe_load_balancer.policies.l1.ultraep import UltraEPPlacementPolicy

__all__ = [
    "EPLBPlacementPlanner",
    "EplbAlgorithm",
    "UltraEPPlacementPolicy",
    "compute_algorithm",
    "rebalance_experts",
    "SUPPORTED_PLACEMENT_POLICIES",
]
