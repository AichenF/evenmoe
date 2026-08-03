"""L1 placement policy implementations.

Currently provides the DeepSeek EPLB family. Future planners (CRAFT,
MoETuner, replica-budget allocators) implement the same
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

__all__ = [
    "EPLBPlacementPlanner",
    "EplbAlgorithm",
    "compute_algorithm",
    "rebalance_experts",
    "SUPPORTED_PLACEMENT_POLICIES",
]
