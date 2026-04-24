"""L1 placement policy implementations.

Reserved for the future L1 optimizer (per-layer replica budget, communication-
aware placement). The current SGLang POC reuses SGLang EPLB as the placement
source and does not implement a planner here. See
:class:`moe_load_balancer.core.l1.PlacementPlanner` for the contract.
"""

from __future__ import annotations

__all__: list[str] = []
