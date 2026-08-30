"""L1+L2-fused placement/routing algorithms, exposed through MLB's placement
planner protocol on the L1 side only.

These are policies whose placement decision and routing decision are solved
together and only make sense as a pair -- housed separately from the general
L1 policy pool because pairing one of these with an unrelated L2 router, the
way deepseek/lplb/waterfill compose freely, is not meaningful.

Also houses UltraEP's own weight-commit backend (``UltraEPWeightTransfer``):
the execution detail that makes memory match a placement its L1 policy just
decided. That is not a general MLB capability -- MLB has never defined a
protocol for it, since turning a solved ``PlacementPlan`` into memory
reality has always been the framework adapter's job -- so it lives here as
UltraEP's own private implementation detail, not a standalone architectural
layer.
"""

from moe_load_balancer.policies.fused.ultraep import (
    UltraEPPlacementPolicy,
    build_ultraep_initial_physical_to_logical_map,
)
from moe_load_balancer.policies.fused.ultraep_weight_transfer import (
    UltraEPWeightTransfer,
)

__all__ = [
    "UltraEPPlacementPolicy",
    "UltraEPWeightTransfer",
    "build_ultraep_initial_physical_to_logical_map",
]
