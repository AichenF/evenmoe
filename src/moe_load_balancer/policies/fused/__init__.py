"""L1+L2-fused placement/routing algorithms, exposed through MLB's placement
planner protocol on the L1 side only.

Not MLB's L3 (see core/l3.py): these are policies whose placement decision
and routing decision are solved together and only make sense as a pair --
housed separately from the general L1 policy pool because pairing one of
these with an unrelated L2 router, the way deepseek/lplb/waterfill compose
freely, is not meaningful.
"""

from moe_load_balancer.policies.fused.ultraep import (
    UltraEPL3Policy,
    build_ultraep_initial_physical_to_logical_map,
)

__all__ = ["UltraEPL3Policy", "build_ultraep_initial_physical_to_logical_map"]
