"""Stateless SGLang data translations.

SGLang Glue owns lifecycle, communication, threshold decisions, recording,
weight movement and committed placement metadata. These adapters only
translate between SGLang-shaped values and framework-neutral MLB types.
"""

from moe_load_balancer.adapters.sglang.placement import (
    SGLangPlacementMaps,
    to_placement_request,
    to_placement_snapshot,
    to_sglang_maps,
)
from moe_load_balancer.adapters.sglang.routing import (
    SGLangRoutingOutput,
    to_routing_request,
    to_sglang_routing_output,
)

__all__ = [
    "SGLangPlacementMaps",
    "SGLangRoutingOutput",
    "to_placement_request",
    "to_placement_snapshot",
    "to_routing_request",
    "to_sglang_maps",
    "to_sglang_routing_output",
]
