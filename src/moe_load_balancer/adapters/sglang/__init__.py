"""SGLang interoperability for the framework-neutral load balancer.

SGLang owns lifecycle, collective transport, threshold decisions, recording,
weight movement and committed placement metadata. MLB decides when L2 needs a
collective and invokes the injected SGLang transport adapter. This package
contains plain data translations plus optional execution bindings.
"""

from moe_load_balancer.adapters.sglang.placement import (
    SGLangPlacementMaps,
    to_placement_request,
    to_placement_snapshot,
    to_sglang_maps,
)
from moe_load_balancer.adapters.sglang.routing import (
    SGLangRoutingOutput,
    count_logical_experts,
    to_routing_request,
    to_sglang_routing_output,
)

__all__ = [
    "SGLangPlacementMaps",
    "SGLangRoutingOutput",
    "count_logical_experts",
    "to_placement_request",
    "to_placement_snapshot",
    "to_routing_request",
    "to_sglang_maps",
    "to_sglang_routing_output",
]
