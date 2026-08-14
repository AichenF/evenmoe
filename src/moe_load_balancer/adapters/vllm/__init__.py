"""Stateless translation between vLLM values and MLB framework-neutral types.

This package must not import vLLM.  Glue on the vLLM side owns every framework
call; these helpers only reshape plain values and tensors.
"""

from moe_load_balancer.adapters.vllm.placement import (
    VllmPlacementMaps,
    to_placement_request,
    to_vllm_maps,
    to_vllm_physical_to_logical,
)
from moe_load_balancer.adapters.vllm.routing import (
    nearest_replica_table,
    stage_from_vllm,
    to_placement_snapshot,
    to_routing_request,
    to_vllm_topk_ids,
)

__all__ = [
    "VllmPlacementMaps",
    "nearest_replica_table",
    "stage_from_vllm",
    "to_placement_request",
    "to_placement_snapshot",
    "to_routing_request",
    "to_vllm_maps",
    "to_vllm_physical_to_logical",
    "to_vllm_topk_ids",
]
