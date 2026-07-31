from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from moe_load_balancer.core.types import (
    PlacementPlan,
    PlacementPolicyConfig,
    PlacementRequest,
    PlacementSnapshot,
    PlacementStats,
    TensorLike,
)


@dataclass(frozen=True)
class SGLangPlacementMaps:
    """Plain candidate maps returned to SGLang Glue."""

    physical_to_logical_map: TensorLike
    logical_to_all_physical_map: TensorLike
    logical_to_physical_count: TensorLike


def to_placement_request(
    logical_count: TensorLike,
    *,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    algorithm: str = "auto",
    active_ranks: Optional[TensorLike] = None,
) -> PlacementRequest:
    """Translate SGLang-completed EPLB statistics into an MLB request.

    SGLang must perform recording, dump, physical-to-logical aggregation,
    all-reduce and threshold gating before calling this function.
    """

    policy_metadata: dict[str, Any] = {
        "algorithm": algorithm,
        "num_physical_experts": num_physical_experts,
        "num_local_physical_experts": num_local_physical_experts,
        "num_groups": num_groups,
        "num_nodes": num_nodes,
    }
    if active_ranks is not None:
        policy_metadata["active_ranks"] = active_ranks

    return PlacementRequest(
        stats=PlacementStats(logical_count=logical_count),
        policy=PlacementPolicyConfig(name="eplb", metadata=policy_metadata),
    )


def to_sglang_maps(plan: PlacementPlan) -> SGLangPlacementMaps:
    """Translate an uncommitted MLB plan into plain SGLang mapping tensors."""

    return SGLangPlacementMaps(
        physical_to_logical_map=plan.physical_to_logical_map,
        logical_to_all_physical_map=plan.logical_to_all_physical_map,
        logical_to_physical_count=plan.logical_to_physical_count,
    )


def to_placement_snapshot(
    live_metadata: object,
    layer_id: int,
) -> PlacementSnapshot:
    """Translate SGLang's current committed placement into an L2 snapshot.

    The function stores nothing and returns tensor views from ``live_metadata``.
    Calling it again after SGLang commits a rebalance observes the new mapping.
    """

    if live_metadata is None:
        raise ValueError("live_metadata must be SGLang's committed placement metadata.")
    num_physical = int(getattr(live_metadata, "num_physical_experts"))
    num_local = int(getattr(live_metadata, "num_local_physical_experts"))
    if num_physical % num_local != 0:
        raise ValueError(
            "SGLang placement requires num_physical_experts to be divisible by "
            "num_local_physical_experts."
        )
    ep_size = num_physical // num_local

    rank_dispatch = getattr(
        live_metadata,
        "logical_to_rank_dispatch_physical_map",
        None,
    )
    explicit_rank_map = getattr(live_metadata, "physical_to_rank_map", None)
    if explicit_rank_map is not None and getattr(explicit_rank_map, "ndim", 1) > 1:
        explicit_rank_map = explicit_rank_map[layer_id]

    return PlacementSnapshot(
        layer_id=layer_id,
        num_logical_experts=int(getattr(live_metadata, "num_logical_experts")),
        num_physical_experts=num_physical,
        ep_size=ep_size,
        num_local_physical_experts=num_local,
        physical_to_logical_map=getattr(live_metadata, "physical_to_logical_map")[layer_id],
        logical_to_physical_candidates=getattr(
            live_metadata,
            "logical_to_all_physical_map",
        )[layer_id],
        logical_to_physical_count=getattr(
            live_metadata,
            "logical_to_all_physical_map_num_valid",
        )[layer_id],
        default_physical_for_logical=(
            rank_dispatch[layer_id] if rank_dispatch is not None else None
        ),
        physical_to_rank_map=explicit_rank_map,
    )
