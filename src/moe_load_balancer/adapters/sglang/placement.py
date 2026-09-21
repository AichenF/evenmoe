from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from moe_load_balancer.core.types import (
    PlacementPlan,
    PlacementPolicyConfig,
    PlacementRequest,
    PlacementSnapshot,
    PlacementStats,
    PlacementTopology,
    TensorLike,
)

if TYPE_CHECKING:
    from sglang.srt.runtime_context import RuntimeContext

    from moe_load_balancer.core.orchestrator import MoELoadBalancer


def _expert_layout(context: RuntimeContext):
    # Bootstrap happens before committed placement metadata exists.
    from sglang.srt.arg_groups.model_override_base import model_config_of
    from sglang.srt.eplb.expert_location import ExpertLocationMetadata

    layout = ExpertLocationMetadata._init_common(
        model_config_of(context.server_args), context=context
    )
    if layout is None:
        raise ValueError("MLB routing requires MoE expert-location metadata.")
    return layout


def _placement(context: RuntimeContext):
    metadata = context.resources.expert_location_metadata
    if metadata is None:
        raise RuntimeError("MLB routing requires committed expert metadata.")
    return metadata


def commit_placement(
    mlb: MoELoadBalancer, context: RuntimeContext, layer_ids: Iterable[int]
) -> None:
    """Notify MLB after SGLang has committed the placement and weights."""
    if not mlb.routing_capabilities.requires_placement_state:
        return
    metadata = _placement(context)
    for layer_id in layer_ids:
        mlb.on_placement_committed(to_placement_snapshot(metadata, layer_id))


@dataclass(frozen=True)
class SGLangPlacementMaps:
    """Plain candidate maps returned to SGLang Glue."""

    physical_to_logical_map: TensorLike
    logical_to_all_physical_map: TensorLike
    logical_to_physical_count: TensorLike


def to_placement_request(
    logical_count: TensorLike,
    *,
    context: Optional[RuntimeContext] = None,
    num_physical_experts: Optional[int] = None,
    num_local_physical_experts: Optional[int] = None,
    num_groups: Optional[int] = None,
    num_nodes: Optional[int] = None,
    algorithm: Optional[str] = None,
    active_ranks: Optional[TensorLike] = None,
    policy_metadata: Optional[dict[str, Any]] = None,
) -> PlacementRequest:
    """Translate SGLang-completed EPLB statistics into an MLB request.

    SGLang must perform recording, dump, physical-to-logical aggregation,
    all-reduce and threshold gating before calling this function.

    Context supplies omitted layout and policy values. Explicit arguments,
    such as ``num_nodes=1`` for flat topology, take precedence.

    ``policy_metadata`` is opaque, per-policy configuration -- ``ultraep``
    reads ``rank``/``num_nvl_ranks`` from it (its placement kernel solves
    once per EP rank and validates the value is in range); every other MLB
    policy ignores it, so it is safe to leave unset.
    """

    if context is not None:
        layout = _expert_layout(context)
        if num_physical_experts is None:
            num_physical_experts = layout["num_physical_experts"]
        if num_local_physical_experts is None:
            num_local_physical_experts = layout["num_local_physical_experts"]
        if num_groups is None:
            num_groups = layout["model_config_for_expert_location"].num_groups
        if num_nodes is None:
            num_nodes = context.parallel.nnodes
        if algorithm is None:
            algorithm = context.config_bag("exec").moe.eplb_algorithm
    if num_physical_experts is None or num_local_physical_experts is None or num_nodes is None:
        raise ValueError("Placement requires a context or explicit expert and node counts.")

    return PlacementRequest(
        stats=PlacementStats(logical_count=logical_count),
        topology=PlacementTopology(
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            active_ranks=active_ranks,
        ),
        policy=PlacementPolicyConfig(
            name="auto" if algorithm is None else algorithm, metadata=policy_metadata or {}
        ),
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
