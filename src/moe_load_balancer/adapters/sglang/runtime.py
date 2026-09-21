from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from moe_load_balancer.adapters.sglang.placement import (
    to_placement_request,
    to_placement_snapshot,
    to_sglang_maps,
)
from moe_load_balancer.adapters.sglang.routing import (
    to_routing_request,
    to_sglang_routing_output,
)
from moe_load_balancer.core.orchestrator import MoELoadBalancer

if TYPE_CHECKING:
    from sglang.srt.runtime_context import RuntimeContext


class _SGLangCollectives:
    def __init__(self, context: RuntimeContext):
        self.context = context

    @property
    def rank(self) -> int:
        return self.context.parallel.moe_ep_group.rank_in_group

    @property
    def world_size(self) -> int:
        return self.context.parallel.moe_ep_group.world_size

    def all_reduce_sum(self, payload):
        return self.context.parallel.moe_ep_group.all_reduce(payload)


def create_load_balancer(context: RuntimeContext) -> MoELoadBalancer:
    """Create a framework-neutral MLB from this engine's configuration."""
    layout = _expert_layout(context)
    return MoELoadBalancer.from_algorithm(
        context.config_bag("exec").moe.moe_load_balancer_algorithm,
        ep_size=layout["ep_size"],
        source_rank=context.parallel.moe_ep_rank,
        experts_per_rank=layout["num_local_physical_experts"],
        collectives=_SGLangCollectives(context),
    )


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


def plan_placement(
    mlb: MoELoadBalancer,
    context: RuntimeContext,
    logical_count,
    *,
    active_ranks=None,
    use_flat_topology=False,
):
    """Adapt context and per-call counts to the common placement API."""
    layout = _expert_layout(context)
    request = to_placement_request(
        logical_count,
        num_physical_experts=layout["num_physical_experts"],
        num_local_physical_experts=layout["num_local_physical_experts"],
        num_groups=layout["model_config_for_expert_location"].num_groups,
        num_nodes=1 if use_flat_topology else context.parallel.nnodes,
        algorithm=context.config_bag("exec").moe.eplb_algorithm,
        active_ranks=active_ranks,
    )
    return to_sglang_maps(mlb.plan_placement(request))


def _placement(context: RuntimeContext):
    metadata = context.resources.expert_location_metadata
    if metadata is None:
        raise RuntimeError("MLB routing requires committed expert metadata.")
    return metadata


def commit_placement(mlb: MoELoadBalancer, context: RuntimeContext, layer_ids: Iterable[int]):
    """Notify MLB after SGLang has committed the placement and weights."""
    if not mlb.routing_capabilities.requires_placement_state:
        return
    metadata = _placement(context)
    for layer_id in layer_ids:
        mlb.on_placement_committed(to_placement_snapshot(metadata, layer_id))


def route_topk(
    mlb: MoELoadBalancer,
    context: RuntimeContext,
    *,
    layer_id: int,
    topk_output,
    token_count,
    stage: str | None,
    routed_scaling_factor: float,
):
    """Adapt one batch; recording and framework side effects stay in SGLang."""
    metadata = _placement(context)
    request = to_routing_request(
        layer_id=layer_id,
        logical_topk_ids=topk_output.topk_ids,
        topk_weights=topk_output.topk_weights,
        placement=to_placement_snapshot(metadata, layer_id),
        stage=stage,
        token_count=token_count,
        routed_scaling_factor=routed_scaling_factor,
    )
    return to_sglang_routing_output(
        mlb.route_tokens(request),
        router_logits=topk_output.router_logits,
        num_routed_experts=metadata.num_physical_experts,
        world_size=metadata.ep_size,
        routed_scaling_factor=routed_scaling_factor,
    )
