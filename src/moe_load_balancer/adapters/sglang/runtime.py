from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from moe_load_balancer.adapters.sglang import placement
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
    layout = placement._expert_layout(context)
    return MoELoadBalancer.from_algorithm(
        context.config_bag("exec").moe.moe_load_balancer_algorithm,
        ep_size=layout["ep_size"],
        source_rank=context.parallel.moe_ep_rank,
        experts_per_rank=layout["num_local_physical_experts"],
        collectives=_SGLangCollectives(context),
    )


def commit_placement(mlb: MoELoadBalancer, context: RuntimeContext, layer_ids: Iterable[int]):
    """Notify MLB after SGLang has committed the placement and weights."""
    if not mlb.routing_capabilities.requires_placement_state:
        return
    metadata = placement._placement(context)
    for layer_id in layer_ids:
        mlb.on_placement_committed(placement.to_placement_snapshot(metadata, layer_id))
