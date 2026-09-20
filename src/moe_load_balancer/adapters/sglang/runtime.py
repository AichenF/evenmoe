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


class _SGLangRuntime:
    """Read SGLang-owned resources through one engine's context."""

    def __init__(self, context: RuntimeContext):
        self.context = context

    def create_load_balancer(self, cls):
        context = self.context
        algorithm = context.config_bag("exec").moe.moe_load_balancer_algorithm
        layout = self._expert_layout()
        return cls.from_algorithm(
            algorithm,
            ep_size=layout["ep_size"],
            source_rank=context.parallel.moe_ep_rank,
            experts_per_rank=layout["num_local_physical_experts"],
            collectives=_SGLangCollectives(context),
        )

    def _expert_layout(self):
        # Bootstrap happens before committed placement metadata exists.
        from sglang.srt.arg_groups.model_override_base import model_config_of
        from sglang.srt.eplb.expert_location import ExpertLocationMetadata

        layout = ExpertLocationMetadata._init_common(
            model_config_of(self.context.server_args), context=self.context
        )
        if layout is None:
            raise ValueError("MLB routing requires MoE expert-location metadata.")
        return layout

    def plan_placement(self, mlb, logical_count, *, active_ranks=None, use_flat_topology=False):
        layout = self._expert_layout()
        request = to_placement_request(
            logical_count,
            num_physical_experts=layout["num_physical_experts"],
            num_local_physical_experts=layout["num_local_physical_experts"],
            num_groups=layout["model_config_for_expert_location"].num_groups,
            num_nodes=1 if use_flat_topology else self.context.parallel.nnodes,
            algorithm=self.context.config_bag("exec").moe.eplb_algorithm,
            active_ranks=active_ranks,
        )
        return to_sglang_maps(mlb.plan_placement(request))

    def _placement(self):
        metadata = self.context.resources.expert_location_metadata
        if metadata is None:
            raise RuntimeError("MLB routing requires committed expert metadata.")
        return metadata

    def commit_placement(self, mlb, layer_ids: Iterable[int]):
        if not mlb.routing_capabilities.requires_placement_state:
            return
        metadata = self._placement()
        for layer_id in layer_ids:
            mlb.on_placement_committed(to_placement_snapshot(metadata, layer_id))

    def route_topk(
        self,
        mlb,
        *,
        layer_id: int,
        logical_topk_ids,
        topk_weights,
        router_logits,
        token_count,
        stage: str | None,
        routed_scaling_factor: float,
    ):
        metadata = self._placement()
        request = to_routing_request(
            layer_id=layer_id,
            logical_topk_ids=logical_topk_ids,
            topk_weights=topk_weights,
            placement=to_placement_snapshot(metadata, layer_id),
            stage=stage,
            token_count=token_count,
            routed_scaling_factor=routed_scaling_factor,
        )
        output = to_sglang_routing_output(
            mlb.route_tokens(request),
            router_logits=router_logits,
            num_routed_experts=metadata.num_physical_experts,
            world_size=metadata.ep_size,
            routed_scaling_factor=routed_scaling_factor,
        )
        resources = self.context.resources
        resources.expert_distribution_recorder.on_select_experts(
            topk_ids=output.recorded_physical_topk_ids
        )
        if resources.experts_capturer is not None:
            resources.experts_capturer.capture(
                layer_id=layer_id, topk_indices=output.recorded_physical_topk_ids
            )
        return output
