from __future__ import annotations

from dataclasses import replace
from typing import Callable, Mapping, Optional

from moe_load_balancer.core.l1 import PlacementPlanner
from moe_load_balancer.core.l2 import (
    GlobalLogicalCountConsumer,
    RoutingCollectives,
    TokenRouter,
)
from moe_load_balancer.core.routing_pipeline import RoutingCapabilities, RoutingPipeline
from moe_load_balancer.core.types import (
    PlacementPlan,
    PlacementPolicyConfig,
    PlacementRequest,
    PlacementSnapshot,
    RoutingDecision,
    RoutingPhase,
    RoutingPolicyConfig,
    RoutingRequest,
    TensorLike,
)
from moe_load_balancer.kernels.lplb.protocol import LPLBKernels

RankLoadCounter = Callable[[object, Optional[PlacementSnapshot], RoutingPolicyConfig], object]


class MoELoadBalancer:
    """Framework-neutral entry point for L1 placement and L2 routing."""

    def __init__(
        self,
        *,
        placement_planners: Optional[Mapping[str, PlacementPlanner]] = None,
        routing_policies: Optional[Mapping[str, TokenRouter]] = None,
        rank_load_counter: Optional[RankLoadCounter] = None,
        routing_pipeline: RoutingPipeline | str | None = None,
        collectives: Optional[RoutingCollectives] = None,
    ) -> None:
        if placement_planners is None:
            from moe_load_balancer.policies.l1.eplb import (
                EPLBPlacementPlanner,
                SUPPORTED_PLACEMENT_POLICIES,
            )
            from moe_load_balancer.policies.fused.ultraep import UltraEPL3Policy

            placement_planners = {
                name: EPLBPlacementPlanner(name) for name in SUPPORTED_PLACEMENT_POLICIES
            }
            placement_planners["ultraep"] = UltraEPL3Policy()
        if rank_load_counter is None:
            from moe_load_balancer.kernels.rank_count import count_routed_per_rank

            rank_load_counter = count_routed_per_rank
        self._placement_planners = dict(placement_planners)
        self._routing_policies = dict(routing_policies or {})
        self._rank_load_counter = rank_load_counter
        self._routing_pipeline = RoutingPipeline.from_value(routing_pipeline)
        self._collectives = collectives

    @classmethod
    def from_algorithm(
        cls,
        algorithm: RoutingPipeline | str,
        *,
        ep_size: int,
        source_rank: int,
        experts_per_rank: int,
        lplb_kernels: Optional[LPLBKernels] = None,
        collectives: Optional[RoutingCollectives] = None,
    ) -> MoELoadBalancer:
        """Build the MLB-owned implementations for one algorithm expression."""

        pipeline = RoutingPipeline.from_value(algorithm)
        routers: dict[str, TokenRouter] = {}
        policy_names = {policy.name for policy in pipeline.post_topk_policies}
        if "static" in policy_names:
            from moe_load_balancer.policies.l2.replica import StaticL2Router

            routers["static"] = StaticL2Router()
        if "dynamic" in policy_names:
            from moe_load_balancer.policies.l2.replica import DynamicL2Router

            routers["dynamic"] = DynamicL2Router()
        if "dynamic_random" in policy_names:
            from moe_load_balancer.policies.l2.replica import DynamicRandomL2Router

            routers["dynamic_random"] = DynamicRandomL2Router()
        if "lplb" in policy_names:
            from moe_load_balancer.policies.l2.lplb import LPLBL2Router

            if lplb_kernels is None:
                from moe_load_balancer.kernels.lplb import CUDALPLBKernels

                lplb_kernels = CUDALPLBKernels()
            routers["lplb"] = LPLBL2Router(
                kernels=lplb_kernels,
                num_gpus=ep_size,
            )
        if "waterfill" in policy_names:
            from moe_load_balancer.policies.l2.waterfill import WaterfillL2Router

            routers["waterfill"] = WaterfillL2Router(
                source_rank=source_rank,
                world_size=ep_size,
                experts_per_rank=experts_per_rank,
            )
        if "ultraep" in policy_names:
            from moe_load_balancer.policies.l2.ultraep import UltraEPL2Router

            routers["ultraep"] = UltraEPL2Router()
        return cls(
            routing_policies=routers,
            routing_pipeline=pipeline,
            collectives=collectives,
        )

    @property
    def routing_pipeline(self) -> RoutingPipeline:
        return self._routing_pipeline

    @property
    def routing_capabilities(self) -> RoutingCapabilities:
        return self._routing_pipeline.capabilities

    @property
    def placement_policy(self) -> Optional[str]:
        """Return the L1 policy coupled to this routing pipeline, if any."""

        return self._routing_pipeline.placement_policy

    def build_initial_physical_to_logical_map(self, **kwargs):
        """Build policy-owned bootstrap placement without exposing its type."""

        policy_name = self.placement_policy
        if policy_name is None:
            return None
        planner = self._placement_planners[policy_name]
        builder = getattr(planner, "build_initial_physical_to_logical_map", None)
        if builder is None:
            raise ValueError(
                f"Placement policy {policy_name!r} has no bootstrap placement builder."
            )
        return builder(**kwargs)

    def on_placement_committed(self, placement: PlacementSnapshot) -> None:
        """Notify active policies that a framework placement was committed.

        No placement version or snapshot cache is maintained by the
        orchestrator. Policies that derive execution state consume the
        framework's current snapshot at this explicit lifecycle event.
        """

        policy_names = (
            tuple(policy.name for policy in self._routing_pipeline.post_topk_policies)
            if self._routing_pipeline.post_topk_policies
            else tuple(self._routing_policies)
        )
        for policy_name in policy_names:
            router = self._routing_policies[policy_name]
            prepare = getattr(router, "prepare_layer", None)
            if prepare is not None:
                prepare(placement)

    def prepare_routing_layer(
        self,
        policy_name: str,
        placement: PlacementSnapshot,
    ) -> None:
        """Prepare one router's per-layer state for a freshly staged placement.

        A single-policy form of on_placement_committed: a framework driving
        exactly one policy through the route_prepared_tokens fast path needs
        that policy's prepare_layer called before the fast path is, but has
        no reason to also notify every other registered policy the way
        on_placement_committed does for a framework driving the full
        route_tokens pipeline.
        """

        try:
            router = self._routing_policies[policy_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown routing policy {policy_name!r}; registered policies are "
                f"{sorted(self._routing_policies)}."
            ) from exc
        prepare = getattr(router, "prepare_layer", None)
        if prepare is not None:
            prepare(placement)

    def route_prepared_tokens(
        self,
        policy_name: str,
        layer_id: int,
        logical_topk_ids: TensorLike,
    ) -> TensorLike:
        """Run one prepared physical router without rebuilding an L2 request."""

        try:
            router = self._routing_policies[policy_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown routing policy {policy_name!r}; registered policies are "
                f"{sorted(self._routing_policies)}."
            ) from exc
        route_prepared = getattr(router, "route_prepared_tokens", None)
        if route_prepared is None:
            raise ValueError(
                f"Routing policy {policy_name!r} has no prepared-token fast path."
            )
        return route_prepared(layer_id, logical_topk_ids)

    def plan_placement(self, request: PlacementRequest) -> PlacementPlan:
        """Run the selected L1 policy and return an uncommitted candidate."""

        policy = PlacementPolicyConfig.from_value(request.policy)
        if policy.is_disabled:
            raise ValueError(
                "Placement policy is disabled. The framework must skip the "
                "plan_placement call when it decides not to rebalance."
            )
        try:
            planner = self._placement_planners[policy.name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown placement policy {policy.name!r}; registered "
                f"policies are {sorted(self._placement_planners)}."
            ) from exc
        return planner.plan(request)

    def transform_topk_inputs(
        self,
        router_logits: TensorLike,
        correction_bias: Optional[TensorLike] = None,
    ) -> tuple[TensorLike, Optional[TensorLike]]:
        """Run the configured pipeline's optional transform before TopK.

        Production physical-routing policies normally leave these inputs
        untouched. This hook keeps experimental stimuli such as ``fake`` in
        MLB without moving framework-owned TopK selection into MLB.
        """

        from moe_load_balancer.policies.l2.replica import transform_topk_inputs

        return transform_topk_inputs(
            self._routing_pipeline.input_transform,
            router_logits,
            correction_bias,
        )

    def route_tokens(self, request: RoutingRequest) -> RoutingDecision:
        """Run physical routing and shared-expert policies in order."""

        current_physical = request.routed_physical_topk_ids
        current_weights = request.topk_weights
        shared_expert_rank = None
        ran_physical_policy = False
        ran_shared_policy = False
        physical_decision_metadata: Mapping[str, object] = {}
        executed: list[str] = []
        skipped: list[str] = []

        policy_values = request.policies or self._routing_pipeline.post_topk_policies
        for policy_value in policy_values:
            policy_config = RoutingPolicyConfig.from_value(policy_value)
            if not policy_config.should_run_for_stage(request.stage):
                skipped.append(policy_config.name)
                continue
            try:
                router = self._routing_policies[policy_config.name]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown routing policy {policy_config.name!r}; registered "
                    f"policies are {sorted(self._routing_policies)}."
                ) from exc

            if router.phase is RoutingPhase.PHYSICAL_ROUTING:
                if ran_physical_policy or ran_shared_policy:
                    raise ValueError(
                        "A physical-routing policy must be the first and only "
                        "physical-routing policy in the L2 pipeline."
                    )
                policy_request = replace(
                    request,
                    topk_weights=current_weights,
                    routed_physical_topk_ids=None,
                    routed_rank_load=None,
                    active_rank_token_count=None,
                    global_logical_count=self._resolve_global_logical_count(
                        router, request
                    ),
                )
                decision = router.route_tokens(policy_request, policy_config)
                current_physical = decision.routed_physical_topk_ids
                current_weights = decision.topk_weights
                physical_decision_metadata = decision.metadata
                ran_physical_policy = True
            elif router.phase is RoutingPhase.SHARED_EXPERT:
                if ran_shared_policy:
                    raise ValueError(
                        "Only one shared-expert policy is supported in one L2 pipeline."
                    )
                if current_physical is None:
                    current_physical = _default_physical_routing(request)

                planned_rank_load = physical_decision_metadata.get("planned_rank_load")
                if planned_rank_load is not None:
                    routed_rank_load = planned_rank_load
                else:
                    routed_rank_load = self._rank_load_counter(
                        current_physical,
                        request.placement,
                        policy_config,
                    )

                load_view = policy_config.metadata.get("load_view", "local")
                if load_view == "global":
                    routed_rank_load, active_rank_token_count = self._global_waterfill_load(
                        routed_rank_load, request
                    )
                elif load_view == "local":
                    active_rank_token_count = None
                else:
                    raise ValueError(f"Unknown Waterfill load view {load_view!r}.")

                policy_request = replace(
                    request,
                    topk_weights=current_weights,
                    routed_physical_topk_ids=current_physical,
                    routed_rank_load=routed_rank_load,
                    active_rank_token_count=active_rank_token_count,
                )
                decision = router.route_tokens(policy_request, policy_config)
                current_physical = decision.routed_physical_topk_ids
                current_weights = decision.topk_weights
                shared_expert_rank = decision.shared_expert_rank
                ran_shared_policy = True
            else:
                raise ValueError(
                    f"Unsupported routing phase {router.phase!r} for policy {policy_config.name!r}."
                )
            executed.append(policy_config.name)

        if current_physical is None:
            current_physical = _default_physical_routing(request)

        return RoutingDecision(
            routed_physical_topk_ids=current_physical,
            topk_weights=current_weights,
            shared_expert_rank=shared_expert_rank,
            # The executing policy's own metadata is part of its decision, not
            # scratch space: LPLB's ``lp_probability`` and ``planned_rank_load``
            # live there and nothing else carries them, so dropping this would
            # silently make every policy-level diagnostic disappear. The
            # pipeline's own keys are written last so a policy cannot shadow
            # them.
            metadata={
                **physical_decision_metadata,
                "executed_policies": tuple(executed),
                "skipped_policies": tuple(skipped),
            },
        )

    def _resolve_global_logical_count(
        self,
        router: TokenRouter,
        request: RoutingRequest,
    ) -> object:
        """Gather the EP-wide per-logical token count, but only if it is used.

        Two things have to hold before paying for it. The router has to consume
        the count at all -- asked through ``GlobalLogicalCountConsumer`` rather
        than by policy name -- and it has to consume it *for this placement*.
        LPLB's LP is degenerate on a layer where no logical expert is
        replicated; it answers ``False`` there and returns identity dispatch, so
        counting would buy an EP collective per layer per forward whose result
        is discarded.

        A framework that already holds the count passes it in and none of this
        runs: vLLM accumulates per-layer counts inside its CUDA graph and
        all-reduces once per step for every layer at once, because a collective
        cannot be captured.
        """
        if request.global_logical_count is not None:
            return request.global_logical_count
        if not isinstance(router, GlobalLogicalCountConsumer):
            return None
        if not router.needs_global_logical_count(request):
            return None
        return self._global_logical_count(request)

    def _global_logical_count(self, request: RoutingRequest) -> object:
        if request.placement is None:
            raise ValueError("LPLB routing requires a committed placement.")
        from moe_load_balancer.kernels.expert_count import count_logical_experts

        local_count = count_logical_experts(
            request.logical_topk_ids,
            request.placement.num_logical_experts,
        )
        return self._all_reduce_sum(local_count, request.placement.ep_size)

    def _global_waterfill_load(
        self,
        local_routed_rank_load: object,
        request: RoutingRequest,
    ) -> tuple[object, object]:
        if request.placement is None:
            raise ValueError("Global Waterfill routing requires a committed placement.")

        import torch

        if not isinstance(local_routed_rank_load, torch.Tensor):
            raise TypeError("Global Waterfill rank load must be a torch.Tensor.")
        world_size = request.placement.ep_size
        rank = 0 if world_size == 1 and self._collectives is None else self._collective_rank()
        if not 0 <= rank < world_size:
            raise ValueError("Routing collective rank is outside the placement EP group.")

        payload = torch.zeros(
            world_size * 2,
            dtype=torch.int64,
            device=local_routed_rank_load.device,
        )
        payload[:world_size].copy_(local_routed_rank_load.to(torch.int64))
        local_token_count = request.token_count
        token_slot = payload[world_size + rank]
        if local_token_count is None:
            token_slot.fill_(request.logical_topk_ids.shape[0])
        elif isinstance(local_token_count, torch.Tensor):
            token_slot.copy_(local_token_count.reshape(-1)[0].to(torch.int64))
        else:
            token_slot.fill_(int(local_token_count))

        reduced = self._all_reduce_sum(payload, world_size)
        if not isinstance(reduced, torch.Tensor):
            raise TypeError("Global Waterfill AllReduce must return a torch.Tensor.")
        return reduced[:world_size], reduced[world_size:]

    def _all_reduce_sum(self, payload: object, expected_world_size: int) -> object:
        if expected_world_size == 1 and self._collectives is None:
            return payload
        if self._collectives is None:
            raise RuntimeError(
                "The configured MLB routing pipeline requires EP AllReduce, but no "
                "RoutingCollectives adapter was provided."
            )
        if self._collectives.world_size != expected_world_size:
            raise ValueError(
                "RoutingCollectives.world_size does not match the committed placement."
            )
        return self._collectives.all_reduce_sum(payload)

    def _collective_rank(self) -> int:
        if self._collectives is None:
            raise RuntimeError("RoutingCollectives is required for this routing pipeline.")
        return self._collectives.rank


def _default_physical_routing(request: RoutingRequest) -> object:
    snapshot = request.placement
    if snapshot is None:
        raise ValueError(
            "RoutingRequest requires either routed_physical_topk_ids or a "
            "committed PlacementSnapshot."
        )
    if snapshot.default_physical_for_logical is None:
        if snapshot.num_logical_experts == snapshot.num_physical_experts:
            return request.logical_topk_ids
        raise ValueError(
            "PlacementSnapshot.default_physical_for_logical is required when "
            "no physical-routing policy runs and logical/physical counts differ."
        )

    import torch

    logical_ids = request.logical_topk_ids
    if not isinstance(logical_ids, torch.Tensor):
        raise TypeError("Default physical routing requires torch.Tensor logical_topk_ids.")
    mapping = snapshot.default_physical_for_logical
    if not isinstance(mapping, torch.Tensor):
        mapping = torch.as_tensor(mapping, device=logical_ids.device)
    valid = logical_ids >= 0
    safe_ids = logical_ids.clamp(min=0).to(torch.int64)
    physical = mapping[safe_ids]
    return torch.where(valid, physical.to(logical_ids.dtype), logical_ids)
