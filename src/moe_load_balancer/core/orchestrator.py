from __future__ import annotations

from dataclasses import replace
from typing import Callable, Mapping, Optional

from moe_load_balancer.core.l1 import PlacementPlanner
from moe_load_balancer.core.l2 import TokenRouter
from moe_load_balancer.core.types import (
    PlacementPlan,
    PlacementRequest,
    PlacementSnapshot,
    RoutingDecision,
    RoutingPhase,
    RoutingPolicyConfig,
    RoutingRequest,
)

RankLoadCounter = Callable[[object, Optional[PlacementSnapshot], RoutingPolicyConfig], object]


class MoELoadBalancer:
    """Framework-neutral entry point for L1 placement and L2 routing."""

    def __init__(
        self,
        *,
        placement_planners: Optional[Mapping[str, PlacementPlanner]] = None,
        routing_policies: Optional[Mapping[str, TokenRouter]] = None,
        rank_load_counter: Optional[RankLoadCounter] = None,
    ) -> None:
        if placement_planners is None:
            from moe_load_balancer.policies.l1.eplb import (
                EPLBPlacementPlanner,
                SUPPORTED_PLACEMENT_POLICIES,
            )

            placement_planners = {
                name: EPLBPlacementPlanner(name) for name in SUPPORTED_PLACEMENT_POLICIES
            }
        if rank_load_counter is None:
            from moe_load_balancer.kernels.rank_count import count_routed_per_rank

            rank_load_counter = count_routed_per_rank
        self._placement_planners = dict(placement_planners)
        self._routing_policies = dict(routing_policies or {})
        self._rank_load_counter = rank_load_counter

    def plan_placement(self, request: PlacementRequest) -> PlacementPlan:
        """Run the selected L1 policy and return an uncommitted candidate."""

        if request.policy.is_disabled:
            raise ValueError(
                "Placement policy is disabled. The framework must skip the "
                "plan_placement call when it decides not to rebalance."
            )
        try:
            planner = self._placement_planners[request.policy.name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown placement policy {request.policy.name!r}; registered "
                f"policies are {sorted(self._placement_planners)}."
            ) from exc
        return planner.plan(request)

    def prepare_routing_layer(
        self,
        policy_name: str,
        placement: PlacementSnapshot,
    ) -> None:
        """Prepare optional per-layer policy state outside the hot path."""

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

    def route_tokens(self, request: RoutingRequest) -> RoutingDecision:
        """Run physical routing and shared-expert policies in order."""

        current_physical = request.routed_physical_topk_ids
        current_weights = request.topk_weights
        shared_expert_rank = None
        ran_physical_policy = False
        ran_shared_policy = False
        executed: list[str] = []
        skipped: list[str] = []

        for policy_config in request.policies:
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
                )
                decision = router.route_tokens(policy_request, policy_config)
                current_physical = decision.routed_physical_topk_ids
                current_weights = decision.topk_weights
                ran_physical_policy = True
            elif router.phase is RoutingPhase.SHARED_EXPERT:
                if ran_shared_policy:
                    raise ValueError(
                        "Only one shared-expert policy is supported in one L2 pipeline."
                    )
                if current_physical is None:
                    current_physical = _default_physical_routing(request)

                if ran_physical_policy or request.routed_rank_load is None:
                    routed_rank_load = self._rank_load_counter(
                        current_physical,
                        request.placement,
                        policy_config,
                    )
                else:
                    routed_rank_load = request.routed_rank_load

                policy_request = replace(
                    request,
                    topk_weights=current_weights,
                    routed_physical_topk_ids=current_physical,
                    routed_rank_load=routed_rank_load,
                    # Combined LPLB + Waterfill deliberately uses Waterfill's
                    # static path and therefore performs no second collective.
                    active_rank_token_count=(
                        None if ran_physical_policy else request.active_rank_token_count
                    ),
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
            metadata={
                "executed_policies": tuple(executed),
                "skipped_policies": tuple(skipped),
            },
        )


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
