from __future__ import annotations

import pytest
import torch

from moe_load_balancer import (
    MoELoadBalancer,
    PlacementSnapshot,
    RoutingDecision,
    RoutingPhase,
    RoutingPolicyConfig,
    RoutingRequest,
    RoutingStage,
)


def _placement() -> PlacementSnapshot:
    return PlacementSnapshot(
        layer_id=0,
        num_logical_experts=2,
        num_physical_experts=4,
        ep_size=2,
        num_local_physical_experts=2,
        physical_to_logical_map=torch.tensor([0, 1, 0, 1]),
        logical_to_physical_candidates=torch.tensor([[0, 2], [1, 3]]),
        logical_to_physical_count=torch.tensor([2, 2]),
        default_physical_for_logical=torch.tensor([0, 1]),
    )


class _PhysicalRouter:
    phase = RoutingPhase.PHYSICAL_ROUTING

    def __init__(self, calls):
        self.calls = calls

    def route_tokens(self, request, policy):
        self.calls.append(("physical", policy.name))
        return RoutingDecision(
            routed_physical_topk_ids=torch.tensor([[0, 2], [2, 2]]),
            topk_weights=request.topk_weights,
        )


class _PassThroughPhysicalRouter:
    phase = RoutingPhase.PHYSICAL_ROUTING

    def route_tokens(self, request, policy):
        del policy
        return RoutingDecision(
            routed_physical_topk_ids=request.logical_topk_ids,
            topk_weights=request.topk_weights * 0.5,
        )


class _SharedRouter:
    phase = RoutingPhase.SHARED_EXPERT

    def __init__(self, calls):
        self.calls = calls
        self.seen_rank_load = None
        self.seen_active = object()

    def route_tokens(self, request, policy):
        self.calls.append(("shared", policy.name))
        self.seen_rank_load = request.routed_rank_load.clone()
        self.seen_active = request.active_rank_token_count
        return RoutingDecision(
            routed_physical_topk_ids=request.routed_physical_topk_ids,
            topk_weights=request.topk_weights,
            shared_expert_rank=torch.zeros(2, dtype=torch.int64),
        )


@pytest.mark.unit
def test_combined_pipeline_runs_lplb_then_count_then_waterfill():
    calls = []
    physical = _PhysicalRouter(calls)
    shared = _SharedRouter(calls)
    mlb = MoELoadBalancer(
        placement_planners={},
        routing_policies={"lplb": physical, "waterfill": shared},
    )
    request = RoutingRequest(
        layer_id=0,
        logical_topk_ids=torch.tensor([[0, 1], [1, 0]]),
        topk_weights=torch.ones((2, 2)),
        placement=_placement(),
        policies=(
            RoutingPolicyConfig(name="lplb"),
            RoutingPolicyConfig(
                name="waterfill",
                metadata={"experts_per_rank": 2},
            ),
        ),
        # Combined mode must ignore these pre-LPLB values.
        routed_rank_load=torch.tensor([99, 1]),
        active_rank_token_count=torch.tensor([5, 5]),
    )

    decision = mlb.route_tokens(request)

    assert calls == [("physical", "lplb"), ("shared", "waterfill")]
    assert torch.equal(shared.seen_rank_load, torch.tensor([1, 3]))
    assert shared.seen_active is None
    assert torch.equal(
        decision.routed_physical_topk_ids,
        torch.tensor([[0, 2], [2, 2]]),
    )
    assert torch.equal(decision.shared_expert_rank, torch.zeros(2, dtype=torch.int64))


@pytest.mark.unit
def test_standalone_dynamic_waterfill_uses_framework_prepared_load():
    calls = []
    shared = _SharedRouter(calls)
    mlb = MoELoadBalancer(
        placement_planners={},
        routing_policies={"waterfill": shared},
    )
    physical_ids = torch.tensor([[0, 1], [2, 3]])
    request = RoutingRequest(
        layer_id=0,
        logical_topk_ids=torch.tensor([[0, 1], [0, 1]]),
        routed_physical_topk_ids=physical_ids,
        topk_weights=torch.ones((2, 2)),
        policies=(
            RoutingPolicyConfig(
                name="waterfill",
                metadata={"experts_per_rank": 2, "world_size": 2},
            ),
        ),
        routed_rank_load=torch.tensor([9, 1]),
        active_rank_token_count=torch.tensor([2, 2]),
    )

    mlb.route_tokens(request)

    assert torch.equal(shared.seen_rank_load, torch.tensor([9, 1]))
    assert torch.equal(shared.seen_active, torch.tensor([2, 2]))


@pytest.mark.unit
def test_physical_only_pipeline_returns_policy_ids_and_weights():
    mlb = MoELoadBalancer(
        placement_planners={},
        routing_policies={"lplb": _PassThroughPhysicalRouter()},
    )
    request = RoutingRequest(
        layer_id=0,
        logical_topk_ids=torch.tensor([[0, 1]]),
        topk_weights=torch.ones((1, 2)),
        placement=_placement(),
        policies=(RoutingPolicyConfig(name="lplb"),),
    )

    decision = mlb.route_tokens(request)

    assert torch.equal(decision.routed_physical_topk_ids, torch.tensor([[0, 1]]))
    assert torch.equal(decision.topk_weights, torch.full((1, 2), 0.5))


@pytest.mark.unit
def test_empty_token_input_still_runs_combined_pipeline_with_zero_rank_load():
    calls = []
    shared = _SharedRouter(calls)
    mlb = MoELoadBalancer(
        placement_planners={},
        routing_policies={
            "lplb": _PassThroughPhysicalRouter(),
            "waterfill": shared,
        },
    )
    request = RoutingRequest(
        layer_id=0,
        logical_topk_ids=torch.empty((0, 2), dtype=torch.int64),
        topk_weights=torch.empty((0, 2)),
        placement=_placement(),
        policies=(
            RoutingPolicyConfig(name="lplb"),
            RoutingPolicyConfig(name="waterfill"),
        ),
        global_logical_count=torch.zeros(2),
    )

    mlb.route_tokens(request)

    assert torch.equal(shared.seen_rank_load, torch.zeros(2, dtype=torch.int64))
    assert calls == [("shared", "waterfill")]


@pytest.mark.unit
def test_no_l2_policy_uses_committed_default_mapping():
    mlb = MoELoadBalancer(placement_planners={}, routing_policies={})
    request = RoutingRequest(
        layer_id=0,
        logical_topk_ids=torch.tensor([[1, 0, -1]]),
        topk_weights=torch.ones((1, 3)),
        placement=_placement(),
    )

    decision = mlb.route_tokens(request)

    assert torch.equal(decision.routed_physical_topk_ids, torch.tensor([[1, 0, -1]]))


@pytest.mark.unit
def test_stage_gated_physical_policy_falls_back_to_committed_mapping():
    calls = []
    mlb = MoELoadBalancer(
        placement_planners={},
        routing_policies={"lplb": _PhysicalRouter(calls)},
    )
    request = RoutingRequest(
        layer_id=0,
        logical_topk_ids=torch.tensor([[0, 1]]),
        topk_weights=torch.ones((1, 2)),
        placement=_placement(),
        stage=RoutingStage.DECODE,
        policies=(
            RoutingPolicyConfig(
                name="lplb",
                enabled_stages=frozenset({RoutingStage.PREFILL}),
            ),
        ),
    )

    decision = mlb.route_tokens(request)

    assert calls == []
    assert torch.equal(decision.routed_physical_topk_ids, torch.tensor([[0, 1]]))
    assert decision.metadata["skipped_policies"] == ("lplb",)


@pytest.mark.unit
def test_physical_policy_cannot_run_after_shared_policy():
    calls = []
    mlb = MoELoadBalancer(
        placement_planners={},
        routing_policies={
            "lplb": _PhysicalRouter(calls),
            "waterfill": _SharedRouter(calls),
        },
    )
    request = RoutingRequest(
        layer_id=0,
        logical_topk_ids=torch.tensor([[0, 1]]),
        routed_physical_topk_ids=torch.tensor([[0, 1]]),
        topk_weights=torch.ones((1, 2)),
        placement=_placement(),
        policies=(
            RoutingPolicyConfig(
                name="waterfill",
                metadata={"experts_per_rank": 2},
            ),
            RoutingPolicyConfig(name="lplb"),
        ),
    )
    with pytest.raises(ValueError, match="physical-routing policy"):
        mlb.route_tokens(request)
