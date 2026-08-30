from dataclasses import FrozenInstanceError

import pytest

from moe_load_balancer import (
    ExpertWeightTransfer,
    PlacementPlan,
    PlacementSnapshot,
    RoutingDecision,
    RoutingPhase,
    RoutingPolicyConfig,
    RoutingRequest,
    RoutingStage,
    TokenRouter,
    WeightTransferRequest,
)


class _PassthroughRouter:
    phase = RoutingPhase.PHYSICAL_ROUTING

    def route_tokens(self, request, policy):
        del policy
        return RoutingDecision(
            routed_physical_topk_ids=request.logical_topk_ids,
            topk_weights=request.topk_weights,
        )


class _RecordingTransfer:
    def __init__(self):
        self.registered = None
        self.transferred = []

    def register_weights(self, expert_weights):
        self.registered = expert_weights

    def transfer(self, request):
        self.transferred.append(request)


def _placement() -> PlacementSnapshot:
    return PlacementSnapshot(
        layer_id=0,
        num_logical_experts=2,
        num_physical_experts=2,
        ep_size=1,
        num_local_physical_experts=2,
        physical_to_logical_map=[0, 1],
        logical_to_physical_candidates=[[0], [1]],
        logical_to_physical_count=[1, 1],
        default_physical_for_logical=[0, 1],
    )


@pytest.mark.unit
def test_routing_stage_normalizes_strings():
    assert RoutingStage.from_str("PREFILL") is RoutingStage.PREFILL
    assert RoutingStage.from_str(RoutingStage.DECODE) is RoutingStage.DECODE
    with pytest.raises(ValueError):
        RoutingStage.from_str("unknown")


@pytest.mark.unit
def test_policy_config_is_frozen_and_stage_aware():
    policy = RoutingPolicyConfig(
        name="waterfill",
        enabled_stages=frozenset({"prefill"}),
        metadata={"experts_per_rank": 4},
    )
    assert policy.should_run_for_stage(RoutingStage.PREFILL)
    assert not policy.should_run_for_stage(RoutingStage.DECODE)
    with pytest.raises(TypeError):
        policy.metadata["experts_per_rank"] = 8  # type: ignore[index]


@pytest.mark.unit
def test_routing_request_normalizes_ordered_pipeline():
    request = RoutingRequest(
        layer_id=0,
        logical_topk_ids=[[0]],
        topk_weights=[[1.0]],
        policies=("lplb", RoutingPolicyConfig(name="waterfill")),
        placement=_placement(),
    )
    assert tuple(policy.name for policy in request.policies) == ("lplb", "waterfill")


@pytest.mark.unit
def test_core_dataclasses_are_frozen():
    request = RoutingRequest(
        layer_id=0,
        logical_topk_ids=[[0]],
        topk_weights=[[1.0]],
    )
    with pytest.raises(FrozenInstanceError):
        request.layer_id = 1  # type: ignore[misc]

    snapshot = _placement()
    with pytest.raises(FrozenInstanceError):
        snapshot.layer_id = 1  # type: ignore[misc]

    plan = PlacementPlan(
        physical_to_logical_map=[[0, 1]],
        logical_to_all_physical_map=[[[0], [1]]],
        logical_to_physical_count=[[1, 1]],
    )
    with pytest.raises(FrozenInstanceError):
        plan.physical_to_logical_map = [[1, 0]]  # type: ignore[misc]

    transfer_request = WeightTransferRequest(layer_id=0, routing_signal=[[0]])
    with pytest.raises(FrozenInstanceError):
        transfer_request.layer_id = 1  # type: ignore[misc]


@pytest.mark.unit
def test_token_router_protocol_requires_phase_and_policy_argument():
    router = _PassthroughRouter()
    assert isinstance(router, TokenRouter)
    request = RoutingRequest(
        layer_id=0,
        logical_topk_ids=[[0]],
        topk_weights=[[1.0]],
    )
    decision = router.route_tokens(request, RoutingPolicyConfig(name="test"))
    assert decision.routed_physical_topk_ids == [[0]]


@pytest.mark.unit
def test_weight_transfer_request_rejects_negative_layer_id():
    with pytest.raises(ValueError, match="non-negative"):
        WeightTransferRequest(layer_id=-1, routing_signal=[[0]])


@pytest.mark.unit
def test_expert_weight_transfer_protocol_is_satisfied_by_duck_typing():
    transfer = _RecordingTransfer()
    assert isinstance(transfer, ExpertWeightTransfer)

    transfer.register_weights(["layer0_weights"])
    request = WeightTransferRequest(layer_id=0, routing_signal=[[0]])
    transfer.transfer(request)

    assert transfer.registered == ["layer0_weights"]
    assert transfer.transferred == [request]


@pytest.mark.unit
def test_routing_request_rejects_positional_construction():
    with pytest.raises(TypeError):
        RoutingRequest(0, [[0]], [[1.0]])  # type: ignore[misc]
