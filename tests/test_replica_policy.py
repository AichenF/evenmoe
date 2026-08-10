from __future__ import annotations

import pytest
import torch

from moe_load_balancer import (
    MoELoadBalancer,
    PlacementSnapshot,
    RoutingPolicyConfig,
    RoutingRequest,
)
from moe_load_balancer.policies.l2.replica import (
    DynamicL2Router,
    StaticL2Router,
    route_dynamic_replicas,
    route_replicas,
    route_static_replicas,
    transform_topk_inputs,
)


def _placement(*, with_default: bool = True) -> PlacementSnapshot:
    return PlacementSnapshot(
        layer_id=0,
        num_logical_experts=4,
        num_physical_experts=6,
        ep_size=2,
        num_local_physical_experts=3,
        physical_to_logical_map=torch.tensor([0, 1, 2, 0, 2, 3]),
        logical_to_physical_candidates=torch.tensor(
            [[0, 3], [1, -1], [2, 4], [5, -1]], dtype=torch.int64
        ),
        logical_to_physical_count=torch.tensor([2, 1, 2, 1]),
        default_physical_for_logical=(torch.tensor([3, 1, 4, 5]) if with_default else None),
    )


def _request(policy: str, logical_ids: torch.Tensor) -> RoutingRequest:
    return RoutingRequest(
        layer_id=0,
        logical_topk_ids=logical_ids,
        topk_weights=torch.ones_like(logical_ids, dtype=torch.float32),
        placement=_placement(),
        policies=(RoutingPolicyConfig(name=policy),),
    )


@pytest.mark.unit
def test_static_policy_matches_sglang_mapping_and_preserves_padding():
    logical_ids = torch.tensor([[0, 1], [2, 3], [-1, 8]], dtype=torch.int32)

    physical_ids = route_static_replicas(logical_ids, _placement().default_physical_for_logical)

    assert torch.equal(
        physical_ids,
        torch.tensor([[3, 1], [4, 5], [-1, 8]], dtype=torch.int32),
    )


@pytest.mark.unit
def test_static_policy_requires_framework_selected_default_map():
    with pytest.raises(ValueError, match="default_physical_for_logical"):
        StaticL2Router().route_tokens(
            RoutingRequest(
                layer_id=0,
                logical_topk_ids=torch.tensor([[0]]),
                topk_weights=torch.ones((1, 1)),
                placement=_placement(with_default=False),
            ),
            RoutingPolicyConfig(name="static"),
        )


@pytest.mark.unit
def test_dynamic_policy_matches_sglang_random_replica_selection():
    logical_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    placement = _placement()

    torch.manual_seed(17)
    expected_index = (
        torch.randint(0, 65536, logical_ids.shape, dtype=torch.int32)
        % placement.logical_to_physical_count[logical_ids]
    )
    expected = placement.logical_to_physical_candidates[logical_ids, expected_index]
    torch.manual_seed(17)

    assert torch.equal(
        route_dynamic_replicas(
            logical_ids,
            placement.logical_to_physical_candidates,
            placement.logical_to_physical_count,
        ),
        expected,
    )


@pytest.mark.unit
def test_dynamic_policy_outputs_only_committed_candidates_and_preserves_padding():
    logical_ids = torch.tensor([[0, 2], [0, 2], [-1, 9]], dtype=torch.int32)

    placement = _placement()
    physical_ids = route_dynamic_replicas(
        logical_ids,
        placement.logical_to_physical_candidates,
        placement.logical_to_physical_count,
    )

    assert set(physical_ids[:2, 0].tolist()) <= {0, 3}
    assert set(physical_ids[:2, 1].tolist()) <= {2, 4}
    assert physical_ids[-1].tolist() == [-1, 9]


@pytest.mark.unit
@pytest.mark.parametrize("policy", ["static", "dynamic"])
def test_replica_helpers_are_fullgraph_compilable(policy):
    placement = _placement()
    logical_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    if policy == "static":
        compiled = torch.compile(route_static_replicas, backend="eager", fullgraph=True)
        physical_ids = compiled(
            logical_ids,
            placement.default_physical_for_logical,
        )
    else:
        compiled = torch.compile(route_dynamic_replicas, backend="eager", fullgraph=True)
        physical_ids = compiled(
            logical_ids,
            placement.logical_to_physical_candidates,
            placement.logical_to_physical_count,
        )

    assert physical_ids.shape == logical_ids.shape


@pytest.mark.unit
def test_fake_policy_owns_pre_topk_stimulus_and_dynamic_routing():
    mlb = MoELoadBalancer(
        routing_policies={"dynamic": DynamicL2Router()},
        routing_pipeline="fake",
    )
    logits = torch.zeros((4, 8))
    bias = torch.arange(8, dtype=torch.float32)

    torch.manual_seed(29)
    expected_logits = torch.empty_like(logits).uniform_(5, 10)
    torch.manual_seed(29)
    transformed_logits, transformed_bias = mlb.transform_topk_inputs(logits, bias)
    request = _request("dynamic", torch.tensor([[0, 1], [2, 3]]))
    request = RoutingRequest(
        layer_id=request.layer_id,
        logical_topk_ids=request.logical_topk_ids,
        topk_weights=request.topk_weights,
        placement=request.placement,
    )
    decision = mlb.route_tokens(request)

    assert transformed_logits is logits
    assert torch.equal(transformed_logits, expected_logits)
    assert torch.equal(transformed_bias, torch.zeros_like(bias))
    assert decision.routed_physical_topk_ids.shape == (2, 2)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "router"),
    [("static", StaticL2Router()), ("dynamic", DynamicL2Router())],
)
def test_replica_policies_run_through_unified_l2_orchestrator(name, router):
    mlb = MoELoadBalancer(routing_policies={name: router}, routing_pipeline=name)
    logical_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

    decision = mlb.route_tokens(_request(name, logical_ids))

    assert decision.routed_physical_topk_ids.shape == logical_ids.shape
    assert decision.metadata["executed_policies"] == (name,)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "router"),
    [("static", StaticL2Router()), ("dynamic", DynamicL2Router())],
)
def test_production_replica_policies_leave_pre_topk_inputs_unchanged(name, router):
    mlb = MoELoadBalancer(routing_policies={name: router})
    logits = torch.randn(2, 4)
    bias = torch.randn(4)

    transformed = mlb.transform_topk_inputs(logits, bias)

    assert transformed[0] is logits
    assert transformed[1] is bias


@pytest.mark.unit
@pytest.mark.parametrize("name", ["static", "dynamic"])
def test_generic_replica_entry_points_keep_dispatch_inside_mlb(name):
    placement = _placement()
    logical_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    logits = torch.zeros((2, 4))

    transformed, _ = transform_topk_inputs(None, logits, None)
    physical_ids = route_replicas(
        name,
        logical_ids,
        default_physical_for_logical=placement.default_physical_for_logical,
        logical_to_physical_candidates=placement.logical_to_physical_candidates,
        logical_to_physical_count=placement.logical_to_physical_count,
    )

    assert physical_ids.shape == logical_ids.shape
    assert transformed is logits


@pytest.mark.unit
def test_fake_transform_is_independent_from_replica_routing():
    logits = torch.zeros((2, 4))

    transformed, _ = transform_topk_inputs("fake", logits, None)

    assert transformed is logits
    assert torch.all((transformed >= 5) & (transformed < 10))


@pytest.mark.unit
@pytest.mark.parametrize("name", ["static", "dynamic"])
def test_generic_replica_dispatch_is_fullgraph_compilable(name):
    placement = _placement()

    def dispatch(logical_ids):
        return route_replicas(
            name,
            logical_ids,
            default_physical_for_logical=placement.default_physical_for_logical,
            logical_to_physical_candidates=placement.logical_to_physical_candidates,
            logical_to_physical_count=placement.logical_to_physical_count,
        )

    compiled = torch.compile(dispatch, backend="eager", fullgraph=True)
    logical_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

    assert compiled(logical_ids).shape == logical_ids.shape
