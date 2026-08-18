from __future__ import annotations

import pytest
import torch
from dataclasses import replace

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
def test_dynamic_policy_spreads_without_drawing_from_an_rng():
    """Dynamic spreads a logical expert's traffic evenly over its replicas, and
    decides which token takes which one from the token index.

    This test used to pin the choice to torch.randint under a fixed seed, which
    made it pass while leaving the behaviour neither reproducible in a real run
    nor agreed upon between ranks that see the same token and hold different
    experts. The distribution is the policy's decision; the draw that realises
    it is part of the contract, not of whichever implementation applies it.
    """
    logical_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    placement = _placement()

    first = route_dynamic_replicas(
        logical_ids,
        placement.logical_to_physical_candidates,
        placement.logical_to_physical_count,
    )
    second = route_dynamic_replicas(
        logical_ids,
        placement.logical_to_physical_candidates,
        placement.logical_to_physical_count,
    )
    assert torch.equal(first, second), "the same input must route the same way"

    # Every choice is a live replica of the expert that was asked for.
    for row in range(logical_ids.shape[0]):
        for col in range(logical_ids.shape[1]):
            lg = int(logical_ids[row, col])
            live = placement.logical_to_physical_candidates[
                lg, : int(placement.logical_to_physical_count[lg])
            ]
            assert int(first[row, col]) in set(live.tolist())


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


@pytest.mark.unit
class TestDeferredDispatchIsHonouredByEveryReplicaPolicy:
    """A runtime that fuses replica selection with expert-load recording cannot
    accept per-token ids without also taking over the recording, so it asks for
    a share table instead. A policy that ignores the request produces no table,
    the caller falls back to its built-in choice, and the policy silently never
    runs -- which is what happened to static and dynamic on vLLM."""

    @staticmethod
    def _snapshot():
        return PlacementSnapshot(
            layer_id=0,
            num_logical_experts=4,
            num_physical_experts=6,
            ep_size=2,
            num_local_physical_experts=3,
            physical_to_logical_map=torch.tensor([0, 1, 2, 3, 0, 1]),
            logical_to_physical_candidates=torch.tensor(
                [[0, 4], [1, 5], [2, -1], [3, -1]]
            ),
            logical_to_physical_count=torch.tensor([2, 2, 1, 1]),
            default_physical_for_logical=torch.tensor([4, 1, 2, 3]),
        )

    def _request(self, policy_name):
        return RoutingRequest(
            layer_id=0,
            logical_topk_ids=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            topk_weights=torch.ones((2, 2)),
            placement=self._snapshot(),
            defer_dispatch=True,
            policies=(RoutingPolicyConfig(name=policy_name),),
        )

    def test_static_puts_all_of_an_expert_on_its_default_replica(self):
        decision = StaticL2Router().route_tokens(
            self._request("static"), RoutingPolicyConfig(name="static")
        )
        sel = decision.metadata["replica_selection"]
        assert sel is not None
        # logical 0's default is slot 4, which sits in candidate column 1.
        assert sel.probability[0].tolist() == [0.0, 1.0]
        # logical 1's default is slot 1, column 0.
        assert sel.probability[1].tolist() == [1.0, 0.0]
        # A single-replica expert keeps its one column and pads the rest.
        assert sel.probability[2].tolist() == [1.0, 0.0]

    def test_dynamic_spreads_an_expert_evenly_over_its_replicas(self):
        decision = DynamicL2Router().route_tokens(
            self._request("dynamic"), RoutingPolicyConfig(name="dynamic")
        )
        sel = decision.metadata["replica_selection"]
        assert sel.probability[0].tolist() == [0.5, 0.5]
        assert sel.probability[2].tolist() == [1.0, 0.0]

    def test_padding_columns_stay_zero(self):
        """A consumer walks a row without consulting the replica count, so a
        column past the live ones must never carry weight."""
        for router in (StaticL2Router(), DynamicL2Router()):
            name = "static" if isinstance(router, StaticL2Router) else "dynamic"
            sel = router.route_tokens(
                self._request(name), RoutingPolicyConfig(name=name)
            ).metadata["replica_selection"]
            assert sel.probability[2, 1].item() == 0.0
            assert sel.probability[3, 1].item() == 0.0

    def test_per_token_ids_are_still_produced_when_not_deferring(self):
        req = self._request("static")
        req = replace(req, defer_dispatch=False)
        decision = StaticL2Router().route_tokens(
            req, RoutingPolicyConfig(name="static")
        )
        assert "replica_selection" not in decision.metadata
        assert decision.routed_physical_topk_ids.tolist() == [[4, 1], [2, 3]]
