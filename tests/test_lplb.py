"""Full joint LPLB policy tests."""

from __future__ import annotations

import torch

from moe_load_balancer.core.types import (
    PlacementSnapshot,
    RoutingPolicyConfig,
    RoutingRequest,
)
from moe_load_balancer.policies.l2.lplb import (
    LPLBL2Router,
    TorchReferenceIPMBackend,
)


def _snapshot() -> PlacementSnapshot:
    # Rank 0 owns physical 0..2; rank 1 owns 3..5.
    # Logical 0/1 are single-copy; logical 2/3 have one copy on each rank.
    return PlacementSnapshot(
        layer_id=0,
        num_logical_experts=4,
        num_physical_experts=6,
        ep_size=2,
        num_local_physical_experts=3,
        physical_to_logical_map=torch.tensor([0, 2, 3, 1, 2, 3]),
        logical_to_physical_candidates=torch.tensor(
            [[0, -1], [3, -1], [1, 4], [2, 5]]
        ),
        logical_to_physical_count=torch.tensor([1, 1, 2, 2]),
    )


def test_joint_lp_moves_replicated_load_away_from_loaded_rank():
    router = LPLBL2Router(
        ep_size=2,
        num_physical_experts=6,
        ipm_backend=TorchReferenceIPMBackend(),
    )
    state = router._get_state(_snapshot())
    # Rank 0 already has 20 fixed tokens. The two replicated logical experts
    # contribute 20 more tokens, so the joint optimum sends them to rank 1.
    router._prepare_inputs(state, torch.tensor([20.0, 0.0, 10.0, 10.0]))
    router.ipm_backend.solve(state.a_full, state.b, state.c, result=state.x)
    router._extract_probability(state)

    assert state.probability[2, 1] > 1000 * state.probability[2, 0]
    assert state.probability[3, 1] > 1000 * state.probability[3, 0]
    assert torch.allclose(
        state.a_full @ state.x,
        state.b,
        atol=5e-4,
        rtol=0,
    )


class _DeterministicBackend:
    def __init__(self):
        self.warmups = []

    def warmup(self, nc, nv, *, device):
        self.warmups.append((nc, nv, device.type))

    def solve(self, a, b, c, *, result):
        del a, b, c
        result.fill_(0.0)
        # Replicated physical order is [1, 2, 4, 5]. Pick rank 1 for both.
        result[:4].copy_(torch.tensor([0.0, 0.0, 1.0, 1.0]))
        return result


def test_route_uses_backend_and_returns_only_valid_physical_copies():
    backend = _DeterministicBackend()
    router = LPLBL2Router(
        ep_size=2,
        num_physical_experts=6,
        ipm_backend=backend,
    )
    snapshot = _snapshot()
    router.prepare_layer(snapshot)
    request = RoutingRequest(
        layer_id=0,
        logical_topk_ids=torch.tensor([[2, 3], [3, 2]]),
        topk_weights=torch.ones((2, 2)),
        policies=(RoutingPolicyConfig(name="lplb"),),
        placement=snapshot,
        global_logical_count=torch.tensor([20.0, 0.0, 10.0, 10.0]),
    )
    decision = router.route_tokens(request, request.policies[0])

    assert backend.warmups and len(backend.warmups) == 1
    assert torch.equal(decision.routed_physical_topk_ids, torch.tensor([[4, 5], [5, 4]]))


def test_prepare_rebuilds_after_in_place_placement_update():
    backend = _DeterministicBackend()
    router = LPLBL2Router(
        ep_size=2,
        num_physical_experts=6,
        ipm_backend=backend,
    )
    snapshot = _snapshot()
    router.prepare_layer(snapshot)
    first = router._layer_states[0]

    snapshot.logical_to_physical_candidates.copy_(
        torch.tensor([[0, -1], [3, -1], [4, 1], [5, 2]])
    )
    router.prepare_layer(snapshot)
    second = router._layer_states[0]

    assert second is not first
    assert len(backend.warmups) == 2


def test_prepare_rebuilds_after_physical_map_update():
    backend = _DeterministicBackend()
    router = LPLBL2Router(
        ep_size=2,
        num_physical_experts=6,
        ipm_backend=backend,
    )
    snapshot = _snapshot()
    router.prepare_layer(snapshot)
    first = router._layer_states[0]

    snapshot.physical_to_logical_map.copy_(torch.tensor([0, 3, 2, 1, 3, 2]))
    router.prepare_layer(snapshot)

    assert router._layer_states[0] is not first
    assert len(backend.warmups) == 2


def test_padded_candidate_table_is_compacted_once_per_placement():
    backend = _DeterministicBackend()
    router = LPLBL2Router(
        ep_size=2,
        num_physical_experts=6,
        ipm_backend=backend,
    )
    compact = _snapshot()
    padded = torch.full((4, 6), -1, dtype=torch.int64)
    padded[:, :2].copy_(compact.logical_to_physical_candidates)
    snapshot = PlacementSnapshot(
        layer_id=compact.layer_id,
        num_logical_experts=compact.num_logical_experts,
        num_physical_experts=compact.num_physical_experts,
        ep_size=compact.ep_size,
        num_local_physical_experts=compact.num_local_physical_experts,
        physical_to_logical_map=compact.physical_to_logical_map,
        logical_to_physical_candidates=padded,
        logical_to_physical_count=compact.logical_to_physical_count,
    )

    first = router._get_state(snapshot)
    second = router._get_state(snapshot)

    assert first is second
    assert first.candidates.shape == (4, 2)
    assert torch.equal(first.candidates, compact.logical_to_physical_candidates)
    assert len(backend.warmups) == 1


def test_orchestrator_prepare_routing_layer_is_generic():
    from moe_load_balancer import MoELoadBalancer

    backend = _DeterministicBackend()
    router = LPLBL2Router(
        ep_size=2,
        num_physical_experts=6,
        ipm_backend=backend,
    )
    balancer = MoELoadBalancer(routing_policies={"lplb": router})
    balancer.prepare_routing_layer("lplb", _snapshot())
    assert len(backend.warmups) == 1
