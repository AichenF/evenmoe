from __future__ import annotations

from dataclasses import replace

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
from moe_load_balancer.kernels.ops.counting import count_routed_per_rank


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


class _PlannedPhysicalRouter(_PhysicalRouter):
    # Stands in for LPLB, which declares GlobalLogicalCountConsumer: the
    # orchestrator gathers the count on the capability, not on the policy name.
    def needs_global_logical_count(self, request):
        del request
        return True

    def route_tokens(self, request, policy):
        decision = super().route_tokens(request, policy)
        return RoutingDecision(
            routed_physical_topk_ids=decision.routed_physical_topk_ids,
            topk_weights=decision.topk_weights,
            metadata={"planned_rank_load": torch.tensor([7, 1])},
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


class _RecordingCollectives:
    rank = 0
    world_size = 2

    def __init__(self, peer_contribution):
        self.peer_contribution = peer_contribution
        self.payloads = []

    def all_reduce_sum(self, payload):
        self.payloads.append(payload.clone())
        return payload + self.peer_contribution.to(payload)


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
        global_logical_count=torch.tensor([2.0, 2.0]),
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
def test_rank_load_counter_extension_keeps_domain_input_contract():
    captured = {}
    placement = _placement()
    policy = RoutingPolicyConfig(name="waterfill")

    def count_rank_load(physical_topk_ids, received_placement, received_policy):
        captured["physical_topk_ids"] = physical_topk_ids
        captured["placement"] = received_placement
        captured["policy"] = received_policy
        return torch.tensor([1, 3])

    shared = _SharedRouter([])
    mlb = MoELoadBalancer(
        placement_planners={},
        routing_policies={"waterfill": shared},
        rank_load_counter=count_rank_load,
    )
    physical_topk_ids = torch.tensor([[0, 2], [2, 2]])
    request = RoutingRequest(
        layer_id=0,
        logical_topk_ids=torch.tensor([[0, 1], [1, 0]]),
        routed_physical_topk_ids=physical_topk_ids,
        topk_weights=torch.ones((2, 2)),
        placement=placement,
        policies=(policy,),
    )

    mlb.route_tokens(request)

    assert captured["physical_topk_ids"] is physical_topk_ids
    assert captured["placement"] is placement
    assert captured["policy"] is policy
    assert torch.equal(shared.seen_rank_load, torch.tensor([1, 3]))


@pytest.mark.unit
@pytest.mark.parametrize("rank_map_kind", ["contiguous", "list", "tensor", "metadata"])
def test_default_rank_counter_resolves_routing_metadata(rank_map_kind):
    placement = _placement()
    metadata = {"world_size": 2, "experts_per_rank": 2}
    expected = torch.tensor([1, 3])
    if rank_map_kind in {"list", "tensor"}:
        rank_map = [1, 0, 0, 1]
        if rank_map_kind == "tensor":
            rank_map = torch.tensor(rank_map)
        placement = replace(placement, physical_to_rank_map=rank_map)
        expected = torch.tensor([3, 1])
    if rank_map_kind == "metadata":
        placement = None
    else:
        # Committed placement takes precedence over policy metadata.
        metadata = {"world_size": 1, "experts_per_rank": 4}

    shared = _SharedRouter([])
    mlb = MoELoadBalancer(
        placement_planners={},
        routing_policies={"waterfill": shared},
    )
    mlb.route_tokens(
        RoutingRequest(
            layer_id=0,
            logical_topk_ids=torch.tensor([[0, 1, -1], [1, 1, -1]]),
            routed_physical_topk_ids=torch.tensor([[0, 2, -1], [2, 2, -1]]),
            topk_weights=torch.ones((2, 3)),
            placement=placement,
            policies=(RoutingPolicyConfig(name="waterfill", metadata=metadata),),
        )
    )

    assert torch.equal(shared.seen_rank_load, expected)


@pytest.mark.unit
def test_dynamic_waterfill_packs_rank_and_active_load_into_one_collective():
    calls = []
    shared = _SharedRouter(calls)
    collectives = _RecordingCollectives(torch.tensor([1, 1, 0, 3]))
    mlb = MoELoadBalancer(
        placement_planners={},
        routing_policies={
            "dynamic": _PhysicalRouter(calls),
            "waterfill": shared,
        },
        collectives=collectives,
    )
    request = RoutingRequest(
        layer_id=0,
        logical_topk_ids=torch.tensor([[0, 1], [0, 1]]),
        topk_weights=torch.ones((2, 2)),
        placement=_placement(),
        token_count=2,
        policies=(
            RoutingPolicyConfig(name="dynamic"),
            RoutingPolicyConfig(
                name="waterfill",
                metadata={"load_view": "global"},
            ),
        ),
    )

    mlb.route_tokens(request)

    assert torch.equal(collectives.payloads[0], torch.tensor([1, 3, 2, 0]))
    assert len(collectives.payloads) == 1
    assert torch.equal(shared.seen_rank_load, torch.tensor([2, 4]))
    assert torch.equal(shared.seen_active, torch.tensor([2, 3]))


@pytest.mark.unit
def test_lplb_waterfill_reuses_planned_load_after_one_collective():
    calls = []
    shared = _SharedRouter(calls)
    collectives = _RecordingCollectives(torch.tensor([1.0, 1.0]))
    mlb = MoELoadBalancer(
        placement_planners={},
        routing_policies={
            "lplb": _PlannedPhysicalRouter(calls),
            "waterfill": shared,
        },
        routing_pipeline="lplb+waterfill",
        collectives=collectives,
    )
    request = RoutingRequest(
        layer_id=0,
        logical_topk_ids=torch.tensor([[0, 1], [1, 0]]),
        topk_weights=torch.ones((2, 2)),
        placement=_placement(),
    )

    mlb.route_tokens(request)

    assert len(collectives.payloads) == 1
    assert torch.equal(collectives.payloads[0], torch.tensor([2.0, 2.0]))
    assert torch.equal(shared.seen_rank_load, torch.tensor([7, 1]))
    assert shared.seen_active is None


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
        global_logical_count=torch.tensor([1.0, 1.0]),
    )

    decision = mlb.route_tokens(request)

    assert torch.equal(decision.routed_physical_topk_ids, torch.tensor([[0, 1]]))
    assert torch.equal(decision.topk_weights, torch.full((1, 2), 0.5))


@pytest.mark.unit
def test_pipeline_hands_back_the_policy_metadata_it_produced():
    """A policy's metadata is part of its answer, not pipeline scratch space.

    LPLB's ``lp_probability`` and ``planned_rank_load`` are carried only there,
    so rebuilding the metadata around the pipeline's own bookkeeping would
    silently drop them: a caller diagnosing the solve offline would find
    nothing to read, with no error to say why.
    """

    class _TableRouter:
        phase = RoutingPhase.PHYSICAL_ROUTING

        def route_tokens(self, request, policy):
            del policy
            return RoutingDecision(
                routed_physical_topk_ids=request.logical_topk_ids,
                topk_weights=request.topk_weights,
                metadata={"lp_probability": "TABLE", "policy": "lplb"},
            )

    mlb = MoELoadBalancer(
        placement_planners={}, routing_policies={"lplb": _TableRouter()}
    )
    decision = mlb.route_tokens(
        RoutingRequest(
            layer_id=0,
            logical_topk_ids=torch.tensor([[0, 1]]),
            topk_weights=torch.ones((1, 2)),
            placement=_placement(),
            policies=(RoutingPolicyConfig(name="lplb"),),
            global_logical_count=torch.tensor([1.0, 1.0]),
        )
    )

    assert decision.metadata["lp_probability"] == "TABLE"
    assert decision.metadata["policy"] == "lplb"
    # The pipeline's own bookkeeping still wins its keys.
    assert decision.metadata["executed_policies"] == ("lplb",)
    assert decision.metadata["skipped_policies"] == ()


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
def test_rank_load_count_ignores_padded_slots_without_dynamic_indexing():
    counts = count_routed_per_rank(
        torch.tensor([[0, 3, -1], [2, -1, 1]]),
        world_size=2,
        experts_per_rank=2,
    )

    assert counts.dtype == torch.int64
    assert torch.equal(counts, torch.tensor([2, 2]))


@pytest.mark.unit
def test_default_rank_counter_does_not_keep_process_wide_async_scratch():
    calls = []
    shared = _SharedRouter(calls)
    mlb = MoELoadBalancer(
        placement_planners={},
        routing_policies={"waterfill": shared},
    )
    request = RoutingRequest(
        layer_id=0,
        logical_topk_ids=torch.tensor([[0, 1], [1, 0]]),
        routed_physical_topk_ids=torch.tensor([[0, 2], [2, 2]]),
        topk_weights=torch.ones((2, 2)),
        placement=_placement(),
        policies=(RoutingPolicyConfig(name="waterfill"),),
    )

    mlb.route_tokens(request)
    first = shared.seen_rank_load
    mlb.route_tokens(request)
    second = shared.seen_rank_load

    assert not hasattr(mlb, "_rank_load_buffers")
    assert torch.equal(first, second)


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


class _CountDeclaringPhysicalRouter:
    """A physical router that states whether it consumes the global count."""

    phase = RoutingPhase.PHYSICAL_ROUTING

    def __init__(self, needs: bool):
        self._needs = needs
        self.seen_count = object()

    def needs_global_logical_count(self, request):
        del request
        return self._needs

    def route_tokens(self, request, policy):
        del policy
        self.seen_count = request.global_logical_count
        return RoutingDecision(
            routed_physical_topk_ids=request.logical_topk_ids,
            topk_weights=request.topk_weights,
        )


def _count_request(**overrides) -> RoutingRequest:
    kwargs = {
        "layer_id": 0,
        "logical_topk_ids": torch.tensor([[0, 1]]),
        "topk_weights": torch.ones((1, 2)),
        "placement": _placement(),
        "policies": (RoutingPolicyConfig(name="lplb"),),
    }
    kwargs.update(overrides)
    return RoutingRequest(**kwargs)


def _mlb_with(router, collectives) -> MoELoadBalancer:
    return MoELoadBalancer(
        placement_planners={},
        routing_policies={"lplb": router},
        collectives=collectives,
    )


@pytest.mark.unit
def test_the_count_collective_is_skipped_when_the_policy_declares_no_need():
    """An LP with nothing to solve must not cost a collective.

    A layer whose logical experts each hold one replica gives LPLB no traffic
    split to decide, so it returns identity dispatch and never reads the count.
    Gathering it anyway is one EP collective per layer per forward for a result
    that is thrown away -- measurable as a straight loss on zero-redundancy
    configurations.
    """
    router = _CountDeclaringPhysicalRouter(needs=False)
    collectives = _RecordingCollectives(torch.tensor([1.0, 1.0]))

    _mlb_with(router, collectives).route_tokens(_count_request())

    assert collectives.payloads == []
    assert router.seen_count is None


@pytest.mark.unit
def test_the_count_collective_runs_when_the_policy_declares_it_needs_one():
    router = _CountDeclaringPhysicalRouter(needs=True)
    collectives = _RecordingCollectives(torch.tensor([1.0, 1.0]))

    _mlb_with(router, collectives).route_tokens(_count_request())

    assert len(collectives.payloads) == 1
    assert router.seen_count is not None


@pytest.mark.unit
def test_a_policy_that_never_declares_the_protocol_gets_no_count():
    """The need is a capability, not a policy name.

    The gate used to read ``policy_config.name == "lplb"``, which both taxed
    every LPLB layer regardless of redundancy and left any other count-consuming
    policy unable to ask.
    """
    router = _PassThroughPhysicalRouter()
    collectives = _RecordingCollectives(torch.tensor([1.0, 1.0]))

    _mlb_with(router, collectives).route_tokens(_count_request())

    assert collectives.payloads == []


@pytest.mark.unit
def test_a_caller_supplied_count_is_never_re_gathered():
    """vLLM accumulates per-layer counts inside its CUDA graph and all-reduces
    once per step for every layer at once, because a collective cannot be
    captured. MLB must pass that through, not issue a second one."""
    router = _CountDeclaringPhysicalRouter(needs=True)
    collectives = _RecordingCollectives(torch.tensor([1.0, 1.0]))
    supplied = torch.tensor([5.0, 7.0])

    _mlb_with(router, collectives).route_tokens(
        _count_request(global_logical_count=supplied)
    )

    assert collectives.payloads == []
    assert torch.equal(router.seen_count, supplied)
