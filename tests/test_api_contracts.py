from dataclasses import FrozenInstanceError

import pytest

from moe_load_balancer import (
    PlacementProvider,
    PlacementSnapshot,
    RoutingDecision,
    RoutingLoadHistory,
    RoutingLoadHistoryProvider,
    RoutingPolicyConfig,
    RoutingRequest,
    RoutingStage,
    TokenRouter,
)
from moe_load_balancer.adapters.sglang import SGLangPlacementAdapter


class _PassthroughRouter:
    def route_tokens(self, request: RoutingRequest) -> RoutingDecision:
        return RoutingDecision(
            routed_physical_topk_ids=request.logical_topk_ids,
            topk_weights=request.topk_weights,
        )


class _LoadHistoryProvider:
    def __init__(self, view: RoutingLoadHistory):
        self._view = view

    def load_history(self, layer_id: int) -> RoutingLoadHistory:
        return self._view


def _placement(num_logical: int = 2, num_physical: int = 2) -> PlacementSnapshot:
    return PlacementSnapshot(
        layer_id=0,
        num_logical_experts=num_logical,
        num_physical_experts=num_physical,
        logical_to_physical_candidates=[[i] for i in range(num_logical)],
        logical_to_physical_count=[1] * num_logical,
    )


@pytest.mark.unit
class TestRoutingStage:
    def test_enum_values(self):
        assert RoutingStage.PREFILL.value == "prefill"
        assert RoutingStage.DECODE.value == "decode"
        assert RoutingStage.MIXED.value == "mixed"
        assert RoutingStage.IDLE.value == "idle"
        assert RoutingStage.SPECULATIVE.value == "speculative"

    def test_from_str_normalizes(self):
        assert RoutingStage.from_str("prefill") is RoutingStage.PREFILL
        assert RoutingStage.from_str("PREFILL") is RoutingStage.PREFILL
        assert RoutingStage.from_str(RoutingStage.DECODE) is RoutingStage.DECODE

    def test_from_str_rejects_unknown(self):
        with pytest.raises(ValueError):
            RoutingStage.from_str("not-a-stage")

    def test_from_str_rejects_non_string(self):
        with pytest.raises(TypeError):
            RoutingStage.from_str(None)  # type: ignore[arg-type]


@pytest.mark.unit
class TestRoutingPolicyConfig:
    def test_default_policy_is_not_stage_gated(self):
        policy = RoutingPolicyConfig()

        assert policy.name == "disabled"
        assert policy.is_disabled
        assert policy.allows_stage(None)
        assert policy.allows_stage(RoutingStage.PREFILL)
        assert policy.allows_stage(RoutingStage.DECODE)
        assert not policy.should_run_for_stage(RoutingStage.PREFILL)

    def test_explicit_all_stage_set_still_requires_known_stage(self):
        policy = RoutingPolicyConfig(
            name="waterfill",
            enabled_stages=frozenset(RoutingStage),
        )

        assert policy.allows_stage(RoutingStage.PREFILL)
        assert not policy.allows_stage(None)

    def test_enabled_stages_gate_request_stage(self):
        policy = RoutingPolicyConfig(
            name="waterfill",
            enabled_stages=frozenset({RoutingStage.PREFILL, RoutingStage.DECODE}),
        )

        assert policy.allows_stage(RoutingStage.PREFILL)
        assert policy.allows_stage(RoutingStage.DECODE)
        assert not policy.allows_stage(RoutingStage.MIXED)
        assert not policy.allows_stage(None)
        assert policy.should_run_for_stage(RoutingStage.PREFILL)
        assert not policy.should_run_for_stage(RoutingStage.MIXED)

    def test_enabled_stages_rejects_empty_set(self):
        with pytest.raises(ValueError):
            RoutingPolicyConfig(name="waterfill", enabled_stages=frozenset())

    def test_from_value_accepts_string_policy_names(self):
        policy = RoutingPolicyConfig.from_value("waterfill")

        assert policy.name == "waterfill"
        assert not policy.is_disabled

    def test_from_value_rejects_unsupported_types(self):
        with pytest.raises(TypeError):
            RoutingPolicyConfig.from_value(None)  # type: ignore[arg-type]

    def test_metadata_is_read_only(self):
        policy = RoutingPolicyConfig(name="waterfill", metadata={"experts_per_rank": 4})

        assert policy.metadata["experts_per_rank"] == 4
        with pytest.raises(TypeError):
            policy.metadata["experts_per_rank"] = 8  # type: ignore[index]


@pytest.mark.unit
class TestFrozenDataclasses:
    def test_placement_snapshot_is_frozen(self):
        snap = _placement()
        with pytest.raises(FrozenInstanceError):
            snap.layer_id = 99  # type: ignore[misc]

    def test_routing_request_is_frozen(self):
        req = RoutingRequest(
            layer_id=0,
            stage=RoutingStage.PREFILL,
            logical_topk_ids=[[0]],
            topk_weights=[[1.0]],
            placement=_placement(),
        )
        with pytest.raises(FrozenInstanceError):
            req.layer_id = 1  # type: ignore[misc]

    def test_routing_request_placement_is_optional(self):
        # Algorithms like Waterfill skip logical->physical translation and pass None.
        req = RoutingRequest(
            layer_id=0,
            logical_topk_ids=[[0]],
            topk_weights=[[1.0]],
        )
        assert req.placement is None

    def test_routing_decision_is_frozen(self):
        decision = RoutingDecision(routed_physical_topk_ids=[[0]], topk_weights=[[1.0]])
        with pytest.raises(FrozenInstanceError):
            decision.routed_physical_topk_ids = [[1]]  # type: ignore[misc]

    def test_routing_load_history_is_frozen(self):
        snapshot = RoutingLoadHistory(historical_rank_load=[1, 2])
        with pytest.raises(FrozenInstanceError):
            snapshot.historical_rank_load = [3, 4]  # type: ignore[misc]


@pytest.mark.unit
class TestProtocolConformance:
    def test_local_router_satisfies_token_router(self):
        assert isinstance(_PassthroughRouter(), TokenRouter)

    def test_local_load_history_provider_satisfies_protocol(self):
        provider = _LoadHistoryProvider(RoutingLoadHistory(historical_rank_load=[1, 2]))
        assert isinstance(provider, RoutingLoadHistoryProvider)

    def test_sglang_placement_adapter_satisfies_provider(self):
        # Construction succeeds without SGLang installed; isinstance against
        # a runtime_checkable Protocol verifies the method shape only.
        adapter = SGLangPlacementAdapter(metadata=object())
        assert isinstance(adapter, PlacementProvider)


@pytest.mark.unit
class TestTokenRouterProtocol:
    def test_local_passthrough_router(self):
        request = RoutingRequest(
            layer_id=0,
            stage=RoutingStage.PREFILL,
            logical_topk_ids=[[0, 1]],
            topk_weights=[[0.7, 0.3]],
            placement=_placement(num_logical=2, num_physical=2),
        )
        decision = _PassthroughRouter().route_tokens(request)
        assert decision.routed_physical_topk_ids == request.logical_topk_ids
        assert decision.topk_weights == request.topk_weights

    def test_stage_is_optional_context(self):
        request = RoutingRequest(
            layer_id=0,
            logical_topk_ids=[[0]],
            topk_weights=[[1.0]],
            policy=RoutingPolicyConfig(name="waterfill"),
        )

        assert request.stage is None
        assert request.policy.allows_stage(request.stage)
        assert request.policy.should_run_for_stage(request.stage)

    def test_request_coerces_string_policy_for_compatibility(self):
        request = RoutingRequest(
            layer_id=0,
            logical_topk_ids=[[0]],
            topk_weights=[[1.0]],
            policy="waterfill",
        )

        assert request.policy == RoutingPolicyConfig(name="waterfill")

    def test_load_inputs_can_be_live_references(self):
        routed_rank_load = [1, 2]
        active_rank_token_count = [10, 20]
        request = RoutingRequest(
            layer_id=0,
            logical_topk_ids=[[0]],
            topk_weights=[[1.0]],
            policy="waterfill",
            routed_rank_load=routed_rank_load,
            active_rank_token_count=active_rank_token_count,
        )

        # Adapters may pass live tensor references; mutating the underlying
        # sequence after RoutingRequest construction is observable through
        # the request, by design.
        routed_rank_load[1] = 3
        active_rank_token_count[0] = 30

        assert request.routed_rank_load is routed_rank_load
        assert request.active_rank_token_count is active_rank_token_count
        assert request.routed_rank_load == [1, 3]
        assert request.active_rank_token_count == [30, 20]

    def test_request_rejects_positional_construction(self):
        with pytest.raises(TypeError):
            RoutingRequest(0, [[0]], [[1.0]], _placement())  # type: ignore[misc]


@pytest.mark.unit
class TestSGLangPlacementAdapter:
    """Exercises ``SGLangPlacementAdapter`` against a synthetic
    ``ExpertLocationMetadata``-shaped object (sglang itself is not imported).
    """

    @staticmethod
    def _fake_metadata():
        import torch

        # Synthetic per-layer placement: 3 layers, 4 logical experts, 8 physical
        # slots (2 replicas each), ep_size=2 → 4 local physical experts per rank.
        num_layers, num_logical, num_physical, max_repl, ep_size = 3, 4, 8, 2, 2
        num_local_physical = num_physical // ep_size

        # logical_to_all_physical_map[L, E] = list of physical slots, padded -1
        log2allphy = torch.full(
            (num_layers, num_logical, max_repl), -1, dtype=torch.int64
        )
        # Each logical e gets physical slots [e*2, e*2+1].
        for L in range(num_layers):
            for e in range(num_logical):
                log2allphy[L, e, 0] = 2 * e
                log2allphy[L, e, 1] = 2 * e + 1
        log2allphy_num_valid = torch.full(
            (num_layers, num_logical), max_repl, dtype=torch.int64
        )

        # physical_to_logical_map[L, P] = which logical e owns physical slot P
        phy2log = torch.zeros((num_layers, num_physical), dtype=torch.int64)
        for L in range(num_layers):
            for p in range(num_physical):
                phy2log[L, p] = p // max_repl

        # Per-rank dispatch table: rank 0 dispatches logical e to slot 2e, rank
        # 1 dispatches to slot 2e+1. The adapter exposes the full table; rank
        # selection happens at the call site.
        rank_dispatch = torch.zeros((num_layers, num_logical), dtype=torch.int64)
        for L in range(num_layers):
            for e in range(num_logical):
                rank_dispatch[L, e] = 2 * e  # rank 0's choice as a placeholder

        class _FakeEL:
            num_logical_experts = num_logical
            num_physical_experts = num_physical
            num_local_physical_experts = num_local_physical
            logical_to_all_physical_map = log2allphy
            logical_to_all_physical_map_num_valid = log2allphy_num_valid
            physical_to_logical_map = phy2log
            logical_to_rank_dispatch_physical_map = rank_dispatch

        return _FakeEL(), num_logical, num_physical, num_local_physical

    def test_snapshot_maps_fields_for_one_layer(self):
        import torch

        em, num_logical, num_physical, num_local_physical = self._fake_metadata()
        adapter = SGLangPlacementAdapter(metadata=em)

        snap = adapter.snapshot(layer_id=1)

        assert snap.layer_id == 1
        assert snap.num_logical_experts == num_logical
        assert snap.num_physical_experts == num_physical
        # Per-layer slices are direct views of the underlying tensors.
        assert torch.equal(
            snap.logical_to_physical_candidates, em.logical_to_all_physical_map[1]
        )
        assert torch.equal(
            snap.logical_to_physical_count,
            em.logical_to_all_physical_map_num_valid[1],
        )
        assert snap.default_physical_for_logical is not None
        assert torch.equal(
            snap.default_physical_for_logical,
            em.logical_to_rank_dispatch_physical_map[1],
        )
        # physical_to_rank_map = arange(num_physical) // num_local_physical
        expected_p2r = (
            torch.arange(num_physical, dtype=torch.int64) // num_local_physical
        )
        assert torch.equal(snap.physical_to_rank_map, expected_p2r)
        assert torch.equal(
            snap.metadata["physical_to_logical_map"], em.physical_to_logical_map[1]
        )

    def test_physical_to_rank_map_is_cached_across_layers(self):
        em, *_ = self._fake_metadata()
        adapter = SGLangPlacementAdapter(metadata=em)
        s0 = adapter.snapshot(layer_id=0)
        s2 = adapter.snapshot(layer_id=2)
        # Same tensor object — derived once, shared across snapshots.
        assert s0.physical_to_rank_map is s2.physical_to_rank_map

    def test_default_physical_is_none_when_dispatch_table_missing(self):
        em, *_ = self._fake_metadata()
        em.logical_to_rank_dispatch_physical_map = None
        adapter = SGLangPlacementAdapter(metadata=em)
        snap = adapter.snapshot(layer_id=0)
        assert snap.default_physical_for_logical is None

    def test_snapshot_reflects_live_rebalance(self):
        """``ExpertLocationMetadata`` rewrite in place is observable via the adapter."""
        import torch

        em, *_ = self._fake_metadata()
        adapter = SGLangPlacementAdapter(metadata=em)
        before = adapter.snapshot(layer_id=0)
        # Simulate EPLBManager rewriting placement on the same metadata object.
        em.logical_to_all_physical_map = em.logical_to_all_physical_map.flip(dims=(1,))
        after = adapter.snapshot(layer_id=0)
        assert not torch.equal(
            before.logical_to_physical_candidates,
            after.logical_to_physical_candidates,
        )
