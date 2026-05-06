from dataclasses import FrozenInstanceError

import pytest

from moe_load_balancer import (
    PlacementProvider,
    PlacementSnapshot,
    RoutingDecision,
    RoutingLoadSnapshot,
    RoutingPolicyConfig,
    RoutingRequest,
    RoutingStage,
    TokenRouter,
)
from moe_load_balancer.adapters.sglang import (
    SGLangPlacementAdapter,
    SGLangRoutingAdapter,
    SGLangRoutingContext,
)


class _PassthroughRouter:
    def route_tokens(self, request: RoutingRequest) -> RoutingDecision:
        return RoutingDecision(
            routed_physical_topk_ids=request.logical_topk_ids,
            topk_weights=request.topk_weights,
        )


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
        policy = RoutingPolicyConfig(name="waterfill", metadata={"mode": "static"})

        assert policy.metadata["mode"] == "static"
        with pytest.raises(TypeError):
            policy.metadata["mode"] = "dynamic"  # type: ignore[index]


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

    def test_routing_decision_is_frozen(self):
        decision = RoutingDecision(routed_physical_topk_ids=[[0]], topk_weights=[[1.0]])
        with pytest.raises(FrozenInstanceError):
            decision.routed_physical_topk_ids = [[1]]  # type: ignore[misc]

    def test_routing_load_snapshot_is_frozen(self):
        snapshot = RoutingLoadSnapshot(current_rank_load=[1, 2])
        with pytest.raises(FrozenInstanceError):
            snapshot.current_rank_load = [3, 4]  # type: ignore[misc]


@pytest.mark.unit
class TestProtocolConformance:
    def test_local_router_satisfies_token_router(self):
        assert isinstance(_PassthroughRouter(), TokenRouter)

    def test_sglang_placement_adapter_satisfies_provider(self):
        # Construction succeeds without SGLang installed; isinstance against
        # a runtime_checkable Protocol verifies the method shape only.
        adapter = SGLangPlacementAdapter(metadata_source=object())
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
            placement=_placement(num_logical=1, num_physical=1),
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
            placement=_placement(num_logical=1, num_physical=1),
            policy="waterfill",
        )

        assert request.policy == RoutingPolicyConfig(name="waterfill")

    def test_rank_load_fields_can_be_live_references(self):
        current_rank_load = [1, 2]
        prior_rank_load = [10, 20]
        request = RoutingRequest(
            layer_id=0,
            logical_topk_ids=[[0]],
            topk_weights=[[1.0]],
            placement=_placement(num_logical=1, num_physical=1),
            policy="waterfill",
            current_rank_load=current_rank_load,
            prior_rank_load=prior_rank_load,
        )

        current_rank_load[1] = 3
        prior_rank_load[0] = 30

        assert request.current_rank_load == [1, 3]
        assert request.prior_rank_load == [30, 20]

    def test_request_rejects_positional_construction(self):
        with pytest.raises(TypeError):
            RoutingRequest(0, [[0]], [[1.0]], _placement())  # type: ignore[misc]


@pytest.mark.unit
class TestSGLangAdapterStubs:
    def test_placement_adapter_is_unimplemented(self):
        adapter = SGLangPlacementAdapter(metadata_source=object())
        with pytest.raises(NotImplementedError):
            adapter.snapshot(layer_id=0)

    def test_routing_adapter_is_unimplemented(self):
        adapter = SGLangRoutingAdapter(
            router=_PassthroughRouter(),
            placement_provider=SGLangPlacementAdapter(metadata_source=object()),
        )
        context = SGLangRoutingContext(
            policy="disabled",
            stage=RoutingStage.PREFILL,
        )
        assert context.policy == RoutingPolicyConfig(name="disabled")
        with pytest.raises(NotImplementedError):
            adapter.route_topk(layer_id=0, logical_topk_output=None, context=context)
