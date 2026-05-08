import pytest

from moe_load_balancer import (
    PlacementSnapshot,
    RoutingPolicyConfig,
    RoutingRequest,
    RoutingStage,
)
from moe_load_balancer.policies.l2 import (
    SharedExpertWaterfillPolicy,
)


def _placement() -> PlacementSnapshot:
    return PlacementSnapshot(
        layer_id=0,
        num_logical_experts=2,
        num_physical_experts=4,
        logical_to_physical_candidates=[[0, 1], [2, 3]],
        logical_to_physical_count=[2, 2],
        default_physical_for_logical=[0, 2],
        physical_to_rank_map=[0, 0, 1, 1],
    )


@pytest.mark.unit
class TestSharedExpertWaterfillPolicy:
    def test_selects_underloaded_rank_from_current_rank_load(self):
        request = RoutingRequest(
            layer_id=0,
            logical_topk_ids=[[0, 1], [1, 0]],
            topk_weights=[[0.6, 0.4], [0.7, 0.3]],
            placement=_placement(),
            current_rank_load=[10, 1],
            policy=RoutingPolicyConfig(name="waterfill", metadata={"local_rank": 0}),
        )

        decision = SharedExpertWaterfillPolicy().route_tokens(request)

        assert decision.routed_physical_topk_ids == [[0, 2], [2, 0]]
        assert decision.topk_weights is request.topk_weights
        assert decision.shared_expert_rank == [1, 1]
        assert decision.metadata["target_total"] == 8

    def test_local_preference_breaks_fallback_ties(self):
        request = RoutingRequest(
            layer_id=0,
            logical_topk_ids=[[0]],
            topk_weights=[[1.0]],
            placement=_placement(),
            current_rank_load=[10, 10],
            policy=RoutingPolicyConfig(
                name="waterfill",
                metadata={"local_rank": 0, "target_total": 10},
            ),
        )

        decision = SharedExpertWaterfillPolicy().route_tokens(request)

        assert decision.shared_expert_rank == [0]

    def test_candidate_ranks_can_be_limited_to_local_and_routed_ranks(self):
        placement = PlacementSnapshot(
            layer_id=0,
            num_logical_experts=1,
            num_physical_experts=3,
            logical_to_physical_candidates=[[1]],
            logical_to_physical_count=[1],
            default_physical_for_logical=[1],
            physical_to_rank_map=[0, 1, 2],
        )
        request = RoutingRequest(
            layer_id=0,
            logical_topk_ids=[[0]],
            topk_weights=[[1.0]],
            placement=placement,
            current_rank_load=[100, 10, 0],
            policy=RoutingPolicyConfig(
                name="waterfill",
                metadata={"local_rank": 0, "allow_all_ranks": False},
            ),
        )

        decision = SharedExpertWaterfillPolicy().route_tokens(request)

        assert decision.routed_physical_topk_ids == [[1]]
        assert decision.shared_expert_rank == [1]

    def test_target_total_zero_uses_derived_target_like_pr_sentinel(self):
        request = RoutingRequest(
            layer_id=0,
            logical_topk_ids=[[0, 1]],
            topk_weights=[[0.6, 0.4]],
            placement=_placement(),
            current_rank_load=[10, 1],
            policy=RoutingPolicyConfig(
                name="waterfill",
                metadata={"local_rank": 0, "target_total": 0},
            ),
        )

        decision = SharedExpertWaterfillPolicy().route_tokens(request)

        assert decision.metadata["target_total"] == 8
        assert decision.shared_expert_rank == [1]

    def test_reference_policy_can_use_dynamic_routed_rank_load_for_target(self):
        request = RoutingRequest(
            layer_id=0,
            logical_topk_ids=[[0, 1, 0, 1, 0, 1, 0, 1]],
            topk_weights=[[0.125] * 8],
            placement=_placement(),
            current_rank_load=[12, 12],
            policy=RoutingPolicyConfig(
                name="waterfill",
                metadata={
                    "local_rank": 0,
                    "routed_rank_load": [8, 8],
                },
            ),
        )

        decision = SharedExpertWaterfillPolicy().route_tokens(request)

        assert decision.metadata["target_total"] == 13

    def test_bool_like_allow_all_ranks_metadata_is_parsed_explicitly(self):
        placement = PlacementSnapshot(
            layer_id=0,
            num_logical_experts=1,
            num_physical_experts=3,
            logical_to_physical_candidates=[[1]],
            logical_to_physical_count=[1],
            default_physical_for_logical=[1],
            physical_to_rank_map=[0, 1, 2],
        )
        request = RoutingRequest(
            layer_id=0,
            logical_topk_ids=[[0]],
            topk_weights=[[1.0]],
            placement=placement,
            current_rank_load=[100, 10, 0],
            policy=RoutingPolicyConfig(
                name="waterfill",
                metadata={"local_rank": 0, "allow_all_ranks": "false"},
            ),
        )

        decision = SharedExpertWaterfillPolicy().route_tokens(request)

        assert decision.shared_expert_rank == [1]

    def test_integer_allow_all_ranks_metadata_is_parsed_explicitly(self):
        placement = PlacementSnapshot(
            layer_id=0,
            num_logical_experts=1,
            num_physical_experts=3,
            logical_to_physical_candidates=[[1]],
            logical_to_physical_count=[1],
            default_physical_for_logical=[1],
            physical_to_rank_map=[0, 1, 2],
        )
        request = RoutingRequest(
            layer_id=0,
            logical_topk_ids=[[0]],
            topk_weights=[[1.0]],
            placement=placement,
            current_rank_load=[100, 10, 0],
            policy=RoutingPolicyConfig(
                name="waterfill",
                metadata={"local_rank": 0, "allow_all_ranks": 0},
            ),
        )

        decision = SharedExpertWaterfillPolicy().route_tokens(request)

        assert decision.shared_expert_rank == [1]

    def test_stage_gating_skips_policy_without_changing_routed_topk(self):
        request = RoutingRequest(
            layer_id=0,
            stage=RoutingStage.DECODE,
            logical_topk_ids=[[0]],
            topk_weights=[[1.0]],
            placement=_placement(),
            current_rank_load=[10, 1],
            policy=RoutingPolicyConfig(
                name="waterfill",
                enabled_stages=frozenset({RoutingStage.PREFILL}),
            ),
        )

        decision = SharedExpertWaterfillPolicy().route_tokens(request)

        assert decision.routed_physical_topk_ids == [[0]]
        assert decision.shared_expert_rank is None
        assert decision.metadata["skipped"]

    def test_requires_request_local_rank_load(self):
        request = RoutingRequest(
            layer_id=0,
            logical_topk_ids=[[0]],
            topk_weights=[[1.0]],
            placement=_placement(),
            policy="waterfill",
        )

        with pytest.raises(ValueError, match="current_rank_load"):
            SharedExpertWaterfillPolicy().route_tokens(request)

    def test_rejects_negative_rank_load(self):
        request = RoutingRequest(
            layer_id=0,
            logical_topk_ids=[[0]],
            topk_weights=[[1.0]],
            placement=_placement(),
            current_rank_load=[1, -1],
            policy="waterfill",
        )

        with pytest.raises(ValueError, match="non-negative"):
            SharedExpertWaterfillPolicy().route_tokens(request)

    def test_invalid_logical_expert_id_raises_value_error(self):
        request = RoutingRequest(
            layer_id=0,
            logical_topk_ids=[[99]],
            topk_weights=[[1.0]],
            placement=_placement(),
            current_rank_load=[1, 1],
            policy="waterfill",
        )

        with pytest.raises(ValueError, match="logical expert id 99"):
            SharedExpertWaterfillPolicy().route_tokens(request)

    def test_empty_candidate_row_raises_value_error(self):
        placement = PlacementSnapshot(
            layer_id=0,
            num_logical_experts=1,
            num_physical_experts=0,
            logical_to_physical_candidates=[[]],
            logical_to_physical_count=[1],
        )
        request = RoutingRequest(
            layer_id=0,
            logical_topk_ids=[[0]],
            topk_weights=[[1.0]],
            placement=placement,
            current_rank_load=[1],
            policy="waterfill",
        )

        with pytest.raises(ValueError, match="no physical candidates"):
            SharedExpertWaterfillPolicy().route_tokens(request)

    def test_invalid_physical_expert_id_raises_value_error(self):
        placement = PlacementSnapshot(
            layer_id=0,
            num_logical_experts=1,
            num_physical_experts=1,
            logical_to_physical_candidates=[[2]],
            logical_to_physical_count=[1],
        )
        request = RoutingRequest(
            layer_id=0,
            logical_topk_ids=[[0]],
            topk_weights=[[1.0]],
            placement=placement,
            current_rank_load=[1],
            policy="waterfill",
        )

        with pytest.raises(ValueError, match="physical expert id 2"):
            SharedExpertWaterfillPolicy().route_tokens(request)
