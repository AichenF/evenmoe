from __future__ import annotations

import pytest
import torch

from moe_load_balancer import (
    ExpertDeploymentConfig,
    MoELoadBalancer,
    PlacementSnapshot,
    RoutingPipeline,
)


def _placement() -> PlacementSnapshot:
    return PlacementSnapshot(
        layer_id=3,
        num_logical_experts=2,
        num_physical_experts=4,
        ep_size=2,
        num_local_physical_experts=2,
        physical_to_logical_map=torch.tensor([0, 1, 0, 1]),
        logical_to_physical_candidates=torch.tensor([[0, 2], [1, 3]]),
        logical_to_physical_count=torch.tensor([2, 2]),
        default_physical_for_logical=torch.tensor([0, 1]),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "canonical", "replica", "post_topk"),
    [
        ("static", "static", "static", ("static",)),
        ("dynamic", "dynamic", "dynamic", ("dynamic",)),
        ("fake", "fake", "dynamic", ("dynamic",)),
        ("fake+static", "fake+static", "static", ("static",)),
        ("fake+lplb", "fake+lplb", "lplb", ("lplb",)),
        ("lp", "lplb", "lplb", ("lplb",)),
        ("waterfill", "static+waterfill", "static", ("static", "waterfill")),
        (
            "dynamic+waterfill_dynamic",
            "dynamic+waterfill_dynamic",
            "dynamic",
            ("dynamic", "waterfill"),
        ),
        ("lplb+waterfill", "lplb+waterfill", "lplb", ("lplb", "waterfill")),
    ],
)
def test_routing_pipeline_parses_and_canonicalizes(value, canonical, replica, post_topk):
    pipeline = RoutingPipeline.from_value(value)

    assert pipeline.name == canonical
    assert pipeline.policies[0] == replica
    assert tuple(policy.name for policy in pipeline.post_topk_policies) == post_topk


@pytest.mark.unit
def test_pipeline_capabilities_hide_algorithm_selection_from_framework_glue():
    pipeline = RoutingPipeline.from_value("lplb+waterfill")

    assert pipeline.capabilities.requires_post_topk_routing
    assert pipeline.capabilities.requires_placement_state
    assert pipeline.capabilities.routes_shared_expert
    assert pipeline.capabilities.requires_deepep
    assert not pipeline.capabilities.requires_rank_dispatch_map


@pytest.mark.unit
def test_default_replica_selection_is_owned_by_mlb():
    pipeline = RoutingPipeline.default_replica_routing()

    assert pipeline.name == "static"
    assert pipeline.capabilities.requires_rank_dispatch_map
    assert pipeline.capabilities.requires_post_topk_routing


@pytest.mark.unit
def test_fake_is_a_pre_topk_preset_with_dynamic_default():
    pipeline = RoutingPipeline.from_value("fake+waterfill")

    assert pipeline.name == "fake+waterfill"
    assert pipeline.input_transform == "fake"
    assert pipeline.policies[0] == "dynamic"
    assert tuple(policy.name for policy in pipeline.post_topk_policies) == (
        "dynamic",
        "waterfill",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        "static+dynamic",
        "waterfill+waterfill_dynamic",
        "lplb+waterfill_dynamic",
        "unknown",
        "static+",
    ],
)
def test_routing_pipeline_rejects_invalid_composition(value):
    with pytest.raises(ValueError):
        RoutingPipeline.from_value(value)


class _PreparingRouter:
    def __init__(self):
        self.seen = None

    def prepare_layer(self, placement):
        self.seen = placement


@pytest.mark.unit
def test_placement_commit_notifies_active_policy_without_policy_name():
    router = _PreparingRouter()
    mlb = MoELoadBalancer(
        placement_planners={},
        routing_policies={"lplb": router},
        routing_pipeline="lplb",
    )
    placement = _placement()

    assert mlb.on_placement_committed(placement) is None
    assert router.seen is placement


@pytest.mark.unit
class TestApplicabilityIsThePolicysCall:
    """Which deployments a policy can act on is a property of the policy.

    A framework that decides it by assumption -- "L2 means replica routing, so
    no redundant experts means nothing to do" -- switches off policies that had
    work to do. Waterfill assigns the shared expert to a rank and does not care
    how many replicas a routed expert has.
    """

    def test_lplb_declines_a_deployment_with_no_replicas(self):
        pipeline = RoutingPipeline.from_value("lplb")
        reason = pipeline.is_applicable(ExpertDeploymentConfig(num_redundant_experts=0))
        assert reason is not None and "redundant" in reason

    def test_lplb_applies_once_experts_are_replicated(self):
        pipeline = RoutingPipeline.from_value("lplb")
        assert pipeline.is_applicable(
            ExpertDeploymentConfig(num_redundant_experts=16)
        ) is None

    def test_waterfill_does_not_depend_on_routed_redundancy(self):
        pipeline = RoutingPipeline.from_value("waterfill")
        assert pipeline.is_applicable(
            ExpertDeploymentConfig(num_redundant_experts=0, routes_shared_expert=True)
        ) is None

    def test_waterfill_declines_a_runtime_that_cannot_dispatch_it(self):
        pipeline = RoutingPipeline.from_value("waterfill")
        reason = pipeline.is_applicable(
            ExpertDeploymentConfig(num_redundant_experts=32, routes_shared_expert=False)
        )
        assert reason is not None and "shared expert" in reason

    def test_the_reason_is_returned_not_a_bare_bool(self):
        """A framework that disables a policy has to be able to say why."""
        reason = RoutingPipeline.from_value("lplb").is_applicable(
            ExpertDeploymentConfig(num_redundant_experts=0)
        )
        assert isinstance(reason, str) and len(reason) > 20


@pytest.mark.unit
class TestCapabilitiesDeclareRuntimeRequirements:
    def test_only_the_lp_consumes_the_global_load(self):
        assert RoutingPipeline.from_value("lplb").capabilities \
            .consumes_global_logical_count is True
        for expr in ("static", "dynamic"):
            assert RoutingPipeline.from_value(expr).capabilities \
                .consumes_global_logical_count is False

    def test_concurrency_and_graph_stability_are_per_policy(self):
        lp = RoutingPipeline.from_value("lplb").capabilities
        assert lp.supports_concurrent_microbatches is False
        assert lp.graph_stability == "realloc_on_placement_change"

        st = RoutingPipeline.from_value("static").capabilities
        assert st.supports_concurrent_microbatches is True
        assert st.graph_stability == "stable"
