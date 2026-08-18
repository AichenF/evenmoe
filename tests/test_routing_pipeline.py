from __future__ import annotations

import pytest
import torch

from moe_load_balancer import (
    DeploymentTopology,
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
        reason = pipeline.is_applicable(DeploymentTopology(num_redundant_experts=0))
        assert reason is not None and "redundant" in reason

    def test_lplb_applies_once_experts_are_replicated(self):
        pipeline = RoutingPipeline.from_value("lplb")
        assert pipeline.is_applicable(
            DeploymentTopology(num_redundant_experts=16)
        ) is None

    def test_waterfill_does_not_depend_on_routed_redundancy(self):
        pipeline = RoutingPipeline.from_value("waterfill")
        assert pipeline.is_applicable(
            DeploymentTopology(num_redundant_experts=0, routes_shared_expert=True)
        ) is None

    def test_waterfill_declines_a_runtime_that_cannot_dispatch_it(self):
        pipeline = RoutingPipeline.from_value("waterfill")
        reason = pipeline.is_applicable(
            DeploymentTopology(num_redundant_experts=32, routes_shared_expert=False)
        )
        assert reason is not None and "shared expert" in reason

    def test_the_reason_is_returned_not_a_bare_bool(self):
        """A framework that disables a policy has to be able to say why."""
        reason = RoutingPipeline.from_value("lplb").is_applicable(
            DeploymentTopology(num_redundant_experts=0)
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

    def test_staleness_tolerance_is_stated(self):
        """The framework supplies a count one step old so the collective can
        live outside a captured graph; a policy that could not tolerate that
        has to be able to say so."""
        caps = RoutingPipeline.from_value("lplb").capabilities
        assert caps.max_input_staleness_steps == 1
        assert RoutingPipeline.from_value("static").capabilities \
            .max_input_staleness_steps == 0

    def test_concurrency_and_graph_stability_are_per_policy(self):
        lp = RoutingPipeline.from_value("lplb").capabilities
        assert lp.supports_concurrent_microbatches is False
        assert lp.graph_stability == "realloc_on_placement_change"
        assert lp.placement_state_lifecycle == "per_placement"

        st = RoutingPipeline.from_value("static").capabilities
        assert st.supports_concurrent_microbatches is True
        assert st.graph_stability == "stable"
        assert st.placement_state_lifecycle == "none"


@pytest.mark.unit
class TestWhoRealisesTheDecision:
    """Selection has to be separable, and both implementations have to remain.

    A framework that keeps selection fused inside its own kernel can only accept
    decisions shaped like whatever that kernel applies -- a per-expert
    distribution. An algorithm whose choice depends on the token, which is what
    a predictive layer needs, cannot be expressed that way at all. Declaring who
    realises the decision is what makes the step something a policy can own.
    """

    def test_the_replica_policies_realise_their_own_decision(self):
        for expr in ("lplb", "static", "dynamic"):
            assert RoutingPipeline.from_value(expr).capabilities.resolves_dispatch

    def test_declaring_it_replaces_selection_and_nothing_else(self):
        """The framework keeps recording either way; only the choice moves."""
        caps = RoutingPipeline.from_value("lplb").capabilities
        assert caps.resolves_dispatch is True
        # Nothing about recording, placement or graph safety changes with it.
        assert caps.requires_post_topk_routing is True
        assert caps.requires_placement_state is True

    def test_a_pipeline_with_no_post_topk_policy_leaves_selection_alone(self):
        """With no policy configured the framework's own selection stands."""
        caps = RoutingPipeline.from_value("none").capabilities
        assert caps.requires_post_topk_routing is False
        assert caps.resolves_dispatch is False
