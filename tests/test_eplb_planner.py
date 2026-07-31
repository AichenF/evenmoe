from __future__ import annotations

import pytest
import torch

from moe_load_balancer import (
    MoELoadBalancer,
    PlacementPolicyConfig,
    PlacementRequest,
    PlacementStats,
)
from moe_load_balancer.adapters.sglang import to_placement_request
from moe_load_balancer.policies.l1.eplb.algorithms import (
    EplbAlgorithm,
    rebalance_experts,
)


@pytest.mark.unit
def test_orchestrator_eplb_matches_raw_deepseek_algorithm():
    logical_count = torch.tensor([[20, 3, 9, 4], [5, 16, 2, 7]])
    request = to_placement_request(
        logical_count,
        num_physical_experts=8,
        num_local_physical_experts=4,
        num_groups=1,
        num_nodes=1,
        algorithm="deepseek",
    )

    plan = MoELoadBalancer().plan_placement(request)
    expected = rebalance_experts(
        tokens_per_expert=logical_count.unsqueeze(0),
        num_physical_experts=8,
        num_local_physical_experts=4,
        num_groups=1,
        num_nodes=1,
        algorithm=EplbAlgorithm.deepseek,
    )

    assert torch.equal(plan.physical_to_logical_map, expected[0])
    assert torch.equal(plan.logical_to_all_physical_map, expected[1])
    assert torch.equal(plan.logical_to_physical_count, expected[2])
    assert not hasattr(plan, "snapshots")


@pytest.mark.unit
def test_cold_start_and_periodic_paths_use_same_request_and_policy_entry():
    logical_count = torch.tensor([[10, 7, 4, 1]])
    kwargs = dict(
        num_physical_experts=8,
        num_local_physical_experts=4,
        num_groups=1,
        num_nodes=1,
        algorithm="deepseek",
    )
    mlb = MoELoadBalancer()

    cold_start = mlb.plan_placement(to_placement_request(logical_count, **kwargs))
    periodic = mlb.plan_placement(to_placement_request(logical_count, **kwargs))

    assert torch.equal(cold_start.physical_to_logical_map, periodic.physical_to_logical_map)
    assert torch.equal(
        cold_start.logical_to_all_physical_map,
        periodic.logical_to_all_physical_map,
    )


@pytest.mark.unit
def test_framework_must_not_call_disabled_l1_policy():
    request = PlacementRequest(
        stats=PlacementStats(logical_count=torch.ones((1, 2))),
        policy=PlacementPolicyConfig(),
    )
    with pytest.raises(ValueError, match="framework must skip"):
        MoELoadBalancer().plan_placement(request)
