import pytest
import torch

from moe_load_balancer import (
    PlacementPolicyConfig,
    PlacementRequest,
    PlacementStats,
    PlacementTopology,
)
from moe_load_balancer.policies.l3.ultraep import (
    UltraEPL3Policy,
    build_ultraep_initial_physical_to_logical_map,
)


class _Backend:
    def __init__(self):
        self.calls = []

    def solve(self, **kwargs):
        self.calls.append(kwargs)
        kwargs["physical_to_logical_map"].copy_(torch.tensor([0, 1, 0, 2, 3, 2], dtype=torch.int32))
        kwargs["logical_to_physical_map"].fill_(-1)
        kwargs["logical_to_physical_map"][:, 0].copy_(torch.tensor([0, 1, 3, 4], dtype=torch.int32))
        kwargs["logical_replica_counts"].fill_(1)
        kwargs["rank_quota_prefix"].zero_()


def _request():
    return PlacementRequest(
        stats=PlacementStats(
            logical_count=torch.tensor(
                [
                    [[4, 2, 6, 0], [2, 2, 0, 4]],
                    [[2, 0, 2, 0], [0, 2, 2, 0]],
                ],
                dtype=torch.float32,
            )
        ),
        topology=PlacementTopology(
            num_physical_experts=6,
            num_local_physical_experts=3,
            num_groups=1,
            num_nodes=1,
        ),
        policy=PlacementPolicyConfig(name="ultraep", metadata={"rank": 0, "num_nvl_ranks": 2}),
    )


@pytest.mark.unit
def test_ultraep_l3_uses_existing_placement_plan_contract():
    backend = _Backend()
    plan = UltraEPL3Policy(backend=backend).plan(_request())

    assert len(backend.calls) == 2
    assert backend.calls[0]["expert_loads"].tolist() == [6, 2, 8, 0]
    assert backend.calls[0]["expert_loads_per_rank"].tolist() == [
        [4, 2, 6, 0],
        [2, 0, 2, 0],
    ]
    assert backend.calls[0]["expert_loads"].dtype == torch.int32
    assert backend.calls[0]["expert_loads_per_rank"].dtype == torch.int32
    assert plan.physical_to_logical_map.shape == (2, 6)
    assert plan.logical_to_all_physical_map.shape == (2, 4, 2)
    assert plan.metadata["rank_quota_prefix"].shape == (2, 4, 2)


@pytest.mark.unit
def test_ultraep_initial_layout_keeps_master_slots_fixed():
    mapping = build_ultraep_initial_physical_to_logical_map(
        num_layers=2,
        num_logical_experts=8,
        ep_size=2,
        num_redundant_experts_per_rank=2,
    )

    assert mapping.dtype == torch.int32
    assert mapping.tolist() == [
        [0, 1, 2, 3, 0, 1, 4, 5, 6, 7, 4, 5],
        [0, 1, 2, 3, 0, 1, 4, 5, 6, 7, 4, 5],
    ]
