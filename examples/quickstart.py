"""Minimal executable L1 placement example used by README and CI."""

import torch

from moe_load_balancer import MoELoadBalancer
from moe_load_balancer.adapters.sglang import to_placement_request


def main() -> None:
    request = to_placement_request(
        torch.tensor([[20, 3, 9, 4], [5, 16, 2, 7]]),
        num_physical_experts=8,
        num_local_physical_experts=4,
        num_groups=1,
        num_nodes=1,
        algorithm="auto",
    )

    plan = MoELoadBalancer().plan_placement(request)

    print(plan.physical_to_logical_map.shape)
    print(plan.metadata["resolved_policy"])


if __name__ == "__main__":
    main()
