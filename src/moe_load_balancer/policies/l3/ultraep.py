"""UltraEP placement algorithm behind MLB's existing planner interface."""

from __future__ import annotations

import torch

from moe_load_balancer.core.types import (
    PlacementPlan,
    PlacementPolicyConfig,
    PlacementRequest,
)
from moe_load_balancer.kernels.ultraep import (
    UltraEPPlacementBackend,
    UltraEPPlacementTuning,
    load_placement_tuning_from_env,
)


def build_ultraep_initial_physical_to_logical_map(
    *,
    num_layers: int,
    num_logical_experts: int,
    ep_size: int,
    num_redundant_experts_per_rank: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build fixed rank-local master slots followed by replica slots."""

    if num_layers <= 0:
        raise ValueError("UltraEP requires at least one MoE layer.")
    if ep_size <= 1 or num_logical_experts % ep_size != 0:
        raise ValueError("UltraEP requires divisible experts and ep_size > 1.")
    if num_redundant_experts_per_rank <= 0:
        raise ValueError("UltraEP requires at least one redundant slot per rank.")

    local_masters = num_logical_experts // ep_size
    rank_maps = []
    for rank in range(ep_size):
        masters = torch.arange(
            rank * local_masters,
            (rank + 1) * local_masters,
            dtype=torch.int32,
            device=device,
        )
        scratch = masters[
            torch.arange(num_redundant_experts_per_rank, device=device) % local_masters
        ]
        rank_maps.append(torch.cat((masters, scratch)))
    return torch.cat(rank_maps).repeat(num_layers, 1)


class UltraEPL3Policy:
    """Compute replica placement from per-rank logical expert loads."""

    name = "ultraep"

    def __init__(
        self,
        *,
        backend: UltraEPPlacementBackend | None = None,
        tuning: UltraEPPlacementTuning | None = None,
    ) -> None:
        self._backend = backend or UltraEPPlacementBackend()
        self._tuning = tuning or load_placement_tuning_from_env()

    @staticmethod
    def build_initial_physical_to_logical_map(**kwargs) -> torch.Tensor:
        return build_ultraep_initial_physical_to_logical_map(**kwargs)

    def plan(self, request: PlacementRequest) -> PlacementPlan:
        policy = PlacementPolicyConfig.from_value(request.policy)
        if policy.name != self.name:
            raise ValueError(f"UltraEPL3Policy received policy {policy.name!r}.")

        logical_count = request.stats.logical_count
        if not isinstance(logical_count, torch.Tensor):
            logical_count = torch.as_tensor(logical_count)
        if logical_count.ndim == 2:
            logical_count = logical_count.unsqueeze(0)
        if logical_count.ndim != 3:
            raise ValueError("logical_count must have shape [samples, layers, experts].")

        topology = request.topology
        ep_size = topology.num_physical_experts // topology.num_local_physical_experts
        num_layers, num_logical_experts = logical_count.shape[1:]
        if ep_size <= 1 or num_logical_experts % ep_size != 0:
            raise ValueError("UltraEP requires divisible experts and ep_size > 1.")
        num_local_master_experts = num_logical_experts // ep_size
        num_local_redundant_experts = topology.num_local_physical_experts - num_local_master_experts
        if num_local_redundant_experts <= 0:
            raise ValueError("UltraEP requires rank-local redundant expert slots.")

        rank = int(policy.metadata.get("rank", 0))
        num_nvl_ranks = int(policy.metadata.get("num_nvl_ranks", ep_size))
        per_rank_loads = logical_count.to(torch.int32).contiguous()
        global_loads = per_rank_loads.sum(dim=0, dtype=torch.int32).contiguous()
        max_replicas = ep_size
        # The initial value here does not matter: the kernel memsets all four
        # of these itself on every call (physical_to_logical_map and
        # logical_to_physical_map to -1, the rest to 0) before it solves.
        # zeros rather than empty only avoids handing the kernel whatever
        # garbage this process's allocator last left behind, on the off
        # chance a future kernel revision reads before it solves.
        physical_to_logical = torch.zeros(
            (num_layers, topology.num_physical_experts),
            dtype=torch.int32,
            device=global_loads.device,
        )
        logical_to_physical = torch.zeros(
            (num_layers, num_logical_experts, max_replicas),
            dtype=torch.int32,
            device=global_loads.device,
        )
        replica_counts = torch.zeros(
            (num_layers, num_logical_experts),
            dtype=torch.int32,
            device=global_loads.device,
        )
        rank_quota_prefix = torch.zeros_like(logical_to_physical)

        for layer_id in range(num_layers):
            # Non-contiguous whenever num_layers > 1: this slices the middle
            # dimension of a [ep_size, num_layers, num_logical_experts]
            # tensor, and the kernel reads it by raw pointer.
            per_rank = per_rank_loads[:, layer_id].contiguous()
            if per_rank.shape[0] != ep_size:
                raise ValueError("UltraEP placement requires one logical-load sample per EP rank.")
            logical_quota = torch.empty_like(logical_to_physical[layer_id])
            logical_quota_prefix = torch.empty_like(logical_quota)
            self._backend.solve(
                expert_loads=global_loads[layer_id],
                expert_loads_per_rank=per_rank,
                physical_to_logical_map=physical_to_logical[layer_id],
                logical_to_physical_map=logical_to_physical[layer_id],
                logical_replica_counts=replica_counts[layer_id],
                logical_instance_quota=logical_quota,
                logical_instance_quota_prefix=logical_quota_prefix,
                rank_quota_prefix=rank_quota_prefix[layer_id],
                num_ranks=ep_size,
                num_local_master_experts=num_local_master_experts,
                num_local_redundant_experts=num_local_redundant_experts,
                num_nvl_ranks=num_nvl_ranks,
                rank=rank,
                tuning=self._tuning,
            )

        # The kernel memsets physical_to_logical_map to -1 before it solves,
        # then only overwrites the slots it actually assigns a replica to: a
        # redundant slot whose would-be replica falls below the tuning's
        # token floor is deliberately left at -1, UltraEP's "not worth
        # materializing" signal. Every physical slot in a PlacementPlan is
        # otherwise guaranteed to name a real logical expert -- the
        # framework's load-history scatter and weight table both index by it
        # unconditionally -- so a slot the solve declined is filled with this
        # rank's own master experts, round-robin, the same fallback the
        # initial bootstrap layout already uses for a redundant slot with no
        # information yet. It is a harmless duplicate, never a wrong answer:
        # nothing routes tokens there unless logical_to_all_physical_map also
        # lists it as a candidate, which this fallback deliberately leaves
        # untouched -- the solve's decision not to route here stands.
        unassigned = physical_to_logical < 0
        if unassigned.any():
            physical_ids = torch.arange(
                topology.num_physical_experts, device=physical_to_logical.device
            )
            local_idx = physical_ids % topology.num_local_physical_experts
            rank_of_slot = physical_ids // topology.num_local_physical_experts
            fallback = (
                rank_of_slot * num_local_master_experts
                + local_idx % num_local_master_experts
            ).to(torch.int32)
            physical_to_logical = torch.where(unassigned, fallback, physical_to_logical)

        return PlacementPlan(
            physical_to_logical_map=physical_to_logical,
            logical_to_all_physical_map=logical_to_physical,
            logical_to_physical_count=replica_counts,
            metadata={
                "requested_policy": self.name,
                "resolved_policy": self.name,
                "rank_quota_prefix": rank_quota_prefix,
            },
        )
