"""Physical-expert rank counting used between L2 routing policies."""

from __future__ import annotations

from typing import Optional

from moe_load_balancer.core.types import (
    PlacementSnapshot,
    RoutingPolicyConfig,
)


def count_routed_per_rank(
    physical_topk_ids: object,
    placement: Optional[PlacementSnapshot],
    policy: RoutingPolicyConfig,
    *,
    out: Optional[object] = None,
) -> object:
    """Count valid physical routed slots per expert-parallel rank."""

    import torch

    if not isinstance(physical_topk_ids, torch.Tensor):
        raise TypeError("Rank-load counting requires torch.Tensor physical_topk_ids.")

    rank_map = None
    if placement is not None:
        world_size = placement.ep_size
        experts_per_rank = placement.num_local_physical_experts
        if placement.physical_to_rank_map is not None:
            rank_map = placement.physical_to_rank_map
            if not isinstance(rank_map, torch.Tensor):
                rank_map = torch.as_tensor(rank_map, device=physical_topk_ids.device)
    else:
        try:
            world_size = int(policy.metadata["world_size"])
            experts_per_rank = int(policy.metadata["experts_per_rank"])
        except KeyError as exc:
            raise ValueError(
                "Rank-load counting without PlacementSnapshot requires "
                "policy.metadata['world_size'] and ['experts_per_rank']."
            ) from exc

    if out is None:
        counts = torch.zeros(
            world_size,
            dtype=torch.int64,
            device=physical_topk_ids.device,
        )
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError("Rank-load output buffer must be a torch.Tensor.")
        if out.shape != (world_size,) or out.dtype != torch.int64:
            raise ValueError("Rank-load output buffer has incompatible shape or dtype.")
        if out.device != physical_topk_ids.device:
            raise ValueError("Rank-load output buffer must be on the input device.")
        counts = out
        counts.zero_()

    if physical_topk_ids.is_cuda:
        from moe_load_balancer.kernels._rank_count_cuda import (
            count_routed_per_rank_cuda,
        )

        count_routed_per_rank_cuda(
            physical_topk_ids,
            counts,
            world_size=world_size,
            experts_per_rank=experts_per_rank,
            physical_to_rank_map=rank_map,
        )
        return counts

    # Keep the input shape fixed so this fallback also avoids dynamic indexing.
    valid = physical_topk_ids >= 0
    safe_ids = physical_topk_ids.clamp(min=0).to(torch.int64)
    ranks = rank_map[safe_ids] if rank_map is not None else safe_ids // experts_per_rank
    flat_ranks = torch.where(valid, ranks, 0).reshape(-1).to(torch.int64)
    contributions = valid.reshape(-1).to(torch.int64)
    return counts.scatter_add_(0, flat_ranks, contributions)
