"""SGLang-specific materialization for a Waterfill routing decision."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def materialize_waterfill_output(
    physical_ids: Tensor,
    weights: Tensor,
    chosen_rank: Tensor,
    *,
    num_routed_experts: int,
    world_size: int,
    routed_scaling_factor: float,
) -> Tuple[Tensor, Tensor]:
    """Convert MLB physical routing into SGLang's fused DeepEP layout.

    CUDA uses one fused kernel.  The CPU path is a small reference used by
    adapter tests and environments without Triton-capable devices.
    """

    if num_routed_experts <= 0 or world_size <= 0:
        raise ValueError("num_routed_experts and world_size must be positive.")
    if num_routed_experts % world_size != 0:
        raise ValueError("num_routed_experts must be divisible by world_size.")
    if physical_ids.ndim != 2 or weights.shape != physical_ids.shape:
        raise ValueError("physical_ids and weights must have the same 2-D shape.")
    if chosen_rank.shape != (physical_ids.shape[0],):
        raise ValueError("chosen_rank must have one entry per token.")

    old_experts_per_rank = num_routed_experts // world_size
    new_experts_per_rank = old_experts_per_rank + 1
    shared_weight = 1.0 / routed_scaling_factor if routed_scaling_factor != 0 else 1.0

    if not physical_ids.is_cuda:
        valid = physical_ids >= 0
        interleaved = torch.where(
            valid,
            physical_ids + physical_ids // old_experts_per_rank,
            physical_ids,
        )
        routed_weights = torch.where(valid, weights, torch.zeros_like(weights))
        has_valid = valid.any(dim=1)
        shared_ids = chosen_rank.to(physical_ids.dtype) * new_experts_per_rank
        shared_ids = shared_ids + old_experts_per_rank
        shared_ids = torch.where(
            has_valid & (chosen_rank >= 0),
            shared_ids,
            torch.full_like(shared_ids, -1),
        )
        shared_weights = torch.where(
            has_valid,
            torch.full_like(has_valid, shared_weight, dtype=weights.dtype),
            torch.zeros_like(has_valid, dtype=weights.dtype),
        )
        return (
            torch.cat((interleaved, shared_ids[:, None]), dim=1),
            torch.cat((routed_weights, shared_weights[:, None]), dim=1),
        )

    num_tokens, topk = physical_ids.shape
    expanded_ids = torch.empty(
        (num_tokens, topk + 1), dtype=physical_ids.dtype, device=physical_ids.device
    )
    expanded_weights = torch.empty(
        (num_tokens, topk + 1), dtype=weights.dtype, device=weights.device
    )
    if num_tokens == 0:
        return expanded_ids, expanded_weights

    from moe_load_balancer.adapters.sglang._waterfill_cuda import (
        materialize_waterfill_cuda,
    )

    materialize_waterfill_cuda(
        physical_ids,
        weights,
        chosen_rank,
        expanded_ids,
        expanded_weights,
        old_experts_per_rank,
        new_experts_per_rank,
        shared_weight,
    )
    return expanded_ids, expanded_weights
