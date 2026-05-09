"""Shared-expert Waterfill L2 router.

The algorithm is a degenerate L2 case: it does not modify the routed top-k
(which already lives in physical-id space when the adapter calls in), and only
decides which EP rank receives each token's shared expert. Static and dynamic
modes are auto-distinguished by whether ``RoutingRequest.active_rank_token_count``
is provided:

- Static (no EP all-reduce): adapter passes ``routed_rank_load = local_routed
  counts`` and ``active_rank_token_count = None``. The Triton kernel derives
  ``target_total`` internally from ``rank_load.sum() / topk`` (correct because
  ``rank_load`` is pure routed counts in this case).

- Dynamic (post EP all-reduce): adapter passes ``routed_rank_load`` (post-AR
  global routed counts) and ``active_rank_token_count`` (post-AR per-rank
  active token counts). ``rank_load`` becomes ``routed + active`` (effective
  load), and the policy explicitly precomputes ``target_total`` because the
  kernel's naive derivation would be incorrect on the combined tensor.

The asymmetric handling of ``target_total`` (kernel-derive in static,
precompute in dynamic) is preserved from PR #19290 as a deliberate performance
optimization: it avoids an extra host-GPU sync per layer in the static path.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor

from moe_load_balancer.core.l2 import TokenRouter
from moe_load_balancer.core.types import RoutingDecision, RoutingRequest

# Local-rank preference fraction for Waterfill weighting (11/10 = 1.1x).
_LOCAL_PREF_NUMER = 11
_LOCAL_PREF_DENOM = 10


class WaterfillL2Router(TokenRouter):
    """Shared-expert Waterfill rank selection router.

    Implements ``core.TokenRouter``. ``route_tokens`` consumes a
    ``RoutingRequest`` whose canonical fields carry per-batch state:

    - ``request.routed_rank_load``: per-rank routed expert assignment count.
    - ``request.active_rank_token_count``: per-rank active token count
      (set only in dynamic mode; ``None`` signals static mode).
    - ``request.policy.metadata["experts_per_rank"]``: DeepEP-style layout
      assumption (each rank holds N contiguously-numbered physical experts).
      This is a stable per-policy config string field, not a tensor.

    The router does not consume ``request.placement`` (Waterfill operates on
    already-physical IDs handed by the adapter).
    """

    def __init__(
        self,
        *,
        source_rank: int,
        local_preference: float = 1.1,
        world_size: int,
    ):
        if world_size <= 0:
            raise ValueError("WaterfillL2Router.world_size must be positive.")
        if not (0 <= source_rank < world_size):
            raise ValueError(
                "WaterfillL2Router.source_rank must satisfy 0 <= source_rank < world_size."
            )
        if local_preference <= 0:
            raise ValueError("WaterfillL2Router.local_preference must be positive.")
        self.source_rank = source_rank
        self.local_preference = local_preference
        self.world_size = world_size

    def route_tokens(self, request: RoutingRequest) -> RoutingDecision:
        # Stage gating using the canonical core protocol path.
        if not request.policy.should_run_for_stage(request.stage):
            return RoutingDecision(
                routed_physical_topk_ids=request.logical_topk_ids,
                topk_weights=request.topk_weights,
                shared_expert_rank=None,
                metadata={"policy": request.policy.name, "skipped": True},
            )

        topk_ids = request.logical_topk_ids
        if not isinstance(topk_ids, Tensor):
            raise TypeError("WaterfillL2Router requires torch.Tensor topk_ids.")
        topk_size = topk_ids.shape[1]

        routed = request.routed_rank_load
        if routed is None or not isinstance(routed, Tensor):
            raise ValueError(
                "WaterfillL2Router requires routed_rank_load as a torch.Tensor."
            )

        active = request.active_rank_token_count
        if active is not None and not isinstance(active, Tensor):
            raise TypeError(
                "WaterfillL2Router.active_rank_token_count must be a torch.Tensor "
                "or None."
            )

        try:
            experts_per_rank = int(request.policy.metadata["experts_per_rank"])
        except KeyError as exc:
            raise ValueError(
                "WaterfillL2Router requires policy.metadata['experts_per_rank']."
            ) from exc

        if active is None:
            # Static path: rank_load = pure routed counts; let kernel derive
            # target_total via (sum + sum/topk + ws - 1) / ws (correct because
            # rank_load is pure A in this case).
            rank_load = routed
            target_total = 0
            allow_all_ranks = True
        else:
            # Dynamic path: rank_load = effective (A + B); precompute
            # target_total because kernel's naive derivation would be wrong
            # on the combined tensor.
            rank_load = routed + active
            total_routed = routed.sum()
            total_tokens = total_routed // topk_size
            total_effective = rank_load.sum()
            # target_total = perfect_balance + slack-for-shared-distribution.
            # The slack term (= total_tokens / ws) is necessary for the
            # algorithm to have enough capacity to distribute all shared
            # expert assignments.
            target_total = int(
                (total_effective + total_tokens + self.world_size - 1)
                // self.world_size
            )
            allow_all_ranks = bool(rank_load.max() <= target_total)

        chosen_rank = _waterfill_select(
            rank_load=rank_load,
            topk_ids=topk_ids,
            target_total=target_total,
            allow_all_ranks=allow_all_ranks,
            source_rank=self.source_rank,
            local_preference=self.local_preference,
            world_size=self.world_size,
            experts_per_rank=experts_per_rank,
        )

        return RoutingDecision(
            routed_physical_topk_ids=topk_ids,
            topk_weights=request.topk_weights,
            shared_expert_rank=chosen_rank,
            metadata={
                "policy": request.policy.name,
                "target_total": target_total,
                "allow_all_ranks": allow_all_ranks,
            },
        )


def _waterfill_select(
    *,
    rank_load: Tensor,
    topk_ids: Tensor,
    target_total: int,
    allow_all_ranks: bool,
    source_rank: int,
    local_preference: float,
    world_size: int,
    experts_per_rank: int,
) -> Tensor:
    """Run the Waterfill rank-selection Triton kernel.

    Returns ``chosen_rank: [num_tokens]`` int64 — the EP rank that should
    receive each token's shared expert.

    Internally this is the per-token decision step extracted from PR #19290's
    fused expand kernel; materialization back into SGLang's ``StandardTopKOutput``
    layout is the adapter's responsibility (in
    ``adapters/sglang/waterfill/runtime.py``).
    """

    num_tokens = topk_ids.shape[0]
    topk = topk_ids.shape[1]
    if num_tokens == 0:
        return torch.empty(0, dtype=torch.int64, device=topk_ids.device)

    chosen_rank = torch.empty(num_tokens, dtype=torch.int64, device=topk_ids.device)
    BLOCK_SIZE = 128
    grid = ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Convert local_preference to integer fraction for kernel-side math.
    pref_numer = max(int(round(local_preference * _LOCAL_PREF_DENOM)), 1)
    pref_denom = _LOCAL_PREF_DENOM

    _waterfill_select_kernel[grid](
        topk_ids,
        rank_load,
        chosen_rank,
        num_tokens,
        topk,
        experts_per_rank,
        world_size,
        source_rank,
        pref_numer,
        pref_denom,
        target_total,
        ALLOW_ALL_RANKS=allow_all_ranks,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return chosen_rank


@triton.jit
def _waterfill_select_kernel(
    topk_ids_ptr,           # [num_tokens, topk] int64 (physical expert ids)
    rank_load_ptr,          # [world_size] int64 (rank_load for kernel weighting)
    chosen_rank_ptr,        # [num_tokens] int64 (output: per-token chosen rank)
    num_tokens,
    topk: tl.constexpr,
    experts_per_rank,
    world_size: tl.constexpr,
    source_rank,
    local_pref_numer,
    local_pref_denom,
    precomputed_target_total,
    ALLOW_ALL_RANKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-token Waterfill rank selection.

    Two ``target_total`` paths:
      - ``precomputed_target_total > 0``: caller supplied a correct value
        (dynamic mode). Use it directly.
      - ``precomputed_target_total == 0``: caller signals "derive internally"
        (static mode, where rank_load is pure routed counts and the naive
        derivation gives the right answer).
    """

    pid = tl.program_id(0)
    token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_idx < num_tokens

    r_idx = tl.arange(0, world_size)
    rank_load_vec = tl.load(rank_load_ptr + r_idx, mask=r_idx < world_size, other=0).to(
        tl.int64
    )
    total_effective_k = tl.sum(rank_load_vec)
    total_tokens_global_k = total_effective_k // topk
    derived_target_total = (
        total_effective_k + total_tokens_global_k + world_size - 1
    ) // world_size
    target_total = tl.where(
        precomputed_target_total > 0,
        precomputed_target_total,
        derived_target_total,
    )

    # Locate the best candidate rank using local-rank preference.
    source_count = tl.load(rank_load_ptr + source_rank)
    best_count = tl.where(mask, source_count, 2**30)
    best_rank = tl.full([BLOCK_SIZE], source_rank, dtype=tl.int64)
    has_valid = tl.zeros([BLOCK_SIZE], dtype=tl.int1)
    src_rank_i32 = tl.full([BLOCK_SIZE], source_rank, dtype=tl.int32)

    if ALLOW_ALL_RANKS:
        candidate_mask = tl.full([BLOCK_SIZE], (1 << world_size) - 1, dtype=tl.int32)
        for r in range(world_size):
            target_count = tl.load(rank_load_ptr + r).to(tl.int64)
            better = (
                target_count * local_pref_numer < best_count * local_pref_denom
            ) & mask
            best_count = tl.where(better, target_count, best_count)
            best_rank = tl.where(
                better, tl.full([BLOCK_SIZE], r, dtype=tl.int64), best_rank
            )
    else:
        candidate_mask = (tl.full([BLOCK_SIZE], 1, dtype=tl.int32) << src_rank_i32).to(
            tl.int32
        )

    for k in range(topk):
        expert_id = tl.load(
            topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1
        ).to(tl.int64)
        valid = expert_id >= 0
        has_valid = has_valid | valid

        if not ALLOW_ALL_RANKS:
            target_rank = expert_id // experts_per_rank
            target_rank = tl.minimum(tl.maximum(target_rank, 0), world_size - 1)
            target_rank_i32 = target_rank.to(tl.int32)
            shift_amt = tl.where(valid, target_rank_i32, 0)
            bit = tl.full([BLOCK_SIZE], 1, dtype=tl.int32) << shift_amt
            candidate_mask = tl.where(
                valid & mask, candidate_mask | bit, candidate_mask
            )

            target_count = tl.load(
                rank_load_ptr + target_rank, mask=mask & valid, other=2**30
            )

            better = (
                (target_count * local_pref_numer < best_count * local_pref_denom)
                & valid
                & mask
            )
            best_count = tl.where(better, target_count, best_count)
            best_rank = tl.where(better, target_rank, best_rank)

    # Compute total weight across candidate ranks (slack-based, with local
    # preference applied to non-source ranks).
    total_w = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for r in range(world_size):
        present = ((candidate_mask >> r) & 1) == 1
        rank_load_r = tl.load(rank_load_ptr + r).to(tl.int64)
        w = tl.where(target_total > rank_load_r, target_total - rank_load_r, 0).to(
            tl.int32
        )
        w_vec = tl.full([BLOCK_SIZE], w, dtype=tl.int32)
        w_vec = tl.where(
            src_rank_i32 == r,
            w_vec,
            (w_vec * local_pref_denom) // local_pref_numer,
        )
        total_w += tl.where(present, w_vec, 0)

    # Deterministic per-token sampling using token_idx + source_rank seed.
    token_seed = token_idx.to(tl.uint32) ^ (
        src_rank_i32.to(tl.uint32) * tl.full([BLOCK_SIZE], 0x9E3779B9, dtype=tl.uint32)
    )
    token_seed = token_seed * tl.full([BLOCK_SIZE], 1664525, dtype=tl.uint32) + tl.full(
        [BLOCK_SIZE], 1013904223, dtype=tl.uint32
    )
    u = tl.where(total_w > 0, token_seed % total_w.to(tl.uint32), 0).to(tl.int32)

    chosen = src_rank_i32
    cum = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for r in range(world_size):
        present = ((candidate_mask >> r) & 1) == 1
        rank_load_r = tl.load(rank_load_ptr + r).to(tl.int64)
        w = tl.where(target_total > rank_load_r, target_total - rank_load_r, 0).to(
            tl.int32
        )
        w_vec = tl.full([BLOCK_SIZE], w, dtype=tl.int32)
        w_vec = tl.where(
            src_rank_i32 == r,
            w_vec,
            (w_vec * local_pref_denom) // local_pref_numer,
        )
        w_vec = tl.where(present, w_vec, 0)
        pick = (total_w > 0) & present & (u >= cum) & (u < (cum + w_vec))
        chosen = tl.where(pick, r, chosen)
        cum += w_vec

    final_rank = tl.where(total_w > 0, chosen.to(tl.int64), best_rank)
    # Tokens with no valid routed expert get -1 (caller materializer will
    # treat them as "drop"; SGLang's empty/padding path uses this signal).
    final_rank = tl.where(has_valid, final_rank, tl.full([BLOCK_SIZE], -1, dtype=tl.int64))

    tl.store(chosen_rank_ptr + token_idx, final_rank, mask=mask)
