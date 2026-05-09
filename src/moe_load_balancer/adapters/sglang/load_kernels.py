"""SGLang/DeepEP-specific GPU helpers for L2 load extraction.

These kernels assume DeepEP-style **contiguous expert-to-rank layout**:

    rank = expert_id // experts_per_rank

i.e. the first ``experts_per_rank`` physical expert IDs live on rank 0, the
next batch on rank 1, etc. This is the SGLang / DeepEP / vLLM / TRT-LLM
convention as of POC Stage 3 (see ``docs/sglang-integration.md``). Adapters
for frameworks using a different layout must provide their own load
extraction kernels.

These helpers are L2-generic — Waterfill consumes them today, LPLB / Metro
will consume them tomorrow. They live at the top of ``adapters/sglang/`` so
the L2-algorithm-specific subpackages (``waterfill/``, future ``lplb/``)
import from a shared location.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _count_routed_per_rank_kernel(
    topk_ids_ptr,  # [num_tokens, topk]
    counts_ptr,  # [world_size] int64 output (atomic add)
    num_tokens,
    topk: tl.constexpr,
    experts_per_rank,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Count routed expert assignments per destination rank.

    Each token's ``topk`` physical expert IDs are mapped to a destination rank
    using the contiguous-layout assumption. ``counts[r]`` ends up holding the
    total number of routed slots landing on rank ``r`` from this batch's
    top-k tensor.
    """
    pid = tl.program_id(0)
    token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_idx < num_tokens

    for r in range(world_size):
        rank_count = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

        for k in range(topk):
            expert_id = tl.load(
                topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1
            ).to(tl.int64)
            valid = expert_id >= 0
            target_rank = expert_id // experts_per_rank
            target_rank = tl.minimum(tl.maximum(target_rank, 0), world_size - 1)
            rank_count += tl.where(
                mask & valid & (target_rank == r),
                tl.full([BLOCK_SIZE], 1, dtype=tl.int64),
                tl.zeros([BLOCK_SIZE], dtype=tl.int64),
            )

        block_total = tl.sum(rank_count)
        if block_total > 0:
            tl.atomic_add(counts_ptr + r, block_total)


def count_local_routed_per_rank(
    topk_ids: Tensor,
    *,
    world_size: int,
    experts_per_rank: int,
    out: Tensor | None = None,
) -> Tensor:
    """Compute per-destination-rank routed expert assignment counts.

    Returns a ``[world_size]`` ``int64`` tensor where entry ``r`` is the count
    of routed slots in ``topk_ids`` whose destination is rank ``r``.

    For zero-token batches, returns a zero buffer (no kernel launched).
    Caller may pre-allocate ``out`` to recycle a buffer across calls; if
    omitted, a fresh tensor is allocated.

    The layout assumption is documented at the module level. Pass a non-empty
    ``[num_tokens, topk]`` ``int64`` tensor for ``topk_ids`` (negative entries
    are treated as "no expert").
    """
    if out is None:
        out = torch.zeros(world_size, dtype=torch.int64, device=topk_ids.device)
    else:
        out.zero_()

    num_tokens = topk_ids.shape[0]
    if num_tokens == 0:
        return out

    topk = topk_ids.shape[1]
    BLOCK_SIZE = 256
    grid = ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _count_routed_per_rank_kernel[grid](
        topk_ids,
        out,
        num_tokens,
        topk,
        experts_per_rank,
        world_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
