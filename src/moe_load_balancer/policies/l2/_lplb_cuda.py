"""CUDA-only LPLB probability dispatch, imported lazily by the policy."""

from __future__ import annotations

import triton
import triton.language as tl
from torch import Tensor


_DISPATCH_BLOCK_DIM = 256
_DEFAULT_K_BOUND = 32


@triton.jit
def _prep_counts_kernel(
    a_full_ptr,
    b_ptr,
    t1_ptr,
    counts_ptr,
    log_single_ptr,
    log_replicated_ptr,
    row_sum_ptr,
    NUM_LOGICAL: tl.constexpr,
    NUM_SINGLE: tl.constexpr,
    NUM_REPLICATED_LOGICAL: tl.constexpr,
    NC: tl.constexpr,
    NV: tl.constexpr,
    COUNT_BLOCK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    count_offsets = tl.arange(0, COUNT_BLOCK)
    count_mask = count_offsets < NUM_LOGICAL
    counts = tl.load(counts_ptr + count_offsets, mask=count_mask, other=0.0)
    total = tl.maximum(tl.sum(counts, axis=0), 1.0)

    row_offsets = tl.arange(0, ROW_BLOCK)
    single_mask = row_offsets < NUM_SINGLE
    single_logical = tl.load(
        log_single_ptr + row_offsets, mask=single_mask, other=0
    )
    single_counts = tl.load(
        counts_ptr + single_logical, mask=single_mask, other=0.0
    ) / total
    tl.store(t1_ptr + row_offsets, single_counts, mask=single_mask)

    replicated_mask = row_offsets < NUM_REPLICATED_LOGICAL
    replicated_logical = tl.load(
        log_replicated_ptr + row_offsets, mask=replicated_mask, other=0
    )
    replicated_counts = tl.load(
        counts_ptr + replicated_logical, mask=replicated_mask, other=0.0
    ) / total
    tl.store(b_ptr + row_offsets, replicated_counts, mask=replicated_mask)
    base_sum = tl.load(row_sum_ptr + row_offsets, mask=replicated_mask, other=0.0)
    tl.store(
        a_full_ptr + row_offsets * NV + (NV - 1),
        replicated_counts - base_sum,
        mask=replicated_mask,
    )


@triton.jit
def _prep_rank_kernel(
    a_full_ptr,
    b_ptr,
    t1_ptr,
    b1_ptr,
    row_sum_ptr,
    NUM_SINGLE: tl.constexpr,
    NUM_REPLICATED_LOGICAL: tl.constexpr,
    NUM_RANKS: tl.constexpr,
    NV: tl.constexpr,
    SINGLE_BLOCK: tl.constexpr,
):
    rank = tl.program_id(0)
    single_offsets = tl.arange(0, SINGLE_BLOCK)
    mask = single_offsets < NUM_SINGLE
    fixed_load = tl.sum(
        tl.load(
            b1_ptr + rank * NUM_SINGLE + single_offsets,
            mask=mask,
            other=0.0,
        )
        * tl.load(t1_ptr + single_offsets, mask=mask, other=0.0),
        axis=0,
    )
    row = NUM_REPLICATED_LOGICAL + rank
    value = -fixed_load
    tl.store(b_ptr + row, value)
    base_sum = tl.load(row_sum_ptr + row)
    tl.store(a_full_ptr + row * NV + (NV - 1), value - base_sum)


@triton.jit
def _extract_probability_kernel(
    probability_ptr,
    x_ptr,
    t1_ptr,
    source_ptr,
    n,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    source = tl.load(source_ptr + offsets, mask=mask, other=-1)
    from_x = tl.load(x_ptr + tl.maximum(source, 0), mask=mask & (source >= 0), other=0.0)
    t1_index = -source - 2
    from_t1 = tl.load(
        t1_ptr + tl.maximum(t1_index, 0),
        mask=mask & (source <= -2),
        other=0.0,
    )
    value = tl.where(source >= 0, tl.maximum(from_x, 0.0), from_t1)
    value = tl.where(source == -1, 0.0, value)
    tl.store(probability_ptr + offsets, value, mask=mask)


def prep_lp_inputs_cuda(
    a_full: Tensor,
    b: Tensor,
    t1: Tensor,
    global_counts: Tensor,
    log_single: Tensor,
    log_replicated: Tensor,
    b1: Tensor,
    a_base_row_sum: Tensor,
) -> None:
    nc, nv = a_full.shape
    num_logical = global_counts.numel()
    num_single = log_single.numel()
    num_replicated_logical = log_replicated.numel()
    num_ranks = b1.shape[0]
    count_block = triton.next_power_of_2(num_logical)
    row_block = triton.next_power_of_2(max(num_single, num_replicated_logical, 1))
    _prep_counts_kernel[(1,)](
        a_full,
        b,
        t1,
        global_counts,
        log_single,
        log_replicated,
        a_base_row_sum,
        NUM_LOGICAL=num_logical,
        NUM_SINGLE=num_single,
        NUM_REPLICATED_LOGICAL=num_replicated_logical,
        NC=nc,
        NV=nv,
        COUNT_BLOCK=count_block,
        ROW_BLOCK=row_block,
    )
    _prep_rank_kernel[(num_ranks,)](
        a_full,
        b,
        t1,
        b1,
        a_base_row_sum,
        NUM_SINGLE=num_single,
        NUM_REPLICATED_LOGICAL=num_replicated_logical,
        NUM_RANKS=num_ranks,
        NV=nv,
        SINGLE_BLOCK=triton.next_power_of_2(max(num_single, 1)),
    )


def extract_probability_cuda(
    probability: Tensor,
    x: Tensor,
    t1: Tensor,
    probability_source: Tensor,
) -> None:
    n = probability.numel()
    _extract_probability_kernel[(triton.cdiv(n, _DISPATCH_BLOCK_DIM),)](
        probability,
        x,
        t1,
        probability_source,
        n,
        BLOCK=_DISPATCH_BLOCK_DIM,
    )


@triton.jit
def _dispatch_probability_kernel(
    output_ptr,
    logical_ids_ptr,
    probability_ptr,
    candidate_ptr,
    random_ptr,
    n,
    NUM_LOGICAL: tl.constexpr,
    MAX_COPIES: tl.constexpr,
    K_BOUND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offsets < n
    logical = tl.load(logical_ids_ptr + offsets, mask=in_bounds, other=0)
    valid = in_bounds & (logical >= 0) & (logical < NUM_LOGICAL)
    columns = tl.arange(0, K_BOUND)
    column_mask = columns < MAX_COPIES
    table_offsets = logical[:, None] * MAX_COPIES + columns[None, :]
    probabilities = tl.load(
        probability_ptr + table_offsets,
        mask=valid[:, None] & column_mask[None, :],
        other=0.0,
    )
    candidates = tl.load(
        candidate_ptr + table_offsets,
        mask=valid[:, None] & column_mask[None, :],
        other=-1,
    )
    row_sum = tl.sum(probabilities, axis=1)
    fallback = tl.where(candidates >= 0, 1.0, 0.0)
    use_fallback = row_sum <= 0.0
    probabilities = tl.where(use_fallback[:, None], fallback, probabilities)
    row_sum = tl.where(use_fallback, tl.sum(fallback, axis=1), row_sum)
    sample = tl.load(random_ptr + offsets, mask=in_bounds, other=0.0) * row_sum
    cumulative = tl.cumsum(probabilities, axis=1)
    chosen = tl.sum((cumulative <= sample[:, None]).to(tl.int32), axis=1)
    chosen = tl.minimum(chosen, MAX_COPIES - 1)
    selected = tl.sum(
        tl.where(columns[None, :] == chosen[:, None], candidates, 0), axis=1
    )
    tl.store(output_ptr + offsets, tl.where(valid, selected, -1), mask=in_bounds)


def dispatch_probability_cuda(
    output: Tensor,
    logical_ids: Tensor,
    probability: Tensor,
    candidates: Tensor,
    random_values: Tensor,
) -> None:
    n = logical_ids.numel()
    num_logical, max_copies = probability.shape
    if max_copies > _DEFAULT_K_BOUND:
        raise ValueError(
            f"LPLB supports at most {_DEFAULT_K_BOUND} physical copies per logical "
            f"expert; got {max_copies}."
        )
    # tl.arange requires a power-of-two extent. Keep MAX_COPIES as the real
    # row width and mask the padded columns inside the kernel.
    k_bound = triton.next_power_of_2(max_copies)
    _dispatch_probability_kernel[(triton.cdiv(n, _DISPATCH_BLOCK_DIM),)](
        output,
        logical_ids,
        probability,
        candidates,
        random_values,
        n,
        NUM_LOGICAL=num_logical,
        MAX_COPIES=max_copies,
        K_BOUND=k_bound,
        BLOCK=_DISPATCH_BLOCK_DIM,
    )
