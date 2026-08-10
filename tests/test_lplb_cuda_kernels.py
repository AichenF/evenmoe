"""Focused GPU regression tests for the vendored LPLB CUDA kernels."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_dispatch_probability_preserves_padded_and_out_of_range_ids():
    """Invalid TopK slots must not index before or after the dispatch tables."""

    from moe_load_balancer.kernels.lplb.cuda_solver import dispatch_probability

    topk_ids = torch.tensor([[0, -1, 2, 3]], dtype=torch.int32, device="cuda")
    probabilities = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.25, 0.75]],
        dtype=torch.float32,
        device="cuda",
    )
    candidates = torch.tensor(
        [[10, -1], [11, 21], [12, 22]],
        dtype=torch.int32,
        device="cuda",
    )

    routed = dispatch_probability(topk_ids, probabilities, candidates)
    torch.cuda.synchronize()

    assert routed[0, 0].item() == 10
    assert routed[0, 1].item() == -1
    assert routed[0, 2].item() in (12, 22)
    assert routed[0, 3].item() == -1


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_lp_prep_accepts_an_all_replicated_placement():
    """A valid placement may have no single-copy logical experts."""

    from moe_load_balancer.core.types import PlacementSnapshot
    from moe_load_balancer.kernels.lplb.cuda_solver import prep_lp_inputs
    from moe_load_balancer.policies.l2.lplb import _build_layer_state

    device = torch.device("cuda")
    snapshot = PlacementSnapshot(
        layer_id=0,
        num_logical_experts=2,
        num_physical_experts=4,
        ep_size=2,
        num_local_physical_experts=2,
        physical_to_logical_map=torch.tensor(
            [0, 1, 0, 1], dtype=torch.int64, device=device
        ),
        logical_to_physical_candidates=torch.tensor(
            [[0, 2], [1, 3]], dtype=torch.int64, device=device
        ),
        logical_to_physical_count=torch.tensor(
            [2, 2], dtype=torch.int64, device=device
        ),
    )
    state = _build_layer_state(snapshot, num_gpus=2, big_m_penalty=1000.0)
    global_counts = torch.tensor([12.0, 4.0], device=device)

    prep_lp_inputs(
        state.A_full,
        state.b,
        state.t1,
        global_counts,
        state.log_single,
        state.log_replicated,
        state.B1,
        state.A_base_row_sum,
    )
    torch.cuda.synchronize()

    counts_norm = global_counts / global_counts.sum().clamp(min=1.0)
    expected_t1 = counts_norm[state.log_single]
    expected_b = torch.cat(
        (
            counts_norm[state.log_replicated],
            -(state.B1 @ expected_t1).flatten(),
        )
    )
    torch.testing.assert_close(state.t1, expected_t1)
    torch.testing.assert_close(state.b, expected_b)
    torch.testing.assert_close(
        state.A_full[:, -1], expected_b - state.A_base_row_sum
    )
