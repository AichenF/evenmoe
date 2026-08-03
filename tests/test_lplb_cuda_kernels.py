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
