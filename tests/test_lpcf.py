"""LP-CF: oracle property tests + Triton kernel parity."""

from __future__ import annotations

import numpy as np
import pytest

from moe_load_balancer.policies.l2.lpcf_reference import closed_form_pair_oracle


EP_SIZE = 4
NUM_PHYS = 16  # 4 per rank
NUM_LOCAL = NUM_PHYS // EP_SIZE
MAX_COPIES = 3


def _placement(rows):
    n = len(rows)
    m = np.full((n, MAX_COPIES), -1, dtype=np.int32)
    k = np.zeros(n, dtype=np.int32)
    for i, phys in enumerate(rows):
        k[i] = len(phys)
        for j, p in enumerate(phys):
            m[i, j] = p
    return m, k


def test_oracle_zero_input_returns_zeros():
    g = np.zeros(5, dtype=np.float32)
    m, k = _placement([(0,), (4,), (8, 12), (1,), (5,)])
    out, L = closed_form_pair_oracle(g, m, k, EP_SIZE, NUM_PHYS, MAX_COPIES)
    assert (out == 0).all()
    assert (L == 0).all()


def test_oracle_k1_writes_g_over_total():
    g = np.array([10, 20, 5], dtype=np.float32)
    m, k = _placement([(0,), (5,), (9,)])
    out, L = closed_form_pair_oracle(g, m, k, EP_SIZE, NUM_PHYS, MAX_COPIES)
    np.testing.assert_allclose(out[:, 0], g / g.sum())
    assert (out[:, 1:] == 0).all()
    np.testing.assert_allclose(L, [10, 20, 5, 0])


def test_oracle_k2_equalizes_against_k1_load():
    # K=1 loads r=0 with 5, r=1 with 15. K=2 logical G=10 should go all to r=0
    # because (15 - 5 + 10) / (2*10) clips to 1.0.
    g = np.array([5, 15, 10], dtype=np.float32)
    m, k = _placement([(0,), (5,), (1, 5)])
    out, _ = closed_form_pair_oracle(g, m, k, EP_SIZE, NUM_PHYS, MAX_COPIES)
    total = g.sum()
    np.testing.assert_allclose(out[2, 0], 10 / total)
    np.testing.assert_allclose(out[2, 1], 0.0, atol=1e-7)


def test_oracle_k2_same_rank_splits_50_50():
    g = np.array([10.0], dtype=np.float32)
    m, k = _placement([(0, 3)])  # both physical slots resolve to rank 0
    out, L = closed_form_pair_oracle(g, m, k, EP_SIZE, NUM_PHYS, MAX_COPIES)
    np.testing.assert_allclose(out[0, 0], 0.5)
    np.testing.assert_allclose(out[0, 1], 0.5)
    np.testing.assert_allclose(L, [10, 0, 0, 0])


def test_oracle_k3_uniform_split():
    g = np.array([12.0], dtype=np.float32)
    m, k = _placement([(0, 5, 10)])
    out, L = closed_form_pair_oracle(g, m, k, EP_SIZE, NUM_PHYS, MAX_COPIES)
    np.testing.assert_allclose(out[0, :3], [1/3, 1/3, 1/3], rtol=1e-6)
    np.testing.assert_allclose(L, [4, 4, 4, 0])


def test_oracle_row_sum_equals_g_over_total():
    rng = np.random.default_rng(0)
    n = 16
    g = rng.uniform(0, 30, n).astype(np.float32)
    k = rng.choice([1, 2, 3], n).astype(np.int32)
    m = np.full((n, MAX_COPIES), -1, dtype=np.int32)
    for i in range(n):
        for j in range(k[i]):
            m[i, j] = rng.integers(NUM_PHYS)
    out, _ = closed_form_pair_oracle(g, m, k, EP_SIZE, NUM_PHYS, MAX_COPIES)
    np.testing.assert_allclose(out.sum(axis=-1), g / g.sum(), atol=1e-6)


def test_oracle_global_mass_conservation():
    rng = np.random.default_rng(1)
    n = 32
    g = rng.uniform(0, 50, n).astype(np.float32)
    k = rng.choice([1, 2, 3], n).astype(np.int32)
    m = np.full((n, MAX_COPIES), -1, dtype=np.int32)
    for i in range(n):
        for j in range(k[i]):
            m[i, j] = rng.integers(NUM_PHYS)
    out, _ = closed_form_pair_oracle(g, m, k, EP_SIZE, NUM_PHYS, MAX_COPIES)
    np.testing.assert_allclose(out.sum(), 1.0, atol=1e-5)


_HAS_CUDA = False
try:
    import torch as _torch
    _HAS_CUDA = _torch.cuda.is_available()
except ImportError:
    pass


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + triton")
def test_triton_cf_kernel_matches_oracle():
    import torch
    from moe_load_balancer.policies.l2.lpcf import (
        _cf_pair_fused_kernel, _CF_BLOCK_DIM, _DEFAULT_K_BOUND,
    )

    rng = np.random.default_rng(42)
    n = 32
    g = rng.uniform(0, 30, n).astype(np.float32)
    k = rng.choice([1, 2, 3], n).astype(np.int32)
    m = np.full((n, MAX_COPIES), -1, dtype=np.int32)
    for i in range(n):
        for j in range(k[i]):
            m[i, j] = rng.integers(NUM_PHYS)

    oracle_out, _ = closed_form_pair_oracle(g, m, k, EP_SIZE, NUM_PHYS, MAX_COPIES)

    device = torch.device("cuda")
    g_t = torch.from_numpy(g).to(device)
    m_t = torch.from_numpy(m).to(device)
    k_t = torch.from_numpy(k).to(device)
    out_t = torch.zeros((n, MAX_COPIES), dtype=torch.float32, device=device)
    L_scratch = torch.zeros(EP_SIZE, dtype=torch.float32, device=device)
    num_local = NUM_PHYS // EP_SIZE
    _cf_pair_fused_kernel[(1,)](
        out_t.view(-1), L_scratch, g_t, m_t.view(-1), k_t,
        NUM_LOGICAL=n,
        MAX_COPIES=MAX_COPIES,
        NUM_LOCAL_PHYS=num_local,
        BLOCK_DIM=_CF_BLOCK_DIM,
        K_BOUND=min(MAX_COPIES, _DEFAULT_K_BOUND),
    )
    np.testing.assert_allclose(out_t.cpu().numpy(), oracle_out, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + triton")
def test_triton_dispatch_picks_only_valid_slot():
    """When all probability is on slot 0, dispatch should always return slot-0 physical."""
    import torch
    from moe_load_balancer.policies.l2.lpcf import (
        _dispatch_probability_kernel, _DISPATCH_BLOCK_DIM, _DEFAULT_K_BOUND,
    )

    NL, NP = 8, 16
    prob = torch.zeros((NL, NP), dtype=torch.float32, device="cuda")
    prob[:, 0] = 1.0
    log2phy_map = torch.full((NL, NP), -1, dtype=torch.int32, device="cuda")
    for i in range(NL):
        log2phy_map[i, 0] = i * 2  # primary physical id per logical

    n = 64
    topk_ids = torch.randint(0, NL, (n,), dtype=torch.int32, device="cuda")
    random_vals = torch.rand(n, dtype=torch.float32, device="cuda")
    out = torch.empty(n, dtype=torch.int32, device="cuda")
    k_bound = min(NP, _DEFAULT_K_BOUND)
    grid = ((n + _DISPATCH_BLOCK_DIM - 1) // _DISPATCH_BLOCK_DIM,)
    _dispatch_probability_kernel[grid](
        out, topk_ids, prob, log2phy_map, random_vals, n,
        NUM_LOGICAL=NL, MAX_COPIES=NP, K_BOUND=k_bound, BLOCK=_DISPATCH_BLOCK_DIM,
    )
    expected = topk_ids * 2
    assert torch.equal(out, expected)


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + triton")
def test_triton_dispatch_zero_row_falls_back_to_uniform():
    """Logical with row_sum=0 should sample uniformly over valid (map>=0) slots."""
    import torch
    from moe_load_balancer.policies.l2.lpcf import (
        _dispatch_probability_kernel, _DISPATCH_BLOCK_DIM, _DEFAULT_K_BOUND,
    )

    NL, NP = 4, 8
    prob = torch.zeros((NL, NP), dtype=torch.float32, device="cuda")
    log2phy_map = torch.full((NL, NP), -1, dtype=torch.int32, device="cuda")
    log2phy_map[0, 0] = 10
    log2phy_map[0, 1] = 20
    # logical 0 has 2 valid replicas, prob all zero -> uniform fallback.

    n = 2048
    topk_ids = torch.zeros(n, dtype=torch.int32, device="cuda")
    random_vals = torch.rand(n, dtype=torch.float32, device="cuda")
    out = torch.empty(n, dtype=torch.int32, device="cuda")
    k_bound = min(NP, _DEFAULT_K_BOUND)
    grid = ((n + _DISPATCH_BLOCK_DIM - 1) // _DISPATCH_BLOCK_DIM,)
    _dispatch_probability_kernel[grid](
        out, topk_ids, prob, log2phy_map, random_vals, n,
        NUM_LOGICAL=NL, MAX_COPIES=NP, K_BOUND=k_bound, BLOCK=_DISPATCH_BLOCK_DIM,
    )
    out_cpu = out.cpu().numpy()
    assert set(out_cpu.tolist()) == {10, 20}
    n10 = int((out_cpu == 10).sum())
    # Expect ~50/50 split; allow wide band given Bernoulli noise on n=2048.
    assert 800 < n10 < 1248


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + triton")
def test_triton_dispatch_preserves_padding_and_rejects_out_of_range_ids():
    """Padding and malformed logical IDs must never index the placement table."""
    import torch
    from moe_load_balancer.policies.l2.lpcf import (
        _dispatch_probability_kernel, _DISPATCH_BLOCK_DIM, _DEFAULT_K_BOUND,
    )

    NL, NP = 4, 8
    prob = torch.zeros((NL, NP), dtype=torch.float32, device="cuda")
    prob[:, 0] = 1.0
    log2phy_map = torch.full((NL, NP), -1, dtype=torch.int32, device="cuda")
    log2phy_map[:, 0] = torch.arange(NL, dtype=torch.int32, device="cuda")
    topk_ids = torch.tensor([-1, 0, 3, NL], dtype=torch.int32, device="cuda")
    random_vals = torch.rand(topk_ids.numel(), dtype=torch.float32, device="cuda")
    out = torch.empty_like(topk_ids)
    k_bound = min(NP, _DEFAULT_K_BOUND)
    grid = ((topk_ids.numel() + _DISPATCH_BLOCK_DIM - 1) // _DISPATCH_BLOCK_DIM,)

    _dispatch_probability_kernel[grid](
        out, topk_ids, prob, log2phy_map, random_vals, topk_ids.numel(),
        NUM_LOGICAL=NL, MAX_COPIES=NP, K_BOUND=k_bound, BLOCK=_DISPATCH_BLOCK_DIM,
    )

    assert torch.equal(
        out,
        torch.tensor([-1, 0, 3, -1], dtype=torch.int32, device="cuda"),
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="needs CUDA + triton")
def test_router_refreshes_cached_placement_after_in_place_eplb_update():
    """An in-place framework metadata update must invalidate the int32 cache."""
    import torch

    from moe_load_balancer.core import (
        PlacementSnapshot,
        RoutingPolicyConfig,
        RoutingRequest,
    )
    from moe_load_balancer.policies.l2.lpcf import LPCFL2Router

    candidates = torch.tensor([[0, 2], [1, 3]], dtype=torch.int64, device="cuda")
    copy_count = torch.tensor([2, 2], dtype=torch.int64, device="cuda")
    snapshot = PlacementSnapshot(
        layer_id=0,
        num_logical_experts=2,
        num_physical_experts=4,
        ep_size=2,
        num_local_physical_experts=2,
        physical_to_logical_map=torch.tensor(
            [0, 1, 0, 1], dtype=torch.int64, device="cuda"
        ),
        logical_to_physical_candidates=candidates,
        logical_to_physical_count=copy_count,
        default_physical_for_logical=torch.tensor(
            [0, 1], dtype=torch.int64, device="cuda"
        ),
    )
    request = RoutingRequest(
        layer_id=0,
        logical_topk_ids=torch.tensor([[0, 1]], dtype=torch.int64, device="cuda"),
        topk_weights=torch.ones((1, 2), dtype=torch.float32, device="cuda"),
        policies=(RoutingPolicyConfig(name="lplb"),),
        placement=snapshot,
        global_logical_count=torch.ones(2, dtype=torch.float32, device="cuda"),
    )
    router = LPCFL2Router(ep_size=2, num_physical_experts=4)
    policy = RoutingPolicyConfig(name="lplb")

    router.route_tokens(request, policy)
    first_cached = router._placement_cache[0][1]
    assert torch.equal(first_cached, candidates.to(torch.int32))

    candidates.copy_(
        torch.tensor([[2, 0], [3, 1]], dtype=torch.int64, device="cuda")
    )
    router.route_tokens(request, policy)
    second_cached = router._placement_cache[0][1]

    assert second_cached.data_ptr() != first_cached.data_ptr()
    assert torch.equal(second_cached, candidates.to(torch.int32))
