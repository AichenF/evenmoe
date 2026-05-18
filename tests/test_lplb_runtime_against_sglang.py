# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Integration tests for :class:`LPLBRuntime` against a live SGLang install.

These tests require:

- SGLang installed in the active environment (any baseline — LPLB kernels
  are vendored inside the SDK as of Phase E; no PR sgl-project/sglang#24515
  fork required).
- A CUDA device (Hopper-class for the cuBLASDx IPM kernel).
- ``tvm-ffi`` and Math-DX (cuBLASDx) for the JIT build path. Both are
  available in the lmsysorg/sglang:dev container by default.

Tests are marked with ``pytest.mark.integration`` (declared in
``pyproject.toml``); a vanilla ``pytest tests/`` run on CPU still passes
because both the SDK's vendored cuda_solver module and CUDA are checked at
module-collection time and the whole module is skipped when either is
missing.
"""

from __future__ import annotations

import pytest
import torch

# Skip the whole module if the SDK's vendored LPLB CUDA pipeline can't be
# imported (it requires torch + tvm-ffi at module import time). The pure
# framework-neutral tests in tests/test_lplb_policy.py cover the policy
# logic without these dependencies.
pytest.importorskip(
    "moe_load_balancer.adapters.sglang.lplb.cuda_solver",
    reason=(
        "LPLB CUDA kernel pipeline requires torch + tvm-ffi importable. "
        "Run inside the lmsysorg/sglang:dev container."
    ),
)
pytest.importorskip(
    "sglang.srt.eplb.expert_location",
    reason="SGLang ExpertLocationMetadata is required for the runtime tests.",
)

if not torch.cuda.is_available():  # pragma: no cover — CPU-only envs
    pytest.skip(
        "LPLBRuntime tests require a CUDA device (Hopper-class for IPM kernel).",
        allow_module_level=True,
    )

# Imports below are deferred until after the skip guards.
from sglang.srt.eplb.expert_location import ExpertLocationMetadata  # noqa: E402
from sglang.srt.layers.moe.topk import StandardTopKOutput  # noqa: E402

from moe_load_balancer.adapters.sglang import SGLangPlacementAdapter  # noqa: E402
from moe_load_balancer.adapters.sglang.lplb import (  # noqa: E402
    LPLBRuntime,
    SGLangLPLBKernels,
)
from moe_load_balancer.core.l2 import LPLBKernels, TokenRouter  # noqa: E402


pytestmark = pytest.mark.integration


# -- Fixtures -------------------------------------------------------------


def _build_metadata(num_layers: int = 2):
    """Build a CUDA-resident ExpertLocationMetadata for a 4-logical / 6-physical
    layout shared across layers.

    Layout per layer:
      logical 0 (replicated, 2 copies): phys 0 (GPU 0), phys 3 (GPU 1)
      logical 1 (single):              phys 1 (GPU 0)
      logical 2 (replicated, 2 copies): phys 2 (GPU 0), phys 4 (GPU 1)
      logical 3 (single):              phys 5 (GPU 1)
    """
    num_logical, num_phy = 4, 6
    layout = [(0, 0), (1, 1), (2, 2), (0, 3), (2, 4), (3, 5)]

    log2phy = torch.full((num_layers, num_logical, num_phy), -1, dtype=torch.int64)
    log_counts = torch.zeros((num_layers, num_logical), dtype=torch.int64)
    phy2log = torch.zeros((num_layers, num_phy), dtype=torch.int64)
    for L in range(num_layers):
        for lid, p in layout:
            slot = log_counts[L, lid].item()
            log2phy[L, lid, slot] = p
            log_counts[L, lid] += 1
            phy2log[L, p] = lid

    return ExpertLocationMetadata(
        physical_to_logical_map=phy2log.cuda(),
        physical_to_logical_map_cpu=phy2log.clone(),
        logical_to_all_physical_map=log2phy.cuda(),
        logical_to_all_physical_map_cpu=log2phy.clone(),
        logical_to_all_physical_map_num_valid=log_counts.cuda(),
        logical_to_rank_dispatch_physical_map=None,
    )


def _build_runtime(*, layer_id: int = 0, ep_group=None) -> LPLBRuntime:
    meta = _build_metadata()
    adapter = SGLangPlacementAdapter(metadata_source=meta)
    return LPLBRuntime(
        layer_id=layer_id,
        ep_group=ep_group,
        num_logical_experts=4,
        num_physical_experts=6,
        num_gpus=2,
        placement_provider=adapter,
    )


def _topk_output(topk_ids: torch.Tensor, topk_weights: torch.Tensor):
    router_logits = torch.zeros(
        (topk_ids.shape[0], topk_ids.shape[1]),
        dtype=torch.float32,
        device=topk_ids.device,
    )
    return StandardTopKOutput(topk_weights, topk_ids, router_logits)


class _RecordingEpGroup:
    """Captures all_reduce calls and forwards the tensor unchanged.

    Models a single-rank EP group: the "global" counts equal the local
    counts. Tracks call count and the shapes/dtypes seen so tests can
    assert the collective participates correctly on empty-batch paths.
    """

    def __init__(self):
        self.all_reduce_calls: list[tuple] = []

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self.all_reduce_calls.append((tuple(tensor.shape), str(tensor.dtype)))
        return tensor


# -- Construction ---------------------------------------------------------


def test_runtime_construct_with_real_sglang_metadata():
    """The runtime should construct cleanly against a real
    :class:`ExpertLocationMetadata` and a :class:`SGLangPlacementAdapter`."""
    runtime = _build_runtime()
    assert runtime.layer_id == 0
    assert runtime.num_logical_experts == 4
    assert runtime.num_physical_experts == 6
    assert runtime.num_gpus == 2
    # The runtime owns concrete adapter instances of the right protocols.
    assert isinstance(runtime.kernels, SGLangLPLBKernels)
    assert isinstance(runtime.kernels, LPLBKernels)
    assert isinstance(runtime.router, TokenRouter)


def test_runtime_rejects_invalid_constructor_args():
    with pytest.raises(ValueError, match="num_logical_experts"):
        LPLBRuntime(
            layer_id=0,
            ep_group=None,
            num_logical_experts=0,
            num_physical_experts=6,
            num_gpus=2,
            placement_provider=SGLangPlacementAdapter(_build_metadata()),
        )
    with pytest.raises(ValueError, match="num_physical_experts"):
        LPLBRuntime(
            layer_id=0,
            ep_group=None,
            num_logical_experts=4,
            num_physical_experts=0,
            num_gpus=2,
            placement_provider=SGLangPlacementAdapter(_build_metadata()),
        )
    with pytest.raises(ValueError, match="num_gpus"):
        LPLBRuntime(
            layer_id=0,
            ep_group=None,
            num_logical_experts=4,
            num_physical_experts=6,
            num_gpus=0,
            placement_provider=SGLangPlacementAdapter(_build_metadata()),
        )


# -- Empty-rank discipline ------------------------------------------------


def test_route_empty_topk_still_participates_in_all_reduce():
    """Under DP-attention, empty ranks must still hit the collective.

    The runtime must call ``ep_group.all_reduce`` even when ``topk_ids`` is
    empty; the policy's empty-batch fast path then skips the LP solve.
    """
    ep_group = _RecordingEpGroup()
    runtime = _build_runtime(ep_group=ep_group)

    empty_ids = torch.empty((0, 3), dtype=torch.int32, device="cuda")
    empty_weights = torch.empty((0, 3), dtype=torch.float32, device="cuda")
    result = runtime.route(_topk_output(empty_ids, empty_weights))

    assert len(ep_group.all_reduce_calls) == 1
    seen_shape, seen_dtype = ep_group.all_reduce_calls[0]
    assert seen_shape == (4,)  # num_logical_experts
    assert seen_dtype == "torch.float32"  # local_counts cast before AR
    # Output preserves the empty input shape/dtype.
    assert result.topk_ids.shape == (0, 3)
    assert result.topk_ids.dtype == torch.int32


def test_route_empty_topk_with_no_ep_group_does_not_crash():
    """For pre-distributed-init paths the runtime should still produce a
    valid empty output without invoking a collective."""
    runtime = _build_runtime(ep_group=None)
    empty_ids = torch.empty((0, 3), dtype=torch.int32, device="cuda")
    empty_weights = torch.empty((0, 3), dtype=torch.float32, device="cuda")
    result = runtime.route(_topk_output(empty_ids, empty_weights))
    assert result.topk_ids.shape == (0, 3)


# -- Populated-batch routing ---------------------------------------------


def test_route_populated_topk_returns_valid_physical_ids():
    """Every output topk slot must be a valid physical replica of the
    corresponding logical input id."""
    runtime = _build_runtime()
    snap = runtime.placement_provider.snapshot(layer_id=0)
    log2phy = snap.logical_to_physical_candidates.cpu()
    log_counts = snap.logical_to_physical_count.cpu()

    topk_ids = torch.tensor(
        [[0, 1, 2], [2, 3, 0], [0, 2, 1], [2, 0, 3]],
        dtype=torch.int32,
        device="cuda",
    )
    topk_weights = torch.full(topk_ids.shape, 0.25, dtype=torch.float32, device="cuda")
    result = runtime.route(_topk_output(topk_ids, topk_weights))

    assert result.topk_ids.shape == topk_ids.shape
    assert result.topk_ids.dtype == topk_ids.dtype
    assert torch.equal(result.topk_weights, topk_weights)

    out_cpu = result.topk_ids.cpu()
    in_cpu = topk_ids.cpu()
    for t in range(in_cpu.shape[0]):
        for k in range(in_cpu.shape[1]):
            logical = int(in_cpu[t, k])
            chosen = int(out_cpu[t, k])
            valid_set = set(
                int(log2phy[logical, i]) for i in range(int(log_counts[logical]))
            )
            assert chosen in valid_set, (
                f"token {t} slot {k}: logical {logical} -> physical {chosen} "
                f"not in valid replicas {valid_set}"
            )


def test_route_single_copy_logicals_map_deterministically():
    """Single-copy logicals have only one valid physical; LPLB must always
    pick that one slot regardless of LP solver output."""
    runtime = _build_runtime()
    # Layer 0 layout: logical 1 has one copy at phys 1; logical 3 at phys 5.
    topk_ids = torch.tensor([[1, 3, 1], [3, 1, 3]], dtype=torch.int32, device="cuda")
    topk_weights = torch.ones(topk_ids.shape, dtype=torch.float32, device="cuda")
    result = runtime.route(_topk_output(topk_ids, topk_weights))

    expected = torch.tensor([[1, 5, 1], [5, 1, 5]], dtype=torch.int32, device="cuda")
    assert torch.equal(result.topk_ids, expected)


def test_route_replicated_logicals_distribute_across_copies():
    """Across many tokens hitting the same replicated logical, LPLB should
    distribute across both physical copies (not all to the same one)."""
    runtime = _build_runtime()
    # 100 tokens all hitting logical 0 (replicas: phys 0 and phys 3).
    n = 100
    topk_ids = torch.zeros((n, 1), dtype=torch.int32, device="cuda")
    topk_weights = torch.ones((n, 1), dtype=torch.float32, device="cuda")
    result = runtime.route(_topk_output(topk_ids, topk_weights))

    chosen = result.topk_ids.cpu().flatten().tolist()
    assert set(chosen).issubset({0, 3})
    # Reasonable spread: neither replica monopolizes (< 95% to one side).
    count_0 = sum(c == 0 for c in chosen)
    count_3 = sum(c == 3 for c in chosen)
    assert max(count_0, count_3) < 0.95 * n, (
        f"degenerate LP distribution: phys 0 = {count_0}, phys 3 = {count_3}"
    )


# -- Placement version invalidation ---------------------------------------


def test_placement_bump_triggers_router_state_rebuild():
    """Bumping the placement version on the adapter must cause the
    LPLBL2Router cache to invalidate on the next route call."""
    runtime = _build_runtime()
    topk_ids = torch.tensor([[0, 2]], dtype=torch.int32, device="cuda")
    topk_weights = torch.ones((1, 2), dtype=torch.float32, device="cuda")

    runtime.route(_topk_output(topk_ids, topk_weights))
    cache_keys_before = sorted(runtime.router._layer_state.keys())
    assert cache_keys_before == [(0, 0)]

    runtime.placement_provider.bump_version()
    runtime.route(_topk_output(topk_ids, topk_weights))

    cache_keys_after = sorted(runtime.router._layer_state.keys())
    # Older (layer_id=0, version=0) entry is evicted; only the new key remains.
    assert cache_keys_after == [(0, 1)]


# -- Local-count buffer reuse --------------------------------------------


def test_local_counts_buffer_reused_across_calls():
    """The runtime should allocate its scatter-add buffer once."""
    runtime = _build_runtime()
    topk_ids = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    topk_weights = torch.ones((1, 3), dtype=torch.float32, device="cuda")

    runtime.route(_topk_output(topk_ids, topk_weights))
    first_id = id(runtime._local_counts)
    runtime.route(_topk_output(topk_ids, topk_weights))
    runtime.route(_topk_output(topk_ids, topk_weights))
    assert id(runtime._local_counts) == first_id
    assert runtime._local_counts.shape == (4,)
    assert runtime._local_counts.device.type == "cuda"
