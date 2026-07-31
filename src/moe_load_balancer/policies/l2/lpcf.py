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
"""LP-CF: closed-form pair-split L2 routing.

For each redundant logical expert the per-batch token mass is split across
its physical replicas in closed form (no LP/IPM solve):

- K=1 logicals: all mass goes to the single replica.
- K=2: split p = clip((L_b - L_a + G) / (2G), 0, 1) against the load
  snapshot after K=1 + K>=3 contributions.
- K>=3: uniform 1/K share per replica (typical R1 placement has <5% K>=3).

One Triton kernel writes the per-(logical, replica) probability table;
a second multinomial-samples each (token, slot) into a physical expert.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor

from moe_load_balancer.core.l2 import TokenRouter
from moe_load_balancer.core.types import (
    RoutingDecision,
    RoutingPhase,
    RoutingPolicyConfig,
    RoutingRequest,
)


_CF_BLOCK_DIM = 128
_DISPATCH_BLOCK_DIM = 256
# Caps the per-thread loop in the dispatch kernel. PlacementSnapshot stores
# logical_to_physical_candidates as (NL, num_physical_experts) but only the
# first K_i columns are valid for each logical; iterating num_physical_experts
# wastes registers and HBM traffic. 32 covers ep_num_redundant_experts <= 31.
_DEFAULT_K_BOUND = 32


@triton.jit
def _cf_pair_fused_kernel(
    log2phy_prob_ptr,        # (NL * MC,) fp32
    L_ptr,                   # (EP_SIZE,) fp32 scratch, pre-zeroed by host
    global_counts_ptr,       # (NL,) fp32
    log2phy_map_ptr,         # (NL * MC,) int32
    log_counts_ptr,          # (NL,) int32
    NUM_LOGICAL: tl.constexpr,
    MAX_COPIES: tl.constexpr,
    NUM_LOCAL_PHYS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    K_BOUND: tl.constexpr,
):
    total = 0.0
    for q_block in tl.static_range(0, NUM_LOGICAL, BLOCK_DIM):
        offs = q_block + tl.arange(0, BLOCK_DIM)
        in_bounds = offs < NUM_LOGICAL
        g_block = tl.load(global_counts_ptr + offs, mask=in_bounds, other=0.0)
        total = total + tl.sum(g_block)
    total = tl.maximum(total, 1.0)
    inv_total = 1.0 / total

    for q_block in tl.static_range(0, NUM_LOGICAL, BLOCK_DIM):
        offs = q_block + tl.arange(0, BLOCK_DIM)
        in_bounds = offs < NUM_LOGICAL
        k = tl.load(log_counts_ptr + offs, mask=in_bounds, other=0)
        g = tl.load(global_counts_ptr + offs, mask=in_bounds, other=0.0)
        phys = tl.load(log2phy_map_ptr + offs * MAX_COPIES + 0,
                       mask=in_bounds, other=-1)
        is_k1 = (k == 1) & (g > 0.0) & (phys >= 0) & in_bounds
        ranks = tl.where(is_k1, phys // NUM_LOCAL_PHYS, 0)
        tl.atomic_add(L_ptr + ranks, g, mask=is_k1)
        tl.store(log2phy_prob_ptr + offs * MAX_COPIES + 0,
                 g * inv_total, mask=is_k1)

    for kk in tl.static_range(K_BOUND):
        for q_block in tl.static_range(0, NUM_LOGICAL, BLOCK_DIM):
            offs = q_block + tl.arange(0, BLOCK_DIM)
            in_bounds = offs < NUM_LOGICAL
            k = tl.load(log_counts_ptr + offs, mask=in_bounds, other=0)
            g = tl.load(global_counts_ptr + offs, mask=in_bounds, other=0.0)
            phys = tl.load(log2phy_map_ptr + offs * MAX_COPIES + kk,
                           mask=in_bounds, other=-1)
            is_k3_slot = (k >= 3) & (g > 0.0) & (k > kk) & (phys >= 0) & in_bounds
            share = g / k.to(tl.float32)
            ranks = tl.where(is_k3_slot, phys // NUM_LOCAL_PHYS, 0)
            tl.atomic_add(L_ptr + ranks, share, mask=is_k3_slot)
            tl.store(log2phy_prob_ptr + offs * MAX_COPIES + kk,
                     share * inv_total, mask=is_k3_slot)

    for q_block in tl.static_range(0, NUM_LOGICAL, BLOCK_DIM):
        offs = q_block + tl.arange(0, BLOCK_DIM)
        in_bounds = offs < NUM_LOGICAL
        k = tl.load(log_counts_ptr + offs, mask=in_bounds, other=0)
        g = tl.load(global_counts_ptr + offs, mask=in_bounds, other=0.0)
        phys_a = tl.load(log2phy_map_ptr + offs * MAX_COPIES + 0,
                         mask=in_bounds, other=-1)
        phys_b = tl.load(log2phy_map_ptr + offs * MAX_COPIES + 1,
                         mask=in_bounds, other=-1)
        is_k2 = (k == 2) & (g > 0.0) & (phys_a >= 0) & (phys_b >= 0) & in_bounds
        r_a = tl.where(is_k2, phys_a // NUM_LOCAL_PHYS, 0)
        r_b = tl.where(is_k2, phys_b // NUM_LOCAL_PHYS, 0)
        same_rank = r_a == r_b
        L_a = tl.load(L_ptr + r_a, mask=is_k2, other=0.0)
        L_b = tl.load(L_ptr + r_b, mask=is_k2, other=0.0)
        safe_g = tl.where(g > 0.0, g, 1.0)
        p_eq = (L_b - L_a + safe_g) / (2.0 * safe_g)
        p = tl.where(same_rank, 0.5,
                     tl.maximum(0.0, tl.minimum(1.0, p_eq)))
        tl.store(log2phy_prob_ptr + offs * MAX_COPIES + 0,
                 (p * g) * inv_total, mask=is_k2)
        tl.store(log2phy_prob_ptr + offs * MAX_COPIES + 1,
                 ((1.0 - p) * g) * inv_total, mask=is_k2)


@triton.jit
def _dispatch_probability_kernel(
    out_topk_ids_ptr,        # (N,) int32
    in_topk_ids_ptr,         # (N,) int32 — logical ids
    log2phy_prob_ptr,        # (NL, MC) fp32
    log2phy_map_ptr,         # (NL, MC) int32
    random_vals_ptr,         # (N,) fp32 uniform [0, 1)
    N,
    MAX_COPIES: tl.constexpr,
    K_BOUND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offs < N
    logical = tl.load(in_topk_ids_ptr + offs, mask=in_bounds, other=0)
    col = tl.arange(0, K_BOUND)
    base = logical[:, None] * MAX_COPIES + col[None, :]
    probs = tl.load(log2phy_prob_ptr + base, mask=in_bounds[:, None], other=0.0)
    maps = tl.load(log2phy_map_ptr + base, mask=in_bounds[:, None], other=-1)
    row_sum = tl.sum(probs, axis=1)
    fallback = tl.where(maps >= 0, 1.0, 0.0)
    use_fb = row_sum <= 0.0
    final_probs = tl.where(use_fb[:, None], fallback, probs)
    final_row_sum = tl.where(use_fb, tl.sum(fallback, axis=1), row_sum)
    u = tl.load(random_vals_ptr + offs, mask=in_bounds, other=0.0) * final_row_sum
    cumsum = tl.cumsum(final_probs, axis=1)
    chosen = tl.sum((cumsum <= u[:, None]).to(tl.int32), axis=1)
    chosen = tl.minimum(chosen, K_BOUND - 1)
    onehot = (col[None, :] == chosen[:, None])
    out = tl.sum(tl.where(onehot, maps, 0), axis=1)
    tl.store(out_topk_ids_ptr + offs, out, mask=in_bounds)


def _ensure_int32_contig(t: Tensor) -> Tensor:
    if t.dtype == torch.int32 and t.is_contiguous():
        return t
    return t.contiguous().to(torch.int32)


class LPCFL2Router(TokenRouter):
    """LP-CF L2 router: closed-form pair-split + multinomial dispatch.

    Stateful: caches the int32 placement views per layer.  The cache key tracks
    tensor identity and PyTorch's in-place mutation version so an EPLB update
    of SGLang's live metadata is visible on the next routing call.
    """

    phase = RoutingPhase.PHYSICAL_ROUTING

    def __init__(self, ep_size: int, num_physical_experts: int) -> None:
        if ep_size <= 0 or num_physical_experts <= 0:
            raise ValueError("ep_size and num_physical_experts must be positive.")
        if num_physical_experts % ep_size != 0:
            raise ValueError(
                f"num_physical_experts={num_physical_experts} not divisible by "
                f"ep_size={ep_size}."
            )
        self.ep_size = ep_size
        self.num_physical_experts = num_physical_experts
        self._L_scratch: Optional[Tensor] = None
        self._log2phy_prob: Optional[Tensor] = None
        self._placement_cache: dict[int, tuple[tuple, Tensor, Tensor]] = {}
        self._dispatch_out_buf: Optional[Tensor] = None
        self._dispatch_random_buf: Optional[Tensor] = None

    def route_tokens(
        self,
        request: RoutingRequest,
        policy: RoutingPolicyConfig,
    ) -> RoutingDecision:
        if request.placement is None:
            raise ValueError("LPCFL2Router requires RoutingRequest.placement.")
        if request.global_logical_count is None:
            raise ValueError(
                "LPCFL2Router requires RoutingRequest.global_logical_count."
            )
        snapshot = request.placement
        candidates = snapshot.logical_to_physical_candidates
        log_counts = snapshot.logical_to_physical_count
        num_logical, max_copies = candidates.shape
        if snapshot.num_physical_experts != self.num_physical_experts:
            raise ValueError(
                f"Placement num_physical_experts={snapshot.num_physical_experts} "
                f"does not match router={self.num_physical_experts}."
            )

        topk_ids = request.logical_topk_ids
        if not isinstance(topk_ids, Tensor):
            raise TypeError("LPCFL2Router requires torch.Tensor logical_topk_ids.")
        if topk_ids.numel() == 0:
            return RoutingDecision(
                routed_physical_topk_ids=topk_ids,
                topk_weights=request.topk_weights,
                metadata={"policy": policy.name, "empty": True},
            )
        device = topk_ids.device
        global_counts = request.global_logical_count
        if global_counts.dtype != torch.float32:
            global_counts = global_counts.to(torch.float32)

        key = (
            candidates.data_ptr(),
            tuple(candidates.shape),
            getattr(candidates, "_version", None),
            log_counts.data_ptr(),
            tuple(log_counts.shape),
            getattr(log_counts, "_version", None),
        )
        cached = self._placement_cache.get(request.layer_id)
        if cached is None or cached[0] != key:
            log2phy_i32 = _ensure_int32_contig(candidates)
            log_counts_i32 = _ensure_int32_contig(log_counts)
            self._placement_cache[request.layer_id] = (
                key,
                log2phy_i32,
                log_counts_i32,
            )
        else:
            _, log2phy_i32, log_counts_i32 = cached

        if (
            self._log2phy_prob is None
            or self._log2phy_prob.shape != (num_logical, max_copies)
            or self._log2phy_prob.device != device
        ):
            self._log2phy_prob = torch.empty(
                (num_logical, max_copies), dtype=torch.float32, device=device,
            )

        if self._L_scratch is None or self._L_scratch.device != device:
            self._L_scratch = torch.zeros(
                self.ep_size, dtype=torch.float32, device=device,
            )
        else:
            self._L_scratch.zero_()

        self._log2phy_prob.zero_()
        num_local = self.num_physical_experts // self.ep_size
        _cf_pair_fused_kernel[(1,)](
            self._log2phy_prob.view(-1),
            self._L_scratch,
            global_counts,
            log2phy_i32.view(-1),
            log_counts_i32,
            NUM_LOGICAL=num_logical,
            MAX_COPIES=max_copies,
            NUM_LOCAL_PHYS=num_local,
            BLOCK_DIM=_CF_BLOCK_DIM,
            K_BOUND=min(max_copies, _DEFAULT_K_BOUND),
        )

        flat_in = _ensure_int32_contig(topk_ids.reshape(-1))
        n = flat_in.numel()
        if self._dispatch_out_buf is None or self._dispatch_out_buf.numel() < n \
                or self._dispatch_out_buf.device != device:
            self._dispatch_out_buf = torch.empty(
                n, dtype=torch.int32, device=device,
            )
            self._dispatch_random_buf = torch.empty(
                n, dtype=torch.float32, device=device,
            )
        out = self._dispatch_out_buf[:n]
        random_vals = self._dispatch_random_buf[:n]
        random_vals.uniform_()

        k_bound = min(max_copies, _DEFAULT_K_BOUND)
        grid = ((n + _DISPATCH_BLOCK_DIM - 1) // _DISPATCH_BLOCK_DIM,)
        _dispatch_probability_kernel[grid](
            out, flat_in,
            self._log2phy_prob, log2phy_i32,
            random_vals, n,
            MAX_COPIES=max_copies,
            K_BOUND=k_bound,
            BLOCK=_DISPATCH_BLOCK_DIM,
        )
        physical = out.view(topk_ids.shape)
        if topk_ids.dtype != torch.int32:
            physical = physical.to(topk_ids.dtype)
        return RoutingDecision(
            routed_physical_topk_ids=physical,
            topk_weights=request.topk_weights,
        )
