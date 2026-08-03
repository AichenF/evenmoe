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
"""LPLB L2 router — LP-based redirection among physical replicas.

Framework-neutral: operates on :class:`PlacementSnapshot` inputs and
routes the four kernels through an injected :class:`LPLBKernels`
implementation supplied by the framework adapter.

Algorithm overview
------------------

The router solves a small LP per MoE layer per forward pass. The LP picks,
for every replicated logical expert, how to split that expert's *global*
token count across its physical replicas so the **peak per-GPU routed load**
is minimized:

    minimize  M + big_m_penalty * mu
    s.t.       C  t            = global_count_replicated      (token conservation)
              B2 t + I v − 1 M = − B1 t1                       (per-GPU peak with single-copy load fixed)
              t, v, M >= 0,  mu Big-M slack

The LP variables are: ``t`` per replicated physical slot, ``v`` per-GPU
slack, ``M`` peak, and one Big-M auxiliary. Single-copy logicals contribute
a fixed per-GPU baseline (``B1 t1``) that the per-batch ``prep_lp_inputs``
kernel folds into the RHS.

Per-layer state
---------------

The LP constraint matrices (``B1``, ``A_base``, ``c_vec``, ``log2phy``, …)
depend only on the placement — not on per-batch data. They are pre-built
once per ``(layer_id, placement_version)`` and cached. The adapter signals
placement changes by bumping
``PlacementSnapshot.metadata["placement_version"]``; the router rebuilds the
state on the next ``route_tokens`` call.

Sync-free
---------

The policy itself performs **no collectives** and **no host-GPU syncs**.
It expects ``RoutingRequest.global_logical_count`` to already be the
post-all-reduce per-logical-expert token counts; the adapter is responsible
for the local scatter-add and EP all-reduce upstream. Placement-derived
matrices are cached per layer/version. Mutable outputs (``A_full``, ``b``,
``t1``, ``x``, ``log2phy_prob``) are isolated per CUDA graph or eager stream
so overlapping execution cannot overwrite another batch's LP state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Hashable, Optional, Tuple

import torch
from torch import Tensor

from moe_load_balancer.core.l2 import TokenRouter
from moe_load_balancer.core.types import (
    PlacementSnapshot,
    RoutingDecision,
    RoutingPhase,
    RoutingPolicyConfig,
    RoutingRequest,
)
from moe_load_balancer.kernels.lplb import LPLBKernels

# Sentinel preserved through the no-redundancy fallback for the
# "-1 padded slot" convention frameworks use for invalid TopK entries.
_INVALID_ID = -1


@dataclass
class _LPLBLayerState:
    """Pre-built LP constraint matrices + reusable workspace for one layer.

    Constructed lazily by ``LPLBL2Router._get_or_build_layer_state`` on the
    first ``route_tokens`` call seeing a ``(layer_id, placement_version)``
    pair. The constant matrices (everything except the workspace tail) are
    derived from ``PlacementSnapshot``; the workspace tensors are
    pre-allocated once and overwritten in place by every routing call.
    """

    # Geometry
    num_logical: int
    num_phy: int
    num_gpus: int
    has_redundancy: bool

    # Logical / physical partition into single-copy and replicated.
    log_single: Tensor  # [num_single] int64
    phy_single: Tensor  # [num_single] int64
    log_replicated: Tensor  # [num_red_log] int64
    phy_replicated: Tensor  # [num_red_phy] int64

    # Constant LP inputs (built once from the placement snapshot).
    B1: Tensor  # [num_gpus, num_single] float32
    A_base: Tensor  # [NC, NV-1] float32 — A_full first NV-1 cols
    A_base_row_sum: Tensor  # [NC] float32 (precomputed for prep kernel)
    c_vec: Tensor  # [NV] float32 — objective coefficients
    log2phy: Tensor  # [num_logical, max_copies] int64 (-1 padded)
    log2phy_map_for_dispatch: Tensor  # [num_logical, max_copies] int32 view

    # Workspace template (cloned per CUDA graph or eager stream).
    A_full: Tensor  # [NC, NV] float32 — first NV-1 cols hold A_base
    b: Tensor  # [NC] float32
    t1: Tensor  # [num_single] float32
    x: Tensor  # [NV] float32
    log2phy_prob: Tensor  # [num_logical, max_copies] float32


def _build_layer_state(
    snapshot: PlacementSnapshot,
    num_gpus: int,
    big_m_penalty: float,
) -> _LPLBLayerState:
    """Build LP matrices + workspace for one MoE layer.

    Consumes a framework-neutral :class:`PlacementSnapshot`. Validates the
    contiguous-per-rank ownership layout (``num_phy % num_gpus == 0``).

    Degenerate case: when no logical expert has more than one physical copy
    (``has_redundancy=False``), most of the LP machinery is skipped. The
    returned state still allocates a ``log2phy_prob`` buffer with the same
    shape as the placement so callers downstream of a future has_redundancy
    transition do not need to special-case the shape.
    """

    log2phy_raw = snapshot.logical_to_physical_candidates
    log_counts_raw = snapshot.logical_to_physical_count

    if not isinstance(log2phy_raw, Tensor):
        raise TypeError(
            "LPLBL2Router requires PlacementSnapshot.logical_to_physical_candidates "
            "to be a torch.Tensor."
        )
    if not isinstance(log_counts_raw, Tensor):
        raise TypeError(
            "LPLBL2Router requires PlacementSnapshot.logical_to_physical_count "
            "to be a torch.Tensor."
        )

    num_logical = snapshot.num_logical_experts
    num_phy = snapshot.num_physical_experts
    if num_phy % num_gpus != 0:
        raise ValueError(
            "LPLBL2Router requires PlacementSnapshot.num_physical_experts "
            f"({num_phy}) divisible by num_gpus ({num_gpus}); the LP "
            "constraint matrices assume per-rank contiguous ownership."
        )
    num_phy_per_gpu = num_phy // num_gpus

    device = log2phy_raw.device
    log2phy = log2phy_raw.to(torch.int64).contiguous()
    log_counts = log_counts_raw.to(torch.int64)
    max_copies = log2phy.shape[1]

    # Reconstruct phy2log from log2phy + log_counts.
    # For every (logical i, slot j < log_counts[i]) pair, log2phy[i, j] is a
    # valid physical expert id and we want phy2log[log2phy[i, j]] = i.
    j_idx = torch.arange(max_copies, device=device).unsqueeze(0).expand(num_logical, -1)
    valid_mask = j_idx < log_counts.unsqueeze(1)
    valid_phy = log2phy[valid_mask]
    valid_log = torch.arange(num_logical, device=device).unsqueeze(1).expand_as(j_idx)[valid_mask]

    # Defensive validation: every valid_phy entry must be a real physical
    # slot index, otherwise the scatter below silently OOBs and the next
    # ``log_counts[phy2log]`` lookup triggers an async device-side assert
    # that surfaces minutes later as an opaque CUDA error. Some placements
    # can leave physical slots that no logical claims (see below); the
    # range check here catches genuinely malformed inputs first.
    if valid_phy.numel() > 0:
        vp_min = int(valid_phy.min().item())
        vp_max = int(valid_phy.max().item())
        if vp_min < 0 or vp_max >= num_phy:
            raise RuntimeError(
                f"LPLB _build_layer_state: valid_phy out of [0, {num_phy}) "
                f"(observed min={vp_min}, max={vp_max}). "
                f"log2phy.shape={tuple(log2phy.shape)}, "
                f"log_counts shape={tuple(log_counts.shape)}, "
                f"log_counts min/max="
                f"{int(log_counts.min().item())}/"
                f"{int(log_counts.max().item())}."
            )

    # Initialize phy2log to a sentinel (num_logical) so we can distinguish
    # physical slots that no logical claims through
    # logical_to_physical_candidates. Some placements reserve
    # ``num_physical_experts`` slots that include redundant capacity but
    # only populate the first ``num_logical`` of them — the rest stay
    # unclaimed. That is a legitimate state: those slots simply carry no
    # expert, so the LP must ignore them (they're not in any logical's
    # replica set, so they contribute nothing to peak GPU load). The
    # masked indexing below skips them safely.
    _PHY_UNCOVERED = num_logical  # sentinel: out-of-range for log_counts
    phy2log = torch.full((num_phy,), _PHY_UNCOVERED, dtype=torch.int64, device=device)
    phy2log[valid_phy] = valid_log
    is_covered = phy2log != _PHY_UNCOVERED

    # Partition logicals into single-copy vs replicated.
    log_single = torch.nonzero(log_counts == 1).flatten().to(torch.int64)
    phy_single = log2phy[log_single, 0].to(torch.int64).contiguous()
    log_replicated = torch.nonzero(log_counts > 1).flatten().to(torch.int64)
    # phy_replicated: physical slots that (a) are claimed by some logical
    # AND (b) that logical has more than one replica. Uncovered slots are
    # masked out before the log_counts gather so the indexing stays in
    # bounds even when ``_PHY_UNCOVERED == num_logical`` would otherwise
    # OOB.
    safe_phy2log = phy2log.clamp(max=num_logical - 1)
    phy_replicated = (
        torch.nonzero(is_covered & (log_counts[safe_phy2log] > 1))
        .flatten()
        .to(torch.int64)
        .contiguous()
    )

    num_single = int(log_single.numel())
    num_red_log = int(log_replicated.numel())
    num_red_phy = int(phy_replicated.numel())
    has_redundancy = num_red_log > 0

    log2phy_map_for_dispatch = log2phy.to(torch.int32).contiguous()

    if not has_redundancy:
        # No LP needed — pre-allocate the minimum state so identity dispatch
        # has the buffers it needs.
        empty_f32 = torch.empty(0, dtype=torch.float32, device=device)
        return _LPLBLayerState(
            num_logical=num_logical,
            num_phy=num_phy,
            num_gpus=num_gpus,
            has_redundancy=False,
            log_single=log_single,
            phy_single=phy_single,
            log_replicated=log_replicated,
            phy_replicated=phy_replicated,
            B1=torch.zeros((num_gpus, num_single), dtype=torch.float32, device=device),
            A_base=empty_f32.reshape(0, 0),
            A_base_row_sum=empty_f32,
            c_vec=empty_f32,
            log2phy=log2phy,
            log2phy_map_for_dispatch=log2phy_map_for_dispatch,
            A_full=empty_f32.reshape(0, 0),
            b=empty_f32,
            t1=torch.empty(num_single, dtype=torch.float32, device=device),
            x=empty_f32,
            log2phy_prob=torch.empty(log2phy.shape, dtype=torch.float32, device=device),
        )

    # GPU ownership matrix B_full[i, j] = 1 iff GPU i owns physical slot j.
    B_full = torch.zeros((num_gpus, num_phy), dtype=torch.float32, device=device)
    for i in range(num_gpus):
        B_full[i, i * num_phy_per_gpu : (i + 1) * num_phy_per_gpu] = 1
    B1 = B_full[:, phy_single].contiguous()
    B2 = B_full[:, phy_replicated]

    # Copy-to-logical mapping C[i, j] = 1 iff phy_replicated[j] belongs to
    # log_replicated[i].
    C = torch.zeros((num_red_log, num_red_phy), dtype=torch.float32, device=device)
    phy2log_rep = phy2log[phy_replicated]
    for i in range(num_red_log):
        C[i, phy2log_rep == log_replicated[i]] = 1.0

    # A_base = [[C 0 0], [B2 I -1]] (the Big-M aux column is appended live
    # via A_full[:, -1] from the prep kernel each batch).
    zeros_top_g = torch.zeros((num_red_log, num_gpus), dtype=torch.float32, device=device)
    zeros_top_1 = torch.zeros((num_red_log, 1), dtype=torch.float32, device=device)
    I_g = torch.eye(num_gpus, dtype=torch.float32, device=device)
    neg_ones = torch.full((num_gpus, 1), -1.0, dtype=torch.float32, device=device)
    A_top = torch.hstack([C, zeros_top_g, zeros_top_1])
    A_bottom = torch.hstack([B2, I_g, neg_ones])
    A_base = torch.vstack([A_top, A_bottom]).contiguous()

    nc, nv_minus_1 = A_base.shape
    nv = nv_minus_1 + 1  # +1 for the Big-M auxiliary column

    # Objective: minimize M (second-to-last variable), penalize the Big-M aux.
    c_vec = torch.zeros(nv, dtype=torch.float32, device=device)
    c_vec[-2] = 1.0
    c_vec[-1] = big_m_penalty

    A_base_row_sum = A_base.sum(dim=1).contiguous()

    # Workspace tensors (overwritten per batch by the kernels).
    A_full = torch.empty(nc, nv, dtype=torch.float32, device=device)
    A_full[:, : nv - 1].copy_(A_base)
    b = torch.empty(nc, dtype=torch.float32, device=device)
    t1 = torch.empty(num_single, dtype=torch.float32, device=device)
    x = torch.empty(nv, dtype=torch.float32, device=device)
    log2phy_prob = torch.empty(log2phy.shape, dtype=torch.float32, device=device)

    return _LPLBLayerState(
        num_logical=num_logical,
        num_phy=num_phy,
        num_gpus=num_gpus,
        has_redundancy=True,
        log_single=log_single,
        phy_single=phy_single,
        log_replicated=log_replicated,
        phy_replicated=phy_replicated,
        B1=B1,
        A_base=A_base,
        A_base_row_sum=A_base_row_sum,
        c_vec=c_vec,
        log2phy=log2phy,
        log2phy_map_for_dispatch=log2phy_map_for_dispatch,
        A_full=A_full,
        b=b,
        t1=t1,
        x=x,
        log2phy_prob=log2phy_prob,
    )


def _identity_dispatch(topk_ids: Tensor, snapshot: PlacementSnapshot) -> Tensor:
    """Map logical top-k ids to physical ids without LP redistribution.

    Used when the layer has no replicated logicals (``has_redundancy=False``)
    so the LP would degenerate. Falls back, in order:

    1. ``snapshot.default_physical_for_logical[logical_id]`` if provided
       (this is the framework's precomputed rank-dispatch baseline).
    2. ``snapshot.logical_to_physical_candidates[logical_id, 0]`` otherwise.

    Tokens with ``logical_id < 0`` (padding / invalid slots) are preserved
    as-is so framework TopK padding conventions are respected.
    """
    log2phy = snapshot.logical_to_physical_candidates
    default_map = snapshot.default_physical_for_logical
    valid = topk_ids >= 0
    safe_ids = topk_ids.to(torch.int64).clamp(min=0)

    if default_map is not None:
        physical = default_map.to(torch.int64)[safe_ids]
    else:
        physical = log2phy.to(torch.int64)[safe_ids, 0]

    result = torch.where(valid, physical, topk_ids.to(torch.int64))
    return result.to(topk_ids.dtype)


class LPLBL2Router(TokenRouter):
    """LP-based per-batch routing among physical replicas.

    Per-batch state arrives via ``RoutingRequest``:

    - ``request.placement``: required. The ``logical_to_physical_candidates``,
      ``logical_to_physical_count``, and ``num_physical_experts`` fields drive
      the LP constraint construction. ``metadata["placement_version"]`` keys
      the per-layer state cache; bump it on EPLB rebalance to force a rebuild.
    - ``request.global_logical_count``: required. Per-logical-expert token
      count summed across the EP group; the adapter is responsible for the
      local scatter-add and EP all-reduce upstream.
    - ``request.logical_topk_ids``: required. ``[num_tokens, top_k]`` tensor
      of logical expert ids from the router.

    Stage gating is driven by the policy config selected by the unified
    orchestrator. The request carries an ordered policy pipeline rather than
    embedding a second, competing single-policy field.
    """

    phase = RoutingPhase.PHYSICAL_ROUTING

    def __init__(
        self,
        *,
        kernels: LPLBKernels,
        num_gpus: int,
        big_m_penalty: float = 1000.0,
        ipm_iters: int = 5,
    ):
        if num_gpus <= 0:
            raise ValueError("LPLBL2Router.num_gpus must be positive.")
        if big_m_penalty <= 0:
            raise ValueError("LPLBL2Router.big_m_penalty must be positive.")
        if ipm_iters <= 0:
            raise ValueError("LPLBL2Router.ipm_iters must be positive.")

        self._kernels = kernels
        self.num_gpus = num_gpus
        self.big_m_penalty = big_m_penalty
        self.ipm_iters = ipm_iters
        # Per-layer state keyed by (layer_id, placement_version).
        self._layer_state: Dict[Tuple[int, int], _LPLBLayerState] = {}
        # Mutable LP workspaces cannot be shared by overlapping CUDA graphs.
        self._execution_state: Dict[Tuple[int, int, Hashable], _LPLBLayerState] = {}

    def prepare_layer(self, snapshot: PlacementSnapshot) -> None:
        """Build placement-dependent matrices outside CUDA graph capture."""

        self._get_or_build_layer_state(snapshot.layer_id, snapshot)

    def route_tokens(
        self,
        request: RoutingRequest,
        policy: Optional[RoutingPolicyConfig] = None,
    ) -> RoutingDecision:
        if policy is None:
            if len(request.policies) != 1:
                raise ValueError(
                    "Direct LPLBL2Router calls require exactly one request policy; "
                    "the unified orchestrator always passes the selected policy explicitly."
                )
            policy = request.policies[0]
        if not policy.should_run_for_stage(request.stage):
            return RoutingDecision(
                routed_physical_topk_ids=request.logical_topk_ids,
                topk_weights=request.topk_weights,
                metadata={"policy": policy.name, "skipped": "stage_gated"},
            )

        if request.placement is None:
            raise ValueError(
                "LPLBL2Router requires RoutingRequest.placement to be a "
                "PlacementSnapshot (logical->physical translation is non-optional)."
            )
        if request.global_logical_count is None:
            raise ValueError(
                "LPLBL2Router requires RoutingRequest.global_logical_count "
                "(the adapter must count + all-reduce per-logical-expert "
                "token counts before invoking the router)."
            )

        topk_ids = request.logical_topk_ids
        if not isinstance(topk_ids, Tensor):
            raise TypeError("LPLBL2Router requires torch.Tensor logical_topk_ids.")

        base_state = self._get_or_build_layer_state(
            request.layer_id,
            request.placement,
        )

        # No-redundancy fast path: LP is degenerate; route to the canonical
        # physical slot for each logical. Still passes through empty batches
        # untouched (caller upstream is expected to short-circuit the path
        # when num_tokens == 0, but be tolerant here).
        if not base_state.has_redundancy:
            return RoutingDecision(
                routed_physical_topk_ids=_identity_dispatch(topk_ids, request.placement),
                topk_weights=request.topk_weights,
                metadata={"policy": policy.name, "lp_skipped": "no_redundancy"},
            )

        # Empty-token fast path: the adapter is responsible for participating
        # in the EP all-reduce on empty ranks before calling us; here we just
        # avoid invoking the kernels on a degenerate input.
        if topk_ids.shape[0] == 0:
            return RoutingDecision(
                routed_physical_topk_ids=topk_ids,
                topk_weights=request.topk_weights,
                metadata={"policy": policy.name, "lp_skipped": "empty_batch"},
            )

        state = self._get_execution_state(
            request.layer_id,
            request.placement,
            topk_ids,
            base_state,
        )

        # Cast global_logical_count to the float32 the prep kernel expects.
        global_counts = request.global_logical_count
        if not isinstance(global_counts, Tensor):
            raise TypeError("LPLBL2Router requires torch.Tensor global_logical_count.")
        if global_counts.dtype != torch.float32:
            global_counts = global_counts.to(torch.float32)

        # Four-stage pipeline. Mutable outputs belong to this execution state.
        self._kernels.prep_lp_inputs(
            state.A_full,
            state.b,
            state.t1,
            global_counts,
            state.log_single,
            state.log_replicated,
            state.B1,
            state.A_base_row_sum,
        )
        self._kernels.solve_ipm(
            state.A_full,
            state.b,
            state.c_vec,
            num_iters=self.ipm_iters,
            result=state.x,
        )
        self._kernels.extract_log2phy_prob(
            state.log2phy_prob,
            state.x,
            state.t1,
            state.phy_single,
            state.phy_replicated,
            state.log2phy,
        )
        routed_physical = self._kernels.dispatch_probability(
            topk_ids,
            state.log2phy_prob,
            state.log2phy_map_for_dispatch,
        )

        return RoutingDecision(
            routed_physical_topk_ids=routed_physical,
            topk_weights=request.topk_weights,
            metadata={"policy": policy.name},
        )

    # -- Internal helpers --------------------------------------------------

    def _get_or_build_layer_state(
        self, layer_id: int, snapshot: PlacementSnapshot
    ) -> _LPLBLayerState:
        """Return the cached state for this (layer, version), building if needed.

        ``snapshot.metadata["placement_version"]`` keys the cache. The
        framework adapter bumps this whenever placement changes (e.g. on
        EPLB rebalance), so the next ``route_tokens`` call observes a miss
        and rebuilds the LP matrices.

        Older versions of the same layer are evicted from the cache when a
        new version arrives to keep the memory footprint bounded.
        """
        version = int(snapshot.metadata.get("placement_version", 0))
        key = (layer_id, version)
        cached = self._layer_state.get(key)
        if cached is not None:
            return cached

        state = _build_layer_state(snapshot, self.num_gpus, self.big_m_penalty)

        # Warmup is per-shape; only meaningful when LP is actually used.
        if state.has_redundancy:
            nc, nv = state.A_full.shape
            self._kernels.warmup(
                nc=nc,
                nv=nv,
                num_iters=self.ipm_iters,
                device=state.A_full.device,
            )

        # Evict any older versions of this layer; only the latest is needed.
        self._layer_state = {k: v for k, v in self._layer_state.items() if k[0] != layer_id}
        self._execution_state = {k: v for k, v in self._execution_state.items() if k[0] != layer_id}
        self._layer_state[key] = state
        return state

    def _get_execution_state(
        self,
        layer_id: int,
        snapshot: PlacementSnapshot,
        topk_ids: Tensor,
        base: _LPLBLayerState,
    ) -> _LPLBLayerState:
        """Return scratch buffers private to one CUDA graph or eager stream.

        Placement-derived matrices remain shared. Only buffers overwritten by
        the four-stage LP pipeline are cloned. This prevents a replay for one
        batch shape from corrupting another overlapping replay.
        """

        if not topk_ids.is_cuda:
            return base

        version = int(snapshot.metadata.get("placement_version", 0))
        if torch.cuda.is_current_stream_capturing():
            execution_key: Hashable = ("graph", topk_ids.data_ptr())
        else:
            execution_key = (
                "stream",
                torch.cuda.current_stream(topk_ids.device).cuda_stream,
            )
        cache_key = (layer_id, version, execution_key)
        cached = self._execution_state.get(cache_key)
        if cached is not None:
            return cached

        state = self._clone_execution_state(base)
        self._execution_state[cache_key] = state
        return state

    @staticmethod
    def _clone_execution_state(base: _LPLBLayerState) -> _LPLBLayerState:
        return replace(
            base,
            A_full=base.A_full.clone(),
            b=torch.empty_like(base.b),
            t1=torch.empty_like(base.t1),
            x=torch.empty_like(base.x),
            log2phy_prob=torch.empty_like(base.log2phy_prob),
        )
