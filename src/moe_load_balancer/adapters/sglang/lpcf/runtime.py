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
"""SGLang adapter runtime for LP-CF.

Per-layer glue that counts logical-expert hits, optionally all-reduces them
across the EP group, and dispatches to :class:`LPCFL2Router`. The runtime
returns a remapped ``StandardTopKOutput`` with ``topk_ids`` in the physical
namespace; ``topk_weights`` and ``router_logits`` are forwarded unchanged.

Default behavior runs on each rank's local logical-expert count (no
collective). Set ``LPCF_USE_GLOBAL_COUNT=1`` to all-reduce per layer.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor

from moe_load_balancer.core.types import (
    PlacementSnapshot,
    RoutingPolicyConfig,
    RoutingRequest,
    RoutingStage,
)
from moe_load_balancer.policies.l2.lpcf import LPCFL2Router


_FORWARD_MODE_TO_STAGE: dict[str, RoutingStage] = {
    "EXTEND": RoutingStage.PREFILL,
    "DECODE": RoutingStage.DECODE,
    "MIXED": RoutingStage.MIXED,
    "IDLE": RoutingStage.IDLE,
    "TARGET_VERIFY": RoutingStage.SPECULATIVE,
    "DRAFT_EXTEND": RoutingStage.SPECULATIVE,
}


def _resolve_stage_from_forward_batch(forward_batch) -> Optional[RoutingStage]:
    if forward_batch is None:
        return None
    forward_mode = getattr(forward_batch, "forward_mode", None)
    if forward_mode is None:
        return None
    name = getattr(forward_mode, "name", None) or str(forward_mode).upper()
    return _FORWARD_MODE_TO_STAGE.get(name)


def _resolve_num_token_non_padded(forward_batch) -> Optional[Tensor]:
    if forward_batch is None:
        return None
    return getattr(forward_batch, "num_token_non_padded", None)


@triton.jit
def _count_local_logical_triton(
    counts_ptr,
    topk_ids_ptr,
    num_token_non_padded_ptr,
    n_total,
    TOPK: tl.constexpr,
    NUM_LOGICAL: tl.constexpr,
    HAS_GATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offs < n_total
    row = offs // TOPK
    if HAS_GATE:
        gate = tl.load(num_token_non_padded_ptr)
        valid = in_bounds & (row < gate)
    else:
        valid = in_bounds
    ids = tl.load(topk_ids_ptr + offs, mask=valid, other=-1)
    valid_id = valid & (ids >= 0) & (ids < NUM_LOGICAL)
    tl.atomic_add(counts_ptr + tl.where(valid_id, ids, 0), 1, mask=valid_id)


# Per-device 0-D placeholder used when no gate tensor is supplied.
_GATE_DUMMY: dict = {}


def _count_local_logical(
    counts: Tensor,
    topk_ids: Tensor,
    num_token_non_padded: Optional[Tensor],
) -> None:
    """``counts`` (pre-zeroed int32 [num_logical]) gets per-logical hit counts
    from ``topk_ids`` ([num_tokens, topk]). ``num_token_non_padded`` (0-D int
    tensor or None) gates valid rows."""
    topk = int(topk_ids.shape[1])
    num_logical = int(counts.shape[0])
    n_total = topk_ids.numel()
    flat = topk_ids.contiguous().view(-1)
    has_gate = num_token_non_padded is not None
    if has_gate:
        gate_arg = num_token_non_padded
    else:
        dev = topk_ids.device
        gate_arg = _GATE_DUMMY.get(dev)
        if gate_arg is None:
            gate_arg = torch.zeros((), dtype=torch.int32, device=dev)
            _GATE_DUMMY[dev] = gate_arg
    block = 1024
    grid = ((n_total + block - 1) // block,)
    _count_local_logical_triton[grid](
        counts, flat, gate_arg, n_total,
        TOPK=topk,
        NUM_LOGICAL=num_logical,
        HAS_GATE=has_gate,
        BLOCK=block,
    )


class LPCFRuntime:
    """SGLang per-layer LP-CF runtime."""

    def __init__(
        self,
        *,
        layer_id: int,
        ep_group,
        num_logical_experts: int,
        num_physical_experts: int,
        num_gpus: int,
        placement_provider,
        enabled_stages: Optional[frozenset[RoutingStage]] = None,
    ):
        if num_logical_experts <= 0:
            raise ValueError("LPCFRuntime.num_logical_experts must be positive.")
        if num_physical_experts <= 0:
            raise ValueError("LPCFRuntime.num_physical_experts must be positive.")
        if num_gpus <= 0:
            raise ValueError("LPCFRuntime.num_gpus must be positive.")

        self.layer_id = layer_id
        self.ep_group = ep_group
        self.num_logical_experts = num_logical_experts
        self.num_physical_experts = num_physical_experts
        self.num_gpus = num_gpus
        self.placement_provider = placement_provider
        self.enabled_stages = enabled_stages

        self.router = LPCFL2Router(
            ep_size=num_gpus,
            num_physical_experts=num_physical_experts,
        )

        # LPCF_USE_GLOBAL_COUNT=1: all-reduce per-logical counts before solving.
        # Default (0): each rank uses its own DP shard counts.
        self._use_global_count = (
            os.environ.get("LPCF_USE_GLOBAL_COUNT", "0") == "1"
        )

        # Lazy — allocated on first route() once we know the device.
        self._local_counts: Optional[Tensor] = None

    def route(
        self,
        topk_output,
        *,
        forward_batch=None,
        num_token_non_padded: Optional[Tensor] = None,
    ):
        """Run LP-CF on this layer and return a remapped ``StandardTopKOutput``.

        ``num_token_non_padded`` (0-D int tensor) gates which rows of
        ``topk_ids`` contribute to the local logical-expert count; rows at
        index ``>= num_token_non_padded`` are padding and contribute zero.

        Empty-rank discipline (LPCF_USE_GLOBAL_COUNT=1): every EP rank must
        invoke this on every layer per forward pass for the collective to
        complete, even on zero-token batches.
        """
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights

        if num_token_non_padded is None:
            num_token_non_padded = _resolve_num_token_non_padded(forward_batch)
        local_counts = self._count_local_logical_hits(topk_ids, num_token_non_padded)

        if self._use_global_count and self.ep_group is not None:
            global_counts = local_counts.float()
            global_counts = self.ep_group.all_reduce(global_counts)
        else:
            global_counts = local_counts.to(torch.float32)

        stage = _resolve_stage_from_forward_batch(forward_batch)
        placement = self.placement_provider.snapshot(self.layer_id)

        request = RoutingRequest(
            layer_id=self.layer_id,
            logical_topk_ids=topk_ids,
            topk_weights=topk_weights,
            placement=placement,
            stage=stage,
            global_logical_count=global_counts,
            token_count=num_token_non_padded,
            policy=RoutingPolicyConfig(
                name="lpcf",
                enabled_stages=self.enabled_stages,
            ),
        )

        decision = self.router.route_tokens(request)

        from sglang.srt.layers.moe.topk import StandardTopKOutput
        return StandardTopKOutput(
            topk_weights=decision.topk_weights,
            topk_ids=decision.routed_physical_topk_ids,
            router_logits=topk_output.router_logits,
        )

    def _count_local_logical_hits(
        self,
        topk_ids: Tensor,
        num_token_non_padded: Optional[Tensor],
    ) -> Tensor:
        device = topk_ids.device
        if self._local_counts is None or self._local_counts.device != device:
            self._local_counts = torch.zeros(
                self.num_logical_experts, dtype=torch.int32, device=device,
            )
        else:
            self._local_counts.zero_()
        if topk_ids.numel() == 0:
            return self._local_counts
        _count_local_logical(self._local_counts, topk_ids, num_token_non_padded)
        return self._local_counts
