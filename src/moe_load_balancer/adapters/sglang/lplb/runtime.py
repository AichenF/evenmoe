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
"""SGLang adapter runtime for LPLB.

Replaces PR #24515's :class:`sglang.srt.eplb.lplb_solver.LPLBSolver` plumbing
with a framework-decoupled split: the LP algorithm lives in
:class:`LPLBL2Router` (framework-neutral); the SGLang-specific glue
(local-count scatter-add, EP all-reduce, ``forward_batch`` decomposition,
``StandardTopKOutput`` materialization) lives here.

The runtime is per MoE layer. ``ModelRunner._init_lplb_runtimes`` builds
one ``LPLBRuntime`` per ``metadata.num_layers``; each holds its own
:class:`LPLBL2Router` because the router caches per-layer LP matrix state
that should not be shared across layers.

Empty-rank discipline
---------------------

Under DP-attention some EP ranks see zero tokens. Every rank still has to
participate in the all-reduce inside :meth:`route`, otherwise the
collective deadlocks. ``model.deepseek_v2.forward_deepep`` (and its
``op_select_experts`` overlap variant) is responsible for calling
:meth:`route` on its empty branch with ``torch.empty((0, top_k), ...)``;
the runtime forwards a zero-count scatter to the EP group so the
collective ordering stays consistent across ranks.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from moe_load_balancer.adapters.sglang.lplb.kernels import SGLangLPLBKernels
from moe_load_balancer.core.types import (
    PlacementSnapshot,
    RoutingPolicyConfig,
    RoutingRequest,
    RoutingStage,
)
from moe_load_balancer.policies.l2.lplb import LPLBL2Router


# -- ForwardBatch decomposition (matches the Waterfill adapter helpers) ----


_FORWARD_MODE_TO_STAGE: dict[str, RoutingStage] = {
    "EXTEND": RoutingStage.PREFILL,
    "DECODE": RoutingStage.DECODE,
    "MIXED": RoutingStage.MIXED,
    "IDLE": RoutingStage.IDLE,
    "TARGET_VERIFY": RoutingStage.SPECULATIVE,
    "DRAFT_EXTEND": RoutingStage.SPECULATIVE,
}


def _resolve_stage_from_forward_batch(forward_batch) -> Optional[RoutingStage]:
    """Map ``forward_batch.forward_mode`` to :class:`RoutingStage`.

    Returns ``None`` when ``forward_batch`` is missing or the mode is
    unrecognized. The policy's stage gating treats ``None`` as
    "stage-agnostic" or "ineligible" depending on
    ``RoutingPolicyConfig.enabled_stages``.
    """
    if forward_batch is None:
        return None
    forward_mode = getattr(forward_batch, "forward_mode", None)
    if forward_mode is None:
        return None
    name = getattr(forward_mode, "name", None) or str(forward_mode).upper()
    return _FORWARD_MODE_TO_STAGE.get(name)


def _resolve_lb_token_mask(forward_batch) -> Optional[Tensor]:
    """SGLang ``ForwardBatch`` does not expose a stable LB mask field today."""
    if forward_batch is None:
        return None
    return getattr(forward_batch, "lb_token_mask", None)


def _resolve_num_token_non_padded(forward_batch) -> Optional[Tensor]:
    if forward_batch is None:
        return None
    return getattr(forward_batch, "num_token_non_padded", None)


# -- Model-arch gate -------------------------------------------------------


def _assert_lplb_supported_model(architecture: Optional[str]) -> None:
    """Defer to PR #24515's per-arch allow-list.

    LPLB requires every EP rank to call ``route`` on every layer (including
    empty-token ranks under DP-attention). MoE families whose empty-token
    paths return ``empty_topk_output`` *before* reaching the routing layer
    would deadlock the all-reduce. PR #24515 enumerates the validated
    architectures; the SDK adapter forwards that check rather than
    duplicating the allow-list.

    Falls back to a no-op when ``sglang.srt.eplb.lplb_solver`` is not
    importable (e.g. SGLang installs without PR #24515). The caller is
    expected to know it's running on a supported build.
    """
    if architecture is None:
        return
    try:
        from sglang.srt.eplb.lplb_solver import assert_lplb_supported_model
    except ImportError:
        return
    assert_lplb_supported_model(architecture)


# -- Runtime ---------------------------------------------------------------


class LPLBRuntime:
    """SGLang runtime adapter for LPLB.

    Construction parameters are SGLang-side runtime context (layer_id, EP
    group, expert count, model architecture). Per-batch state arrives via
    ``route(topk_output, forward_batch=...)``.
    """

    def __init__(
        self,
        *,
        layer_id: int,
        ep_group,
        num_logical_experts: int,
        num_physical_experts: int,
        num_gpus: int,
        placement_provider,
        model_architecture: Optional[str] = None,
        big_m_penalty: float = 1000.0,
        ipm_iters: int = 5,
        enabled_stages: Optional[frozenset[RoutingStage]] = None,
    ):
        if num_logical_experts <= 0:
            raise ValueError("LPLBRuntime.num_logical_experts must be positive.")
        if num_physical_experts <= 0:
            raise ValueError("LPLBRuntime.num_physical_experts must be positive.")
        if num_gpus <= 0:
            raise ValueError("LPLBRuntime.num_gpus must be positive.")

        # Model-arch gate fires before we touch any GPU resources so failures
        # surface at server startup, not on the first forward pass.
        _assert_lplb_supported_model(model_architecture)

        self.layer_id = layer_id
        self.ep_group = ep_group
        self.num_logical_experts = num_logical_experts
        self.num_physical_experts = num_physical_experts
        self.num_gpus = num_gpus
        self.placement_provider = placement_provider
        self.enabled_stages = enabled_stages

        self.kernels = SGLangLPLBKernels()
        self.router = LPLBL2Router(
            kernels=self.kernels,
            num_gpus=num_gpus,
            big_m_penalty=big_m_penalty,
            ipm_iters=ipm_iters,
        )

        # Reused local-count buffer. Allocated lazily on the first route()
        # call because we don't know the device until then.
        self._local_counts: Optional[Tensor] = None

    # -- public API --------------------------------------------------------

    def route(self, topk_output, *, forward_batch=None):
        """Run LPLB on this layer for the current forward pass.

        Mirrors the role of ``LPLBSolver.solve`` + ``topk_ids_logical_to_physical``
        in PR #24515 but routes through the framework-neutral
        :class:`LPLBL2Router`. Returns a new ``StandardTopKOutput`` with
        ``topk_ids`` remapped to physical experts; ``topk_weights`` and
        ``router_logits`` are preserved.

        Empty-rank discipline: this method *always* runs the local count +
        EP all-reduce, even on empty batches. Callers must invoke it on
        every layer on every rank for the collective to complete.
        """
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights

        # Step 1: count local logical-expert hits.
        local_counts = self._count_local_logical(topk_ids)

        # Step 2: EP all-reduce. Every rank participates (zero-token ranks
        # contribute a zero vector). Output is identical across ranks.
        global_counts = self._all_reduce_logical_counts(local_counts)

        # Step 3: decompose forward_batch.
        stage = _resolve_stage_from_forward_batch(forward_batch)
        lb_token_mask = _resolve_lb_token_mask(forward_batch)
        token_count = _resolve_num_token_non_padded(forward_batch)

        # Step 4: framework-neutral RoutingRequest. The placement snapshot
        # comes from a PlacementProvider that may itself wrap SGLang's live
        # ExpertLocationMetadata.
        placement = self.placement_provider.snapshot(self.layer_id)

        request = RoutingRequest(
            layer_id=self.layer_id,
            logical_topk_ids=topk_ids,
            topk_weights=topk_weights,
            placement=placement,
            stage=stage,
            global_logical_count=global_counts,
            token_count=token_count,
            lb_token_mask=lb_token_mask,
            policy=RoutingPolicyConfig(
                name="lplb",
                enabled_stages=self.enabled_stages,
            ),
        )

        decision = self.router.route_tokens(request)

        # Step 5: pack the decision back into a StandardTopKOutput.
        return self._build_topk_output(
            topk_output, decision.routed_physical_topk_ids, decision.topk_weights
        )

    # -- Internal helpers --------------------------------------------------

    def _count_local_logical(self, topk_ids: Tensor) -> Tensor:
        """Scatter-add per-logical-expert token counts.

        Mirrors :meth:`LPLBSolver.solve` step 1 in PR #24515. Uses
        ``scatter_add_`` (no nonzero/host-sync) so an empty topk_ids tensor
        is a true no-op. Buffer is reused across calls.
        """
        device = topk_ids.device
        if self._local_counts is None or self._local_counts.device != device:
            self._local_counts = torch.zeros(
                self.num_logical_experts, dtype=torch.int32, device=device
            )
        else:
            self._local_counts.zero_()

        flat_ids = topk_ids.flatten()
        # scatter_add over an empty flat_ids is a no-op, no host sync.
        self._local_counts.scatter_add_(
            0,
            flat_ids.to(torch.int64),
            torch.ones_like(flat_ids, dtype=torch.int32),
        )
        return self._local_counts

    def _all_reduce_logical_counts(self, local_counts: Tensor) -> Tensor:
        """Aggregate per-logical-expert counts across the EP group.

        Every EP rank must call this on every layer per forward pass; the
        collective inside the SGLang group may be in-place or out-of-place
        depending on the backend (pynccl in-place, ca_comm / pymscclpp
        out-of-place for small tensors), so we capture the return value.
        """
        global_counts = local_counts.float()
        if self.ep_group is not None:
            global_counts = self.ep_group.all_reduce(global_counts)
        return global_counts

    def _build_topk_output(
        self,
        original_output,
        routed_physical_topk_ids: Tensor,
        topk_weights: Tensor,
    ):
        """Repack the routing decision into SGLang's ``StandardTopKOutput``.

        Same convention as the Waterfill runtime: ``router_logits`` is
        preserved from the input (LPLB doesn't change the gating logits).
        """
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        return StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=routed_physical_topk_ids,
            router_logits=original_output.router_logits,
        )
