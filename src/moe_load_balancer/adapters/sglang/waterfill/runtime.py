"""SGLang adapter runtime for shared-expert Waterfill.

The runtime is the per-layer SGLang-side glue that:

1. Extracts per-rank routed counts from SGLang's TopK output (via
   :mod:`moe_load_balancer.adapters.sglang.load_kernels`).
2. Optionally runs the SGLang EP-group all-reduce in dynamic mode.
3. Decomposes the SGLang ``ForwardBatch`` to pull stage / token-mask /
   non-padded-token-count signals.
4. Builds a framework-neutral :class:`RoutingRequest` and dispatches to the
   :class:`WaterfillL2Router` (which lives in ``policies/l2/waterfill.py``).
5. Materializes the resulting :class:`RoutingDecision` into SGLang's
   ``StandardTopKOutput`` layout (with the shared-expert column appended in
   DeepEP interleaved-id layout).

The class is the canonical replacement for PR #19290's
``DeepEPWaterfillBalancer``. Construction parameters are SGLang-side runtime
context (EP rank/world, expert count, layout assumption); per-batch state is
read from ``forward_batch`` at ``expand_topk`` time.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from moe_load_balancer.adapters.sglang.load_kernels import count_local_routed_per_rank
from moe_load_balancer.core.types import (
    RoutingPolicyConfig,
    RoutingRequest,
    RoutingStage,
)
from moe_load_balancer.policies.l2.waterfill import WaterfillL2Router

# Sentinel expert id; DeepEP ignores expert_id < 0 entries on dispatch.
LOCAL_SHARED_MARKER = -1


class WaterfillRuntime:
    """SGLang runtime adapter for shared-expert Waterfill.

    Mirrors the API of PR #19290's ``DeepEPWaterfillBalancer.expand_topk`` but
    routes the per-token rank decision through ``WaterfillL2Router`` (the
    framework-neutral L2 algorithm) instead of inlining it in a fused Triton
    kernel.
    """

    MIN_BATCH_FOR_BALANCE = 64

    def __init__(
        self,
        *,
        layer_id: int,
        source_rank: int,
        world_size: int,
        num_routed_experts: int,
        local_preference: float = 1.1,
        routed_scaling_factor: float = 1.0,
        enabled_stages: Optional[frozenset[RoutingStage]] = None,
    ):
        if world_size <= 0:
            raise ValueError("WaterfillRuntime.world_size must be positive.")
        if not (0 <= source_rank < world_size):
            raise ValueError(
                "WaterfillRuntime.source_rank must satisfy 0 <= source_rank < world_size."
            )
        if num_routed_experts <= 0 or num_routed_experts % world_size != 0:
            raise ValueError(
                "WaterfillRuntime.num_routed_experts must be positive and divisible "
                "by world_size."
            )

        self.layer_id = layer_id
        self.source_rank = source_rank
        self.world_size = world_size
        self.num_routed_experts = num_routed_experts
        self.experts_per_rank = num_routed_experts // world_size
        # Each rank gets one extra slot for the shared expert column.
        self._new_experts_per_rank = self.experts_per_rank + 1
        self.shared_weight = (
            1.0 / routed_scaling_factor if routed_scaling_factor != 0 else 1.0
        )
        self.enabled_stages = enabled_stages

        # Lazy import: reading SGLang env requires sglang to be installed, which
        # is true at runtime but not necessarily during CPU pytest collection.
        from sglang.srt.environ import envs

        self.use_static_waterfill = not envs.SGLANG_DISABLE_STATIC_WATERFILL.get()

        self.router = WaterfillL2Router(
            source_rank=source_rank,
            local_preference=local_preference,
            world_size=world_size,
        )

        # Reused across calls to avoid allocator pressure on the count buffer.
        self._counts_buf: Optional[Tensor] = None

    # -- public API --------------------------------------------------------

    def expand_topk(
        self,
        topk_output,
        num_tokens: int,
        *,
        forward_batch=None,
    ):
        """Expand SGLang TopK output with a Waterfill-balanced shared expert column.

        ``forward_batch`` is the SGLang ``ForwardBatch`` for this forward
        pass; the adapter decomposes it internally to pull stage, token mask,
        and non-padded token count signals. Passing the whole object instead
        of pre-decomposed fields keeps the SGLang glue minimal — extending
        the set of consumed fields does not require touching SGLang code.
        """

        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights

        if num_tokens == 0:
            return self._make_empty_topk_output(topk_output)

        local_counts = self._count_local_routed(topk_ids)

        # Low-batch fast path: skip Waterfill rank selection entirely (matches
        # PR #19290 behavior). Dynamic still participates in EP all-reduce so
        # the collective ordering stays consistent across ranks.
        if num_tokens < self.MIN_BATCH_FOR_BALANCE:
            if not self.use_static_waterfill:
                self._all_reduce_dynamic_rank_load(local_counts, num_tokens)
            return self._materialize_local_shared(topk_ids, topk_weights)

        # Decide static vs dynamic mode and prepare canonical inputs.
        if self.use_static_waterfill:
            routed_rank_load = local_counts
            active_rank_token_count: Optional[Tensor] = None
        else:
            global_routed, local_tokens_per_rank = self._all_reduce_dynamic_rank_load(
                local_counts, num_tokens
            )
            routed_rank_load = global_routed
            active_rank_token_count = local_tokens_per_rank

        # Decompose forward_batch into the canonical RoutingRequest fields.
        stage = _resolve_stage_from_forward_batch(forward_batch)
        lb_token_mask = _resolve_lb_token_mask(forward_batch)
        token_count = _resolve_num_token_non_padded(forward_batch)

        request = RoutingRequest(
            layer_id=self.layer_id,
            logical_topk_ids=topk_ids,
            topk_weights=topk_weights,
            placement=None,
            stage=stage,
            routed_rank_load=routed_rank_load,
            active_rank_token_count=active_rank_token_count,
            token_count=token_count,
            lb_token_mask=lb_token_mask,
            policy=RoutingPolicyConfig(
                name="waterfill",
                enabled_stages=self.enabled_stages,
                metadata={"experts_per_rank": self.experts_per_rank},
            ),
        )

        decision = self.router.route_tokens(request)

        # Stage-gated skip: policy returned shared_expert_rank=None.
        if decision.shared_expert_rank is None:
            return self._materialize_local_shared(topk_ids, topk_weights)

        return self._materialize_with_chosen_rank(
            topk_ids=decision.routed_physical_topk_ids,
            topk_weights=decision.topk_weights,
            chosen_rank=decision.shared_expert_rank,
        )

    # -- Internal helpers --------------------------------------------------

    def _count_local_routed(self, topk_ids: Tensor) -> Tensor:
        if self._counts_buf is None:
            self._counts_buf = torch.zeros(
                self.world_size, dtype=torch.int64, device=topk_ids.device
            )
        return count_local_routed_per_rank(
            topk_ids,
            world_size=self.world_size,
            experts_per_rank=self.experts_per_rank,
            out=self._counts_buf,
        )

    @staticmethod
    def _all_reduce_dynamic_rank_load(
        local_routed_counts: Tensor, num_tokens: int
    ) -> Tuple[Tensor, Tensor]:
        """Aggregate dynamic load via the SGLang EP communicator.

        Returns ``(global_routed_counts, local_tokens_per_rank)``. The first
        is the EP-summed routed assignment count per rank (the "A" component
        in Waterfill's effective load). The second is each rank's own batch
        size, broadcast across all EP ranks (the "B" component).
        """
        from sglang.srt.distributed import get_moe_ep_group
        from sglang.srt.distributed.communication_op import (
            moe_expert_parallel_all_reduce,
        )

        group = get_moe_ep_group()
        world = group.world_size
        buf = torch.zeros(
            world * 2, dtype=torch.int64, device=local_routed_counts.device
        )
        buf[:world] = local_routed_counts
        rank = group.rank_in_group
        buf[world + rank : world + rank + 1].fill_(num_tokens)
        buf = moe_expert_parallel_all_reduce(buf)
        return buf[:world], buf[world:]

    def _make_empty_topk_output(self, topk_output):
        """Return an empty expanded StandardTopKOutput for zero-token batches."""
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
        device = topk_ids.device
        topk = topk_ids.shape[1]
        empty_ids = torch.empty(0, topk + 1, dtype=topk_ids.dtype, device=device)
        empty_weights = torch.empty(0, topk + 1, dtype=topk_weights.dtype, device=device)
        empty_logits = torch.empty(
            (0, topk + 1), dtype=torch.float32, device=device
        )
        return StandardTopKOutput(empty_weights, empty_ids, empty_logits)

    def _materialize_local_shared(self, topk_ids: Tensor, topk_weights: Tensor):
        """Low-batch / stage-gated path: expand TopK with shared expert pinned to local rank."""
        expanded_ids, expanded_weights = _expand_topk_local_shared(
            topk_ids,
            topk_weights,
            num_routed_experts=self.num_routed_experts,
            world_size=self.world_size,
            source_rank=self.source_rank,
            shared_weight=self.shared_weight,
        )
        return _build_topk_output(expanded_weights, expanded_ids)

    def _materialize_with_chosen_rank(
        self,
        *,
        topk_ids: Tensor,
        topk_weights: Tensor,
        chosen_rank: Tensor,
    ):
        """Materialize Waterfill's per-token rank decision into expanded TopK."""
        expanded_ids, expanded_weights = _expand_topk_with_chosen_rank(
            topk_ids,
            topk_weights,
            chosen_rank,
            num_routed_experts=self.num_routed_experts,
            world_size=self.world_size,
            source_rank=self.source_rank,
            shared_weight=self.shared_weight,
        )
        return _build_topk_output(expanded_weights, expanded_ids)


# -- Materializers (torch ops; torch.compile-friendly) ---------------------


def _expand_topk_local_shared(
    topk_ids: Tensor,
    topk_weights: Tensor,
    *,
    num_routed_experts: int,
    world_size: int,
    source_rank: int,
    shared_weight: float,
) -> Tuple[Tensor, Tensor]:
    """Expand TopK to width ``topk + 1`` pinning shared expert to local rank.

    Used for the low-batch fast path and the stage-gated bypass path.
    """
    return _expand_topk_with_chosen_rank(
        topk_ids,
        topk_weights,
        chosen_rank=torch.full(
            (topk_ids.shape[0],),
            source_rank,
            dtype=torch.int64,
            device=topk_ids.device,
        ),
        num_routed_experts=num_routed_experts,
        world_size=world_size,
        source_rank=source_rank,
        shared_weight=shared_weight,
    )


@triton.jit
def _expand_topk_with_chosen_rank_kernel(
    topk_ids_ptr,         # [N, topk] int (input)
    topk_weights_ptr,     # [N, topk] float (input)
    chosen_rank_ptr,      # [N] int64 (input)
    expanded_ids_ptr,     # [N, topk+1] int (output, same dtype as topk_ids)
    expanded_weights_ptr,  # [N, topk+1] float (output, same dtype as topk_weights)
    num_tokens,
    topk: tl.constexpr,
    old_epr,              # = num_routed_experts // world_size
    new_epr,              # = old_epr + 1
    shared_weight,        # float scalar
    local_shared_marker,  # int64 scalar (LOCAL_SHARED_MARKER = -1)
    BLOCK_SIZE: tl.constexpr,
):
    """Fused expansion of SGLang TopK output with a shared-expert column.

    Replaces the ~10-op PyTorch sequence (each op = one kernel launch) with a
    single Triton kernel:
      1. Remap routed expert IDs: ``new_id = old_id + (old_id // old_epr)``
         (shifts each rank's IDs up to leave room for the shared expert at
         the end of each rank's range).
      2. Compute ``has_valid`` per token from the routed columns.
      3. Write the shared-expert column: ``chosen_rank * new_epr + old_epr``
         (placed at the END of the chosen rank's range), or
         ``LOCAL_SHARED_MARKER`` for invalid tokens.
      4. Pass routed weights through, zero them for invalid slots, and write
         ``shared_weight`` (or 0) in the shared column.
    """
    pid = tl.program_id(0)
    token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_idx < num_tokens

    # Pass 1 (combined): write routed columns AND track has_valid.
    has_valid = tl.zeros([BLOCK_SIZE], dtype=tl.int1)
    for k in range(topk):
        e = tl.load(
            topk_ids_ptr + token_idx * topk + k, mask=mask, other=-1
        ).to(tl.int64)
        w = tl.load(
            topk_weights_ptr + token_idx * topk + k, mask=mask, other=0.0
        )
        valid = e >= 0
        has_valid = has_valid | valid
        # ID remap: new_id = old_id + (old_id // old_epr) for valid IDs.
        new_id = tl.where(valid, e + (e // old_epr), e)
        new_w = tl.where(valid, w, 0.0)
        out_idx = token_idx * (topk + 1) + k
        tl.store(expanded_ids_ptr + out_idx, new_id, mask=mask)
        tl.store(expanded_weights_ptr + out_idx, new_w, mask=mask)

    # Pass 2: write shared expert column (id + weight).
    chosen = tl.load(chosen_rank_ptr + token_idx, mask=mask, other=-1).to(tl.int64)
    valid_chosen = (chosen >= 0) & has_valid
    shared_id = tl.where(
        valid_chosen,
        chosen * new_epr + old_epr,
        tl.full([BLOCK_SIZE], local_shared_marker, dtype=tl.int64),
    )
    out_idx_shared = token_idx * (topk + 1) + topk
    tl.store(expanded_ids_ptr + out_idx_shared, shared_id, mask=mask)
    shared_w = tl.where(has_valid, shared_weight, 0.0)
    tl.store(expanded_weights_ptr + out_idx_shared, shared_w, mask=mask)


def _expand_topk_with_chosen_rank(
    topk_ids: Tensor,
    topk_weights: Tensor,
    chosen_rank: Tensor,
    *,
    num_routed_experts: int,
    world_size: int,
    source_rank: int,  # Kept for API compat; the materialization itself is
                       # symmetric in the rank dimension and does not need it.
    shared_weight: float,
) -> Tuple[Tensor, Tensor]:
    """Expand SGLang TopK output by appending a shared-expert column.

    Routed expert IDs are remapped from ``old_experts_per_rank`` namespace to
    ``new_experts_per_rank = old + 1`` namespace by adding ``id // old_epr``
    (one extra slot per rank reserved for the shared expert).

    The shared-expert column gets ``chosen_rank * new_epr + old_epr`` (the
    last slot on the chosen rank). Tokens with no valid routed expert
    (``has_valid == False``) are pinned to ``LOCAL_SHARED_MARKER = -1``,
    which DeepEP ignores during dispatch.

    Implemented as a single fused Triton kernel to replace the ~10-op PyTorch
    chain that previously dominated the per-layer overhead.
    """
    del source_rank  # unused; symmetric in the rank dimension.
    num_tokens = topk_ids.shape[0]
    topk = topk_ids.shape[1]
    device = topk_ids.device
    old_epr = num_routed_experts // world_size
    new_epr = old_epr + 1

    expanded_ids = torch.empty(
        num_tokens, topk + 1, dtype=topk_ids.dtype, device=device
    )
    expanded_weights = torch.empty(
        num_tokens, topk + 1, dtype=topk_weights.dtype, device=device
    )
    if num_tokens == 0:
        return expanded_ids, expanded_weights

    BLOCK_SIZE = 256
    grid = ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _expand_topk_with_chosen_rank_kernel[grid](
        topk_ids,
        topk_weights,
        chosen_rank,
        expanded_ids,
        expanded_weights,
        num_tokens,
        topk,
        old_epr,
        new_epr,
        shared_weight,
        LOCAL_SHARED_MARKER,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return expanded_ids, expanded_weights


# -- ForwardBatch decomposition (SGLang-specific) --------------------------


_FORWARD_MODE_TO_STAGE: dict[str, RoutingStage] = {
    "EXTEND": RoutingStage.PREFILL,
    "DECODE": RoutingStage.DECODE,
    "MIXED": RoutingStage.MIXED,
    "IDLE": RoutingStage.IDLE,
    "TARGET_VERIFY": RoutingStage.SPECULATIVE,
    "DRAFT_EXTEND": RoutingStage.SPECULATIVE,
}


def _resolve_stage_from_forward_batch(forward_batch) -> Optional[RoutingStage]:
    """Map SGLang ``forward_batch.forward_mode`` to a ``RoutingStage`` enum.

    Returns ``None`` if ``forward_batch`` is ``None`` or the forward mode is
    unrecognized — policy stage gating treats ``None`` as "stage-agnostic
    eligibility" or "stage required" depending on
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
    """Pull an optional load-balancer-eligible token mask from ``forward_batch``.

    Currently SGLang's ``ForwardBatch`` does not expose a stable load-balance
    mask field; this returns ``None`` for now. A future SGLang change can
    surface a mask attribute (e.g. ``forward_batch.lb_token_mask``) that this
    helper would forward without further plumbing.
    """
    if forward_batch is None:
        return None
    return getattr(forward_batch, "lb_token_mask", None)


def _resolve_num_token_non_padded(forward_batch) -> Optional[Tensor]:
    """Pull SGLang's non-padded token count tensor."""
    if forward_batch is None:
        return None
    return getattr(forward_batch, "num_token_non_padded", None)


# -- StandardTopKOutput plumbing ------------------------------------------


def _build_topk_output(expanded_weights: Tensor, expanded_ids: Tensor):
    """Assemble a SGLang ``StandardTopKOutput`` from expanded tensors.

    SGLang's ``StandardTopKOutput`` carries (topk_weights, topk_ids,
    router_logits). The router_logits field is informational and we cannot
    reliably reconstruct a meaningful one from the expanded layout, so we
    populate a zero placeholder of the appropriate shape; downstream code
    in PR #19290 does not consume it for the Waterfill path.
    """
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    num_tokens, topk_plus_one = expanded_ids.shape
    router_logits = torch.zeros(
        (num_tokens, topk_plus_one), dtype=torch.float32, device=expanded_ids.device
    )
    return StandardTopKOutput(expanded_weights, expanded_ids, router_logits)
