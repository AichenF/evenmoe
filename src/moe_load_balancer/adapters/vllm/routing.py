from __future__ import annotations

from typing import Optional, Sequence

from moe_load_balancer.core.types import (
    PlacementSnapshot,
    RoutingDecision,
    RoutingPolicyConfig,
    RoutingRequest,
    RoutingStage,
    TensorLike,
)

# vLLM ForwardMode -> MLB RoutingStage.  vLLM does not distinguish a dedicated
# "idle" routing mode the way SGLang does; dummy/profile runs reuse DECODE.
_VLLM_MODE_TO_STAGE = {
    "prefill": RoutingStage.PREFILL,
    "decode": RoutingStage.DECODE,
    "mixed": RoutingStage.MIXED,
    "spec": RoutingStage.SPECULATIVE,
}


def collapse_candidates_to_local(
    candidates: TensorLike,
    counts: TensorLike,
    *,
    ep_rank: int,
    num_local_physical_experts: int,
) -> tuple[TensorLike, TensorLike]:
    """Collapse the candidate list to the nearest local replica.

    For each logical expert with at least one replica on this EP rank, the
    candidate list is replaced by just that replica; experts with no local
    replica keep their full (remote) list.

    **This is for rank-dispatch policies only** -- ``static`` and anything else
    that reports ``requires_rank_dispatch_map``. Do not apply it to a
    replica-routing policy such as LPLB, whose entire job is to decide how a
    replicated expert's traffic splits: leaving one candidate answers that
    question in advance, and a rank that holds a copy of every expert it sees is
    left with no redundancy at all, so the LP is skipped outright.

    SGLang draws the same line. The collapse there lives in
    ``compute_logical_to_rank_dispatch_physical_map``, built only when the
    pipeline reports ``requires_rank_dispatch_map``, and in the
    ``init_by_mapping`` / ``init_trivial`` startup paths. ``init_by_eplb`` --
    taken on every rebalance, and the configuration its published EPLB numbers
    were measured in -- passes the policy the full global list.

    Args:
        candidates: ``[num_logical, max_replicas]`` physical slot ids, -1 where
            unused.
        counts:     ``[num_logical]`` number of valid replicas per expert.
        ep_rank:    this rank's index in the EP group.
        num_local_physical_experts: number of physical slots owned by each rank.

    Returns:
        (collapsed_candidates, collapsed_counts) with the same shapes.
    """
    import torch

    if not isinstance(candidates, torch.Tensor) or not isinstance(counts, torch.Tensor):
        raise TypeError("collapse_candidates_to_local requires torch.Tensor inputs.")

    low = ep_rank * num_local_physical_experts
    high = low + num_local_physical_experts

    is_local = (candidates >= low) & (candidates < high) & (candidates >= 0)
    has_local = is_local.any(dim=-1)  # [num_logical]

    # For experts with a local copy, find the first (lowest-slot) local replica.
    # The sentinel only has to lose a `min`, so take it from the dtype rather
    # than from the data: reading `candidates.max()` would sync the device on
    # every call, and callers may run this per layer inside a forward pass.
    sentinel = torch.iinfo(candidates.dtype).max
    local_only_vals = torch.where(is_local, candidates, torch.full_like(candidates, sentinel))
    first_local = local_only_vals.min(dim=-1).values  # [num_logical]

    # Build collapsed tensor: keep original for non-local experts, replace for local.
    local_row = torch.full_like(candidates, -1)
    local_row[:, 0] = first_local  # [local_id, -1, -1, ...]

    collapsed = torch.where(has_local.unsqueeze(-1), local_row, candidates)
    collapsed_counts = torch.where(has_local, torch.ones_like(counts), counts)
    return collapsed, collapsed_counts


def build_physical_to_rank_map(
    num_physical_experts: int,
    ep_size: int,
    device: object,
) -> TensorLike:
    """Build a [num_physical_experts] tensor mapping each slot to its EP rank.

    SGLang passes this as ``PlacementSnapshot.physical_to_rank_map``, letting
    LPLB incorporate cross-GPU transfer cost into its placement decisions.
    """
    import torch

    num_local = num_physical_experts // ep_size
    return torch.arange(num_physical_experts, device=device) // num_local


def nearest_replica_table(
    candidates: TensorLike,
    counts: TensorLike,
    *,
    ep_rank: int,
    num_local_physical_experts: int,
) -> TensorLike:
    """Build a per-rank "nearest replica" table from vLLM's global candidates.

    SGLang ships such a table (``logical_to_rank_dispatch_physical_map``) and
    additionally collapses its candidate list to the local replica when one
    exists.  vLLM has neither: ``compute_logical_maps`` enumerates every
    replica globally and the built-in kernel picks one by hashing the token
    index.

    Policies that need a deterministic default -- ``static`` most obviously --
    therefore have nothing to read on vLLM unless it is synthesized.  This
    reproduces SGLang's intent: prefer a replica owned by this rank, else fall
    back to the first valid replica.

    Returns ``[num_logical_experts]`` of physical slot ids.
    """

    import torch

    if not isinstance(candidates, torch.Tensor) or not isinstance(counts, torch.Tensor):
        raise TypeError("nearest_replica_table requires torch.Tensor inputs.")

    max_replicas = candidates.shape[-1]
    slot_index = torch.arange(max_replicas, device=candidates.device)
    valid = slot_index.unsqueeze(0) < counts.unsqueeze(-1).to(slot_index.dtype)
    valid = valid & (candidates >= 0)

    low = ep_rank * num_local_physical_experts
    high = low + num_local_physical_experts
    is_local = valid & (candidates >= low) & (candidates < high)

    # Rank preference: a local replica first, then any valid one.
    #   local valid -> 0, non-local valid -> 1, invalid -> 2
    priority = torch.where(
        is_local,
        torch.zeros_like(candidates),
        torch.where(valid, torch.ones_like(candidates), torch.full_like(candidates, 2)),
    )

    # Among *non-local* replicas the tie is broken by a rank-rotated slot order
    # rather than by slot 0. With 8 EP ranks and 2 replicas only two ranks hold
    # a copy; if the other six all fall back to the lowest slot they pile onto
    # one replica, which concentrates the very traffic this table exists to
    # place well -- worse than the uniform choice it replaces. Rotating spreads
    # them, and leaves the local case untouched.
    rotated = (slot_index.unsqueeze(0) - ep_rank) % counts.clamp(min=1).unsqueeze(-1).to(
        slot_index.dtype
    )
    tiebreak = torch.where(is_local, slot_index.unsqueeze(0), rotated)
    key = priority * (max_replicas + 1) + tiebreak
    chosen = key.argmin(dim=-1, keepdim=True)
    return candidates.gather(-1, chosen).squeeze(-1)


def to_placement_snapshot(
    layer_state: object,
    layer_id: int,
    *,
    physical_to_logical_map: TensorLike,
    num_logical_experts: int,
    num_physical_experts: int,
    ep_size: int,
    candidates: Optional[TensorLike] = None,
    counts: Optional[TensorLike] = None,
    default_physical_for_logical: Optional[TensorLike] = None,
    physical_to_rank_map: Optional[TensorLike] = None,
) -> PlacementSnapshot:
    """Translate vLLM's per-layer ``EplbLayerState`` into an L2 snapshot.

    ``candidates`` and ``counts`` are the logical-to-physical candidate map and
    replica-count tensor to pass to MLB.  If omitted, they are read from
    ``layer_state.logical_to_physical_map`` and
    ``layer_state.logical_replica_count`` (the raw global-all-replicas lists
    that vLLM keeps).

    Callers that want SGLang-compatible behaviour should pre-collapse the
    candidate list with :func:`collapse_candidates_to_local` and pass the
    result here.  Without collapsing, LPLB sees every global replica and may
    route cross-GPU even when a local copy exists.

    ``physical_to_logical_map`` must be supplied by Glue: vLLM keeps it on the
    model-level ``EplbModelState``, not on the layer state.

    ``physical_to_rank_map`` maps each physical slot to its EP rank; SGLang
    always provides this so LPLB can incorporate cross-GPU transfer cost.
    Build it with :func:`build_physical_to_rank_map`.
    """

    if candidates is None:
        candidates = getattr(layer_state, "logical_to_physical_map", None)
    if counts is None:
        counts = getattr(layer_state, "logical_replica_count", None)
    if candidates is None or counts is None:
        raise ValueError(
            "vLLM L2 routing requires a layer state with logical_to_physical_map "
            "and logical_replica_count; EPLB is probably not enabled."
        )
    if num_physical_experts % ep_size != 0:
        raise ValueError("vLLM placement requires num_physical_experts to be divisible by ep_size.")

    return PlacementSnapshot(
        layer_id=layer_id,
        num_logical_experts=num_logical_experts,
        num_physical_experts=num_physical_experts,
        ep_size=ep_size,
        num_local_physical_experts=num_physical_experts // ep_size,
        physical_to_logical_map=physical_to_logical_map,
        logical_to_physical_candidates=candidates,
        logical_to_physical_count=counts,
        default_physical_for_logical=default_physical_for_logical,
        physical_to_rank_map=physical_to_rank_map,
    )


def to_routing_request(
    *,
    layer_id: int,
    logical_topk_ids: TensorLike,
    topk_weights: TensorLike,
    policies: Sequence[RoutingPolicyConfig | str] = (),
    placement: Optional[PlacementSnapshot] = None,
    stage: Optional[RoutingStage | str] = None,
    token_count: Optional[TensorLike] = None,
    routed_scaling_factor: float = 1.0,
    defer_dispatch: bool = False,
    global_logical_count: Optional[TensorLike] = None,
    expert_load_view: Optional[TensorLike] = None,
    record_enabled: Optional[TensorLike] = None,
    num_unpadded_tokens: Optional[TensorLike] = None,
) -> RoutingRequest:
    """Translate vLLM's logical TopK and live placement into MLB input.

    ``defer_dispatch`` asks for the replica-share table instead of per-token
    ids. vLLM fuses replica selection with expert-load recording in one kernel,
    so applying the split itself is how it keeps the recording -- see
    ``RoutingRequest.defer_dispatch``.

    ``global_logical_count`` is the pre-computed, all-reduced per-logical-expert
    token count for this layer.  When provided, MLB skips its internal
    count_logical_experts + EP all_reduce (which are not CUDA-graph-safe).
    vLLM computes counts inside the graph and all_reduces them outside, then
    passes the result here as a stable tensor so LP solve is fully capturable.

    ``expert_load_view`` / ``record_enabled`` / ``num_unpadded_tokens`` offer
    the framework's load accounting to the policy. vLLM fuses selection with
    recording, so a policy that resolves ids leaves the recording behind as a
    second launch over the same elements unless it takes the view and counts
    while it is already there.
    """
    normalized_stage = None if stage is None else RoutingStage.from_str(stage)
    return RoutingRequest(
        layer_id=layer_id,
        logical_topk_ids=logical_topk_ids,
        topk_weights=topk_weights,
        policies=tuple(policies),
        placement=placement,
        stage=normalized_stage,
        token_count=token_count,
        routed_scaling_factor=routed_scaling_factor,
        defer_dispatch=defer_dispatch,
        global_logical_count=global_logical_count,
        expert_load_view=expert_load_view,
        record_enabled=record_enabled,
        num_unpadded_tokens=num_unpadded_tokens,
    )


def to_vllm_topk_ids(decision: RoutingDecision) -> TensorLike:
    """Translate an MLB decision into the tensor vLLM's router writes back.

    vLLM's routing boundary (``BaseRouter._apply_eplb_mapping``) returns only
    physical ``topk_ids``; weights are produced upstream and are not rewritten.

    A shared-expert decision cannot be expressed here.  On vLLM's CUDA path the
    shared expert is a per-rank replicated MLP that never enters EP dispatch
    (``fused_moe/layer.py`` gates shared-expert fusion behind the AITER env),
    so there is no "which rank runs it" choice to write back.  Any policy that
    produces one is a configuration error rather than something to silently
    drop.
    """

    if decision.shared_expert_rank is not None:
        raise NotImplementedError(
            "Shared-expert routing (Waterfill) has no writeback path on vLLM: "
            "shared experts are replicated per rank and do not go through EP "
            "dispatch. Select a routed-replica policy such as 'lplb', "
            "'dynamic' or 'static'."
        )
    return decision.routed_physical_topk_ids


def stage_from_vllm(name: Optional[str]) -> Optional[RoutingStage]:
    """Map a vLLM-side stage label onto MLB's stage enum."""

    if name is None:
        return None
    return _VLLM_MODE_TO_STAGE.get(name.lower())
