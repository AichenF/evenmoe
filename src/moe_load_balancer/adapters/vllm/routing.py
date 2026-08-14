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

    # Rank preference: local replicas first, then any valid one, in slot order.
    #   local valid -> 0, non-local valid -> 1, invalid -> 2
    priority = torch.where(
        is_local,
        torch.zeros_like(candidates),
        torch.where(valid, torch.ones_like(candidates), torch.full_like(candidates, 2)),
    )
    key = priority * (max_replicas + 1) + slot_index.unsqueeze(0)
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
    default_physical_for_logical: Optional[TensorLike] = None,
) -> PlacementSnapshot:
    """Translate vLLM's per-layer ``EplbLayerState`` into an L2 snapshot.

    ``EplbLayerState`` already holds per-layer views of
    ``logical_to_physical_map`` and ``logical_replica_count``
    (``set_layer_state`` slices them by ``moe_layer_idx``), so those map onto
    MLB's candidate/count pair with no reshaping.

    ``physical_to_logical_map`` must be supplied by Glue: vLLM keeps it on the
    model-level ``EplbModelState``, not on the layer state.

    IMPORTANT semantic note.  vLLM's ``compute_logical_maps`` enumerates *all*
    physical replicas of a logical expert in ascending slot order.  SGLang's
    ``logical_to_all_physical_map`` collapses to the nearest local replica when
    one exists, so an L2 policy tuned against SGLang sees a strictly smaller
    action space than it will see here.  Replica-choice quality measured on
    SGLang does not transfer; re-measure on vLLM.
    """

    candidates = getattr(layer_state, "logical_to_physical_map", None)
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
        # vLLM has no precomputed per-rank dispatch table (SGLang's
        # logical_to_rank_dispatch_physical_map); its baseline picks a replica
        # with a Knuth hash of the token index instead.  Glue may synthesize
        # one with :func:`nearest_replica_table`; policies that require it
        # (``static``) fail loudly when it is absent rather than silently
        # routing somewhere else.
        default_physical_for_logical=default_physical_for_logical,
        physical_to_rank_map=None,
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
) -> RoutingRequest:
    """Translate vLLM's logical TopK and live placement into MLB input.

    ``defer_dispatch`` asks for the replica-share table instead of per-token
    ids. vLLM fuses replica selection with expert-load recording in one kernel,
    so applying the split itself is how it keeps the recording -- see
    ``RoutingRequest.defer_dispatch``.
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
