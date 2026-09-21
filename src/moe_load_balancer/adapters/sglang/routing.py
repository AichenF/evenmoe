from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

from moe_load_balancer.adapters.sglang.placement import _placement, to_placement_snapshot

from moe_load_balancer.core.types import (
    PlacementSnapshot,
    RoutingDecision,
    RoutingPolicyConfig,
    RoutingRequest,
    RoutingStage,
    TensorLike,
)

if TYPE_CHECKING:
    from sglang.srt.runtime_context import RuntimeContext


@dataclass(frozen=True)
class SGLangRoutingOutput:
    """Plain tensors that SGLang Glue writes into ``StandardTopKOutput``."""

    topk_weights: TensorLike
    topk_ids: TensorLike
    router_logits: Optional[TensorLike]
    recorded_physical_topk_ids: TensorLike


def count_logical_experts(
    logical_topk_ids: TensorLike,
    num_logical_experts: int,
    dtype: Optional[object] = None,
) -> TensorLike:
    """Compatibility export for the MLB-owned logical count helper."""

    from moe_load_balancer.kernels.expert_count import count_logical_experts as count

    return count(logical_topk_ids, num_logical_experts, dtype=dtype)


def to_routing_request(
    *,
    context: Optional[RuntimeContext] = None,
    layer_id: int,
    logical_topk_ids: TensorLike,
    topk_weights: TensorLike,
    policies: Sequence[RoutingPolicyConfig | str] = (),
    placement: Optional[PlacementSnapshot] = None,
    stage: Optional[RoutingStage | str] = None,
    token_count: Optional[TensorLike] = None,
    routed_scaling_factor: float = 1.0,
) -> RoutingRequest:
    """Translate TopK and placement, reading context when no snapshot is supplied."""

    if placement is None and context is not None:
        placement = to_placement_snapshot(_placement(context), layer_id)
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
    )


def to_sglang_routing_output(
    decision: RoutingDecision,
    *,
    context: Optional[RuntimeContext] = None,
    router_logits: Optional[TensorLike] = None,
    num_routed_experts: Optional[int] = None,
    world_size: Optional[int] = None,
    routed_scaling_factor: float = 1.0,
) -> SGLangRoutingOutput:
    """Translate an MLB decision into the tensors consumed by SGLang Glue.

    The unexpanded physical IDs are returned separately for SGLang's recorder.
    If an L2 policy selected a shared rank, this function appends the shared
    expert and converts routed IDs into DeepEP's interleaved ID layout.
    Context supplies geometry unless explicit counts are provided.
    """

    physical_ids = decision.routed_physical_topk_ids
    weights = decision.topk_weights
    if decision.shared_expert_rank is None:
        return SGLangRoutingOutput(
            topk_weights=weights,
            topk_ids=physical_ids,
            router_logits=router_logits,
            recorded_physical_topk_ids=physical_ids,
        )

    if context is not None and (num_routed_experts is None or world_size is None):
        metadata = _placement(context)
        if num_routed_experts is None:
            num_routed_experts = metadata.num_physical_experts
        if world_size is None:
            world_size = metadata.ep_size
    if num_routed_experts is None or world_size is None:
        raise ValueError(
            "Shared-expert materialization requires num_routed_experts and world_size."
        )
    if num_routed_experts <= 0 or world_size <= 0:
        raise ValueError("num_routed_experts and world_size must be positive.")
    if num_routed_experts % world_size != 0:
        raise ValueError("num_routed_experts must be divisible by world_size.")

    import torch

    if not isinstance(physical_ids, torch.Tensor) or not isinstance(weights, torch.Tensor):
        raise TypeError("SGLang routing materialization requires torch.Tensor inputs.")
    chosen_rank = decision.shared_expert_rank
    if not isinstance(chosen_rank, torch.Tensor):
        chosen_rank = torch.as_tensor(chosen_rank, device=physical_ids.device)

    expanded_ids, expanded_weights = _materialize_shared_expert_output(
        physical_ids,
        weights,
        chosen_rank,
        num_routed_experts=num_routed_experts,
        world_size=world_size,
        routed_scaling_factor=routed_scaling_factor,
    )

    return SGLangRoutingOutput(
        topk_weights=expanded_weights,
        topk_ids=expanded_ids,
        # Router logits are indexed by logical routed expert and are not part
        # of DeepEP's physical TopK layout.  Expanding this [tokens, experts]
        # tensor was unnecessary and copied O(tokens * num_experts) data.
        router_logits=router_logits,
        recorded_physical_topk_ids=physical_ids,
    )


def _materialize_shared_expert_output(
    physical_ids: TensorLike,
    weights: TensorLike,
    chosen_rank: TensorLike,
    *,
    num_routed_experts: int,
    world_size: int,
    routed_scaling_factor: float,
) -> tuple[TensorLike, TensorLike]:
    """Translate a shared-rank decision into SGLang's DeepEP TopK layout."""

    import torch

    if physical_ids.ndim != 2 or weights.shape != physical_ids.shape:
        raise ValueError("physical_ids and weights must have the same 2-D shape.")
    if chosen_rank.shape != (physical_ids.shape[0],):
        raise ValueError("chosen_rank must have one entry per token.")

    old_experts_per_rank = num_routed_experts // world_size
    new_experts_per_rank = old_experts_per_rank + 1
    shared_weight = 1.0 / routed_scaling_factor if routed_scaling_factor != 0 else 1.0

    if not physical_ids.is_cuda:
        valid = physical_ids >= 0
        interleaved = torch.where(
            valid,
            physical_ids + physical_ids // old_experts_per_rank,
            physical_ids,
        )
        routed_weights = torch.where(valid, weights, torch.zeros_like(weights))
        has_valid = valid.any(dim=1)
        shared_ids = chosen_rank.to(physical_ids.dtype) * new_experts_per_rank
        shared_ids = shared_ids + old_experts_per_rank
        shared_ids = torch.where(
            has_valid & (chosen_rank >= 0),
            shared_ids,
            torch.full_like(shared_ids, -1),
        )
        shared_weights = torch.where(
            has_valid,
            torch.full_like(has_valid, shared_weight, dtype=weights.dtype),
            torch.zeros_like(has_valid, dtype=weights.dtype),
        )
        return (
            torch.cat((interleaved, shared_ids[:, None]), dim=1),
            torch.cat((routed_weights, shared_weights[:, None]), dim=1),
        )

    num_tokens, topk = physical_ids.shape
    expanded_ids = torch.empty(
        (num_tokens, topk + 1),
        dtype=physical_ids.dtype,
        device=physical_ids.device,
    )
    expanded_weights = torch.empty(
        (num_tokens, topk + 1),
        dtype=weights.dtype,
        device=weights.device,
    )
    if num_tokens == 0:
        return expanded_ids, expanded_weights

    from moe_load_balancer.kernels.shared_expert_materialization import (
        materialize_shared_expert_cuda,
    )

    materialize_shared_expert_cuda(
        physical_ids,
        weights,
        chosen_rank,
        expanded_ids,
        expanded_weights,
        old_experts_per_rank,
        new_experts_per_rank,
        shared_weight,
    )
    return expanded_ids, expanded_weights
