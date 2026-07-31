from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from moe_load_balancer.core.types import (
    PlacementSnapshot,
    RoutingDecision,
    RoutingPolicyConfig,
    RoutingRequest,
    RoutingStage,
    TensorLike,
)


@dataclass(frozen=True)
class SGLangRoutingOutput:
    """Plain tensors that SGLang Glue writes into ``StandardTopKOutput``."""

    topk_weights: TensorLike
    topk_ids: TensorLike
    router_logits: Optional[TensorLike]
    recorded_physical_topk_ids: TensorLike


def to_routing_request(
    *,
    layer_id: int,
    logical_topk_ids: TensorLike,
    topk_weights: TensorLike,
    policies: Sequence[RoutingPolicyConfig | str],
    placement: Optional[PlacementSnapshot] = None,
    routed_physical_topk_ids: Optional[TensorLike] = None,
    stage: Optional[RoutingStage | str] = None,
    routed_rank_load: Optional[TensorLike] = None,
    active_rank_token_count: Optional[TensorLike] = None,
    global_logical_count: Optional[TensorLike] = None,
    token_count: Optional[TensorLike] = None,
    lb_token_mask: Optional[TensorLike] = None,
    routed_scaling_factor: float = 1.0,
) -> RoutingRequest:
    """Translate SGLang TopK and prepared framework statistics into MLB input.

    Communication is intentionally absent: SGLang Glue supplies any
    all-reduced ``global_logical_count`` or dynamic Waterfill load tensors.
    """

    normalized_stage = None if stage is None else RoutingStage.from_str(stage)
    return RoutingRequest(
        layer_id=layer_id,
        logical_topk_ids=logical_topk_ids,
        topk_weights=topk_weights,
        policies=tuple(policies),
        placement=placement,
        routed_physical_topk_ids=routed_physical_topk_ids,
        stage=normalized_stage,
        routed_rank_load=routed_rank_load,
        active_rank_token_count=active_rank_token_count,
        global_logical_count=global_logical_count,
        token_count=token_count,
        lb_token_mask=lb_token_mask,
        routed_scaling_factor=routed_scaling_factor,
    )


def to_sglang_routing_output(
    decision: RoutingDecision,
    *,
    router_logits: Optional[TensorLike] = None,
    num_routed_experts: Optional[int] = None,
    world_size: Optional[int] = None,
    routed_scaling_factor: float = 1.0,
) -> SGLangRoutingOutput:
    """Translate an MLB decision into the tensors consumed by SGLang Glue.

    The unexpanded physical IDs are returned separately for SGLang's recorder.
    If Waterfill selected a shared rank, this function appends the shared
    expert and converts routed IDs into DeepEP's interleaved ID layout.
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

    old_experts_per_rank = num_routed_experts // world_size
    new_experts_per_rank = old_experts_per_rank + 1
    valid = physical_ids >= 0
    interleaved_ids = torch.where(
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
    shared_weight = 1.0 / routed_scaling_factor if routed_scaling_factor != 0 else 1.0
    shared_weights = torch.where(
        has_valid,
        torch.full(
            has_valid.shape,
            shared_weight,
            dtype=weights.dtype,
            device=weights.device,
        ),
        torch.zeros(has_valid.shape, dtype=weights.dtype, device=weights.device),
    )

    expanded_ids = torch.cat((interleaved_ids, shared_ids[:, None]), dim=1)
    expanded_weights = torch.cat((routed_weights, shared_weights[:, None]), dim=1)

    expanded_logits = router_logits
    if isinstance(router_logits, torch.Tensor):
        zeros = torch.zeros(
            (router_logits.shape[0], 1),
            dtype=router_logits.dtype,
            device=router_logits.device,
        )
        expanded_logits = torch.cat((router_logits, zeros), dim=1)

    return SGLangRoutingOutput(
        topk_weights=expanded_weights,
        topk_ids=expanded_ids,
        router_logits=expanded_logits,
        recorded_physical_topk_ids=physical_ids,
    )
