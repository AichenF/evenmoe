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


def count_logical_experts(
    logical_topk_ids: TensorLike,
    num_logical_experts: int,
) -> TensorLike:
    """Translate SGLang TopK IDs into a plain logical-count tensor.

    SGLang DP-attention can retain ``-1`` padding on otherwise non-empty
    ranks. CUDA handles that in one fused kernel instead of materializing a
    validity mask, clamped IDs, and scatter contributions separately.
    Communication remains owned by SGLang Glue.
    """

    if num_logical_experts <= 0:
        raise ValueError("num_logical_experts must be positive.")

    import torch

    if not isinstance(logical_topk_ids, torch.Tensor):
        raise TypeError("Logical expert counting requires a torch.Tensor.")
    counts = torch.zeros(
        num_logical_experts,
        dtype=torch.float32,
        device=logical_topk_ids.device,
    )
    if logical_topk_ids.is_cuda:
        from moe_load_balancer.adapters.sglang._logical_count_cuda import (
            count_logical_experts_cuda,
        )

        count_logical_experts_cuda(logical_topk_ids, counts)
        return counts

    valid = (logical_topk_ids >= 0) & (logical_topk_ids < num_logical_experts)
    if valid.any():
        counts.copy_(
            torch.bincount(
                logical_topk_ids[valid].to(torch.int64),
                minlength=num_logical_experts,
            ).to(torch.float32)
        )
    return counts


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

    from moe_load_balancer.adapters.sglang.waterfill import (
        materialize_waterfill_output,
    )

    expanded_ids, expanded_weights = materialize_waterfill_output(
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
