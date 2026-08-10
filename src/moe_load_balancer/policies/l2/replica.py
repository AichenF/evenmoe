"""Baseline and experimental physical-replica routing policies."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from moe_load_balancer.core.l2 import TokenRouter
from moe_load_balancer.core.types import (
    RoutingDecision,
    RoutingPhase,
    RoutingPolicyConfig,
    RoutingRequest,
)


def transform_fake_routing_inputs(
    router_logits: Tensor,
    correction_bias: Optional[Tensor],
) -> tuple[Tensor, Optional[Tensor]]:
    """Create the uniform random routing stimulus used by SGLang fake mode."""

    if not isinstance(router_logits, Tensor):
        raise TypeError("Fake routing requires torch.Tensor router_logits.")
    router_logits.uniform_(5, 10)
    if correction_bias is not None:
        if not isinstance(correction_bias, Tensor):
            raise TypeError("Fake routing correction_bias must be a torch.Tensor or None.")
        correction_bias = torch.zeros_like(correction_bias)
    return router_logits, correction_bias


def transform_topk_inputs(
    transform_name: Optional[str],
    router_logits: Tensor,
    correction_bias: Optional[Tensor],
) -> tuple[Tensor, Optional[Tensor]]:
    """Apply an optional experimental transform before framework TopK."""

    if transform_name == "fake":
        return transform_fake_routing_inputs(router_logits, correction_bias)
    if transform_name is None:
        return router_logits, correction_bias
    raise ValueError(f"Unknown pre-TopK transform {transform_name!r}.")


def route_static_replicas(
    logical_topk_ids: Tensor,
    default_physical_for_logical: Tensor,
) -> Tensor:
    """Map every logical expert to the framework-selected default replica."""

    if not isinstance(logical_topk_ids, Tensor):
        raise TypeError("Replica routing requires torch.Tensor logical_topk_ids.")
    if not isinstance(default_physical_for_logical, Tensor):
        raise TypeError("Static replica routing requires a torch.Tensor default map.")
    safe_ids, valid = _safe_logical_ids(
        logical_topk_ids,
        default_physical_for_logical.shape[0],
    )
    physical_ids = default_physical_for_logical[safe_ids]
    return torch.where(valid, physical_ids, logical_topk_ids)


def route_dynamic_replicas(
    logical_topk_ids: Tensor,
    logical_to_physical_candidates: Tensor,
    logical_to_physical_count: Tensor,
) -> Tensor:
    """Randomly choose one committed physical replica for every logical ID."""

    if not isinstance(logical_topk_ids, Tensor):
        raise TypeError("Replica routing requires torch.Tensor logical_topk_ids.")
    if not isinstance(logical_to_physical_candidates, Tensor) or not isinstance(
        logical_to_physical_count, Tensor
    ):
        raise TypeError("Dynamic replica routing requires torch.Tensor placement maps.")

    safe_ids, valid = _safe_logical_ids(
        logical_topk_ids,
        logical_to_physical_count.shape[0],
    )
    chosen_dispatch_index = (
        torch.randint(
            0,
            65536,
            logical_topk_ids.shape,
            dtype=torch.int32,
            device=logical_topk_ids.device,
        )
        % logical_to_physical_count[safe_ids]
    )
    physical_ids = logical_to_physical_candidates[safe_ids, chosen_dispatch_index]
    return torch.where(valid, physical_ids, logical_topk_ids)


def route_replicas(
    policy_name: Optional[str],
    logical_topk_ids: Tensor,
    *,
    default_physical_for_logical: Optional[Tensor],
    logical_to_physical_candidates: Tensor,
    logical_to_physical_count: Tensor,
) -> Tensor:
    """Route replica IDs through one MLB-selected inline policy."""

    if policy_name is None:
        return logical_topk_ids
    if policy_name == "static":
        if default_physical_for_logical is None:
            raise ValueError("Static replica routing requires a default physical map.")
        return route_static_replicas(logical_topk_ids, default_physical_for_logical)
    if policy_name == "dynamic":
        return route_dynamic_replicas(
            logical_topk_ids,
            logical_to_physical_candidates,
            logical_to_physical_count,
        )
    raise ValueError(f"Unknown inline replica-routing policy {policy_name!r}.")


def _safe_logical_ids(
    logical_topk_ids: Tensor,
    num_logical_experts: int,
) -> tuple[Tensor, Tensor]:
    if num_logical_experts <= 0:
        raise ValueError("Replica routing requires at least one logical expert.")
    valid = (logical_topk_ids >= 0) & (logical_topk_ids < num_logical_experts)
    safe_ids = logical_topk_ids.clamp(
        min=0,
        max=num_logical_experts - 1,
    ).to(torch.int64)
    return safe_ids, valid


class StaticL2Router(TokenRouter):
    """Deterministic routing through the framework-selected replica map."""

    phase = RoutingPhase.PHYSICAL_ROUTING

    def route_tokens(
        self,
        request: RoutingRequest,
        policy: RoutingPolicyConfig,
    ) -> RoutingDecision:
        if request.placement is None:
            raise ValueError("Static replica routing requires a committed placement.")
        default_map = request.placement.default_physical_for_logical
        if default_map is None:
            raise ValueError(
                "Static replica routing requires " "PlacementSnapshot.default_physical_for_logical."
            )
        physical_ids = route_static_replicas(
            request.logical_topk_ids,
            default_map,
        )
        return RoutingDecision(
            routed_physical_topk_ids=physical_ids,
            topk_weights=request.topk_weights,
            metadata={"policy": policy.name},
        )


class DynamicL2Router(TokenRouter):
    """Random routing among the committed replicas of each logical expert."""

    phase = RoutingPhase.PHYSICAL_ROUTING

    def route_tokens(
        self,
        request: RoutingRequest,
        policy: RoutingPolicyConfig,
    ) -> RoutingDecision:
        if request.placement is None:
            raise ValueError("Dynamic replica routing requires a committed placement.")
        physical_ids = route_dynamic_replicas(
            request.logical_topk_ids,
            request.placement.logical_to_physical_candidates,
            request.placement.logical_to_physical_count,
        )
        return RoutingDecision(
            routed_physical_topk_ids=physical_ids,
            topk_weights=request.topk_weights,
            metadata={"policy": policy.name},
        )
