"""Baseline and experimental physical-replica routing policies."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from moe_load_balancer.core.l2 import TokenRouter
from moe_load_balancer.kernels.token_draw import replica_index
from moe_load_balancer.core.types import (
    ReplicaSelection,
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
    # From the token index, not an RNG: the choice has to be reproducible and
    # agreed upon by every rank that sees the token. See kernels.token_draw.
    chosen_dispatch_index = replica_index(
        logical_topk_ids.shape,
        logical_topk_ids.shape[-1],
        logical_to_physical_count[safe_ids],
        logical_topk_ids.device,
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



def one_hot_replica_shares(
    default_physical_for_logical: Tensor,
    candidates: Tensor,
    counts: Tensor,
) -> Tensor:
    """Share table putting all of each expert's traffic on its default replica.

    Static routing already decides a per-expert replica; a share table is the
    same decision written in the layout a caller can apply itself. Padding
    columns stay zero so a consumer can walk a row without a separate count.
    """

    shares = torch.zeros(candidates.shape, dtype=torch.float32, device=candidates.device)
    slot = torch.arange(candidates.shape[1], device=candidates.device)
    live = slot.unsqueeze(0) < counts.unsqueeze(-1).to(slot.dtype)
    chosen = live & (candidates == default_physical_for_logical.unsqueeze(-1))
    shares[chosen] = 1.0
    # An expert whose default is absent from its candidate list would otherwise
    # get an all-zero row, which a consumer reads as "no preference" and fills
    # with its own choice. Fall back to the first live column so the policy's
    # answer stays deterministic.
    missing = ~chosen.any(dim=-1) & (counts > 0)
    if bool(missing.any()):
        shares[missing, 0] = 1.0
    return shares


def uniform_replica_shares(candidates: Tensor, counts: Tensor) -> Tensor:
    """Share table spreading each expert's traffic evenly over its replicas.

    This is what dynamic routing samples per token, expressed once per expert.
    """

    slot = torch.arange(candidates.shape[1], device=candidates.device)
    live = slot.unsqueeze(0) < counts.unsqueeze(-1).to(slot.dtype)
    safe = counts.clamp(min=1).to(torch.float32).unsqueeze(-1)
    return torch.where(
        live,
        torch.ones(candidates.shape, dtype=torch.float32, device=candidates.device) / safe,
        torch.zeros(candidates.shape, dtype=torch.float32, device=candidates.device),
    )


def _fused_replica_dispatch(request, shares, candidates):
    """Route through the shared dispatch kernel, recording load in the same pass.

    All three replica policies differ only in the share table they hand over --
    None for a uniform draw, one-hot for a pinned replica, an LP solution for
    LPLB -- so they can share one launch instead of each assembling ids from
    tensor ops and leaving the framework to count them afterwards. Measured on
    H20 at 29.6 us per layer against 151.9 us for the tensor-op route.

    Returns ``(physical_ids, recorded)``; ``recorded`` is False when no load
    view was offered, and the framework then does its own accounting pass.
    """
    from moe_load_balancer.kernels.lplb import CUDALPLBKernels

    kernels = getattr(request, "_kernels", None) or CUDALPLBKernels()
    ids = kernels.dispatch_probability(
        request.logical_topk_ids,
        shares,
        candidates,
        expert_load_view=request.expert_load_view,
        record_enabled=request.record_enabled,
        num_unpadded_tokens=request.num_unpadded_tokens,
    )
    return ids, request.expert_load_view is not None


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
        if request.defer_dispatch:
            # The caller applies the split itself. Runtimes that fuse replica
            # selection with expert-load recording cannot take per-token ids
            # without also taking over the recording, so a policy that ignores
            # this request is simply unusable there.
            return RoutingDecision(
                routed_physical_topk_ids=request.logical_topk_ids,
                topk_weights=request.topk_weights,
                metadata={
                    "policy": policy.name,
                    "replica_selection": ReplicaSelection(
                        probability=one_hot_replica_shares(
                            default_map,
                            request.placement.logical_to_physical_candidates,
                            request.placement.logical_to_physical_count,
                        ),
                        candidates=request.placement.logical_to_physical_candidates,
                    ),
                },
            )

        candidates = request.placement.logical_to_physical_candidates
        if request.logical_topk_ids.is_cuda and request.expert_load_view is not None:
            shares = one_hot_replica_shares(
                default_map, candidates, request.placement.logical_to_physical_count
            )
            physical_ids, recorded = _fused_replica_dispatch(
                request, shares, candidates)
        else:
            physical_ids = route_static_replicas(request.logical_topk_ids, default_map)
            recorded = False
        return RoutingDecision(
            routed_physical_topk_ids=physical_ids,
            recorded_load=recorded,
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
        if request.defer_dispatch:
            return RoutingDecision(
                routed_physical_topk_ids=request.logical_topk_ids,
                topk_weights=request.topk_weights,
                metadata={
                    "policy": policy.name,
                    "replica_selection": ReplicaSelection(
                        probability=uniform_replica_shares(
                            request.placement.logical_to_physical_candidates,
                            request.placement.logical_to_physical_count,
                        ),
                        candidates=request.placement.logical_to_physical_candidates,
                    ),
                },
            )

        candidates = request.placement.logical_to_physical_candidates
        if request.logical_topk_ids.is_cuda and request.expert_load_view is not None:
            # No share table: the kernel then draws the way the framework's own
            # does, hash modulo live replicas, which is what keeps this policy
            # bit-identical to the built-in path it is the control for.
            physical_ids, recorded = _fused_replica_dispatch(request, None, candidates)
        else:
            physical_ids = route_dynamic_replicas(
                request.logical_topk_ids,
                candidates,
                request.placement.logical_to_physical_count,
            )
            recorded = False
        return RoutingDecision(
            routed_physical_topk_ids=physical_ids,
            recorded_load=recorded,
            topk_weights=request.topk_weights,
            metadata={"policy": policy.name},
        )
