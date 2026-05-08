from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from moe_load_balancer.core.types import TensorLike


class SGLangWaterfillDispatchPlan(NamedTuple):
    """SGLang DeepEP Waterfill dispatch-plan inputs.

    ``rank_load`` is the materializer-owned live load object. The fused SGLang
    materializer expects a ``torch.Tensor``; tests may use Python sequences.
    """

    rank_load: TensorLike
    allow_all_ranks: bool
    target_total: int


def build_static_waterfill_dispatch_plan(rank_load: TensorLike) -> SGLangWaterfillDispatchPlan:
    """Build the no-runtime-all-reduce dispatch plan for SGLang static Waterfill."""

    return SGLangWaterfillDispatchPlan(
        rank_load=rank_load,
        allow_all_ranks=True,
        target_total=0,
    )


def build_dynamic_waterfill_dispatch_plan(
    routed_rank_load: TensorLike,
    local_tokens_per_rank: TensorLike | None,
    *,
    topk_size: int,
) -> SGLangWaterfillDispatchPlan:
    """Build a SGLang dynamic Waterfill dispatch plan from all-reduced load.

    ``routed_rank_load`` is the all-reduced routed-expert token count per EP
    rank. ``local_tokens_per_rank`` is the all-reduced vector of each rank's
    active token count; when supplied, it is added to routed load to match the
    effective load consumed by SGLang's fused materializer. ``target_total``
    matches the host-side precomputed value that SGLang passes to that
    materializer.
    """

    if topk_size <= 0:
        raise ValueError("topk_size must be positive.")
    if _supports_tensor_reductions(routed_rank_load):
        return _build_dynamic_tensor_dispatch_plan(
            routed_rank_load,
            local_tokens_per_rank,
            topk_size=topk_size,
        )

    routed_load_values = _as_number_list(routed_rank_load, "routed_rank_load")
    _validate_rank_load_values(routed_load_values, "routed_rank_load")
    if local_tokens_per_rank is None:
        effective_load_values = routed_load_values
        effective_rank_load = routed_rank_load
    else:
        token_values = _as_number_list(local_tokens_per_rank, "local_tokens_per_rank")
        _validate_rank_load_values(token_values, "local_tokens_per_rank")
        if len(token_values) != len(routed_load_values):
            raise ValueError("local_tokens_per_rank must match routed_rank_load length.")
        effective_load_values = [
            routed + tokens for routed, tokens in zip(routed_load_values, token_values)
        ]
        effective_rank_load = _add_rank_load(routed_rank_load, local_tokens_per_rank)

    target_total = _dynamic_target_total(
        routed_rank_load=routed_load_values,
        effective_rank_load=effective_load_values,
        topk_size=topk_size,
    )
    return SGLangWaterfillDispatchPlan(
        rank_load=effective_rank_load,
        allow_all_ranks=max(effective_load_values) <= target_total,
        target_total=target_total,
    )


def _build_dynamic_tensor_dispatch_plan(
    routed_rank_load: TensorLike,
    local_tokens_per_rank: TensorLike | None,
    *,
    topk_size: int,
) -> SGLangWaterfillDispatchPlan:
    effective_rank_load = (
        routed_rank_load
        if local_tokens_per_rank is None
        else routed_rank_load + local_tokens_per_rank
    )
    rank_count = len(effective_rank_load)
    total_routed = routed_rank_load.sum()
    total_tokens = total_routed // topk_size
    total_effective = effective_rank_load.sum()
    target_total = int((total_effective + total_tokens + rank_count - 1) // rank_count)
    return SGLangWaterfillDispatchPlan(
        rank_load=effective_rank_load,
        allow_all_ranks=bool(effective_rank_load.max() <= target_total),
        target_total=target_total,
    )


def _dynamic_target_total(
    *,
    routed_rank_load: Sequence[float],
    effective_rank_load: Sequence[float],
    topk_size: int,
) -> int:
    total_routed = int(sum(routed_rank_load))
    total_tokens = total_routed // topk_size
    total_effective = int(sum(effective_rank_load))
    rank_count = len(effective_rank_load)
    return (total_effective + total_tokens + rank_count - 1) // rank_count


def _supports_tensor_reductions(value: TensorLike) -> bool:
    return (
        not isinstance(value, Sequence)
        and hasattr(value, "sum")
        and hasattr(value, "max")
        and hasattr(value, "__len__")
    )


def _validate_rank_load_values(values: Sequence[float], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty.")
    if any(load < 0 for load in values):
        raise ValueError(f"{name} values must be non-negative.")


def _add_rank_load(left: TensorLike, right: TensorLike) -> TensorLike:
    if isinstance(left, Sequence) and isinstance(right, Sequence):
        if isinstance(left, (str, bytes)) or isinstance(right, (str, bytes)):
            raise TypeError("rank load values must not be strings.")
        return [float(left_item) + float(right_item) for left_item, right_item in zip(left, right)]
    return left + right


def _as_number_list(value: TensorLike, name: str) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a 1D sequence.")
    return [float(item) for item in value]
