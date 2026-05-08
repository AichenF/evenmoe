from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

from moe_load_balancer.core.l2 import TokenRouter
from moe_load_balancer.core.types import (
    PlacementSnapshot,
    RoutingDecision,
    RoutingRequest,
    TensorLike,
)


@dataclass(frozen=True)
class WaterfillPolicyConfig:
    """Framework-neutral knobs for shared-expert Waterfill.

    This config intentionally describes only the policy decision. Framework
    adapters remain responsible for producing ``current_rank_load`` and
    materializing the returned shared-expert rank into their TopK layout.
    """

    local_rank: int = 0
    local_preference: float = 1.1
    allow_all_ranks: bool = True
    target_total: Optional[int] = None
    routed_rank_load: Optional[TensorLike] = None

    @classmethod
    def from_request(cls, request: RoutingRequest) -> "WaterfillPolicyConfig":
        metadata = request.policy.metadata
        return cls(
            local_rank=int(metadata.get("local_rank", 0)),
            local_preference=float(metadata.get("local_preference", 1.1)),
            allow_all_ranks=_metadata_bool(metadata.get("allow_all_ranks", True)),
            target_total=(
                None
                if metadata.get("target_total") is None
                else _positive_or_none(metadata["target_total"], "target_total")
            ),
            routed_rank_load=metadata.get("routed_rank_load"),
        )

    def __post_init__(self) -> None:
        if self.local_rank < 0:
            raise ValueError("WaterfillPolicyConfig.local_rank must be non-negative.")
        if self.local_preference <= 0:
            raise ValueError("WaterfillPolicyConfig.local_preference must be positive.")
        if self.target_total is not None and self.target_total < 0:
            raise ValueError("WaterfillPolicyConfig.target_total must be non-negative.")


class SharedExpertWaterfillPolicy(TokenRouter):
    """Example L2 policy for shared-expert Waterfill.

    The policy consumes request-local rank load and chooses the target EP rank
    for each token's shared expert. It does not encode framework-specific expert
    IDs, append TopK columns, or call collectives. Those are adapter
    responsibilities.

    This class is a reference implementation of the framework-neutral
    ``TokenRouter`` interface. High-throughput integrations should use an
    adapter-owned vectorized implementation for their runtime layout.
    """

    def route_tokens(self, request: RoutingRequest) -> RoutingDecision:
        routed_physical_topk_ids = _logical_to_physical_topk(
            request.logical_topk_ids,
            request.placement,
        )
        if not request.policy.should_run_for_stage(request.stage):
            return RoutingDecision(
                routed_physical_topk_ids=routed_physical_topk_ids,
                topk_weights=request.topk_weights,
                shared_expert_rank=None,
                metadata={"policy": request.policy.name, "skipped": True},
            )

        if request.current_rank_load is None:
            raise ValueError("SharedExpertWaterfillPolicy requires current_rank_load.")

        config = WaterfillPolicyConfig.from_request(request)
        rank_load = _as_number_list(request.current_rank_load, "current_rank_load")
        _validate_rank_load_values(rank_load, "current_rank_load")
        if config.local_rank >= len(rank_load):
            raise ValueError("WaterfillPolicyConfig.local_rank must be smaller than rank count.")
        routed_rank_load = None
        if config.routed_rank_load is not None:
            # Dynamic Waterfill can use effective load for rank capacity while
            # deriving the active token count from routed load.
            routed_rank_load = _as_number_list(config.routed_rank_load, "routed_rank_load")
            _validate_rank_load_values(routed_rank_load, "routed_rank_load")
            if len(routed_rank_load) != len(rank_load):
                raise ValueError("routed_rank_load must match current_rank_load length.")

        topk_rows = _as_2d_int_list(routed_physical_topk_ids, "routed_physical_topk_ids")
        target_total = _target_total(
            rank_load=rank_load,
            routed_rank_load=routed_rank_load,
            topk_size=_topk_size(topk_rows),
            rank_count=len(rank_load),
            configured=config.target_total,
        )
        shared_expert_rank = [
            _select_rank_for_token(
                token_index=i,
                physical_topk=row,
                rank_load=rank_load,
                placement=request.placement,
                config=config,
                target_total=target_total,
            )
            for i, row in enumerate(topk_rows)
        ]

        return RoutingDecision(
            routed_physical_topk_ids=routed_physical_topk_ids,
            topk_weights=request.topk_weights,
            shared_expert_rank=shared_expert_rank,
            metadata={
                "policy": request.policy.name,
                "allow_all_ranks": config.allow_all_ranks,
                "local_rank": config.local_rank,
                "target_total": target_total,
            },
        )


def _logical_to_physical_topk(
    logical_topk_ids: TensorLike,
    placement: PlacementSnapshot,
) -> list[list[int]]:
    rows = _as_2d_int_list(logical_topk_ids, "logical_topk_ids")
    default_physical = _optional_int_list(
        placement.default_physical_for_logical,
        "default_physical_for_logical",
    )
    candidates = _optional_2d_int_list(
        placement.logical_to_physical_candidates,
        "logical_to_physical_candidates",
    )
    counts = _optional_int_list(
        placement.logical_to_physical_count,
        "logical_to_physical_count",
    )

    mapped: list[list[int]] = []
    for row in rows:
        mapped_row: list[int] = []
        for logical_id in row:
            if logical_id < 0:
                mapped_row.append(logical_id)
            elif default_physical is not None:
                _check_index(logical_id, len(default_physical), "logical expert")
                physical_id = default_physical[logical_id]
                _check_index(physical_id, placement.num_physical_experts, "physical expert")
                mapped_row.append(physical_id)
            else:
                if candidates is None or counts is None:
                    raise ValueError(
                        "PlacementSnapshot needs default_physical_for_logical or "
                        "logical_to_physical_candidates/count."
                    )
                _check_index(logical_id, len(candidates), "logical expert")
                _check_index(logical_id, len(counts), "logical expert")
                if counts[logical_id] <= 0:
                    mapped_row.append(-1)
                else:
                    if not candidates[logical_id]:
                        raise ValueError(
                            f"logical expert id {logical_id} has no physical candidates."
                        )
                    physical_id = candidates[logical_id][0]
                    _check_index(physical_id, placement.num_physical_experts, "physical expert")
                    mapped_row.append(physical_id)
        mapped.append(mapped_row)
    return mapped


def _select_rank_for_token(
    *,
    token_index: int,
    physical_topk: Sequence[int],
    rank_load: Sequence[float],
    placement: PlacementSnapshot,
    config: WaterfillPolicyConfig,
    target_total: int,
) -> int:
    candidates = _candidate_ranks(
        physical_topk=physical_topk,
        placement=placement,
        rank_count=len(rank_load),
        local_rank=config.local_rank,
        allow_all_ranks=config.allow_all_ranks,
    )
    weighted = [
        (
            rank,
            _waterfill_weight(
                rank=rank,
                load=rank_load[rank],
                local_rank=config.local_rank,
                local_preference=config.local_preference,
                target_total=target_total,
            ),
        )
        for rank in candidates
    ]
    total_weight = sum(weight for _, weight in weighted)
    if total_weight <= 0:
        return min(
            candidates,
            key=lambda rank: (
                _adjusted_load(
                    rank=rank,
                    load=rank_load[rank],
                    local_rank=config.local_rank,
                    local_preference=config.local_preference,
                ),
                rank != config.local_rank,
                rank,
            ),
        )

    threshold = _deterministic_fraction(token_index, config.local_rank) * total_weight
    cumulative = 0.0
    for rank, weight in weighted:
        cumulative += weight
        if threshold < cumulative:
            return rank
    return weighted[-1][0]


def _candidate_ranks(
    *,
    physical_topk: Sequence[int],
    placement: PlacementSnapshot,
    rank_count: int,
    local_rank: int,
    allow_all_ranks: bool,
) -> list[int]:
    if allow_all_ranks:
        return list(range(rank_count))

    ranks = {local_rank}
    physical_to_rank = _optional_int_list(
        placement.physical_to_rank_map,
        "physical_to_rank_map",
    )
    if physical_to_rank is None:
        return sorted(ranks)

    for physical_id in physical_topk:
        if physical_id < 0:
            continue
        _check_index(physical_id, len(physical_to_rank), "physical expert")
        rank = physical_to_rank[physical_id]
        if 0 <= rank < rank_count:
            ranks.add(rank)
    return sorted(ranks)


def _waterfill_weight(
    *,
    rank: int,
    load: float,
    local_rank: int,
    local_preference: float,
    target_total: int,
) -> float:
    weight = max(float(target_total) - load, 0.0)
    if rank != local_rank:
        weight /= local_preference
    return weight


def _adjusted_load(
    *,
    rank: int,
    load: float,
    local_rank: int,
    local_preference: float,
) -> float:
    if rank == local_rank:
        return load / local_preference
    return load


def _target_total(
    *,
    rank_load: Sequence[float],
    routed_rank_load: Optional[Sequence[float]],
    topk_size: int,
    rank_count: int,
    configured: Optional[int],
) -> int:
    if configured is not None and configured > 0:
        return configured
    total_effective = int(sum(rank_load))
    total_routed = int(sum(routed_rank_load)) if routed_rank_load is not None else total_effective
    total_tokens = total_routed // max(topk_size, 1)
    return (total_effective + total_tokens + rank_count - 1) // rank_count


def _topk_size(rows: Sequence[Sequence[int]]) -> int:
    return max((len(row) for row in rows), default=1)


def _deterministic_fraction(token_index: int, local_rank: int) -> float:
    value = (token_index & 0xFFFFFFFF) ^ ((local_rank * 0x9E3779B9) & 0xFFFFFFFF)
    value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
    return value / 2**32


def _as_2d_int_list(value: TensorLike, name: str) -> list[list[int]]:
    value = _tolist(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a 2D sequence.")
    rows: list[list[int]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise TypeError(f"{name} must be a 2D sequence.")
        rows.append([int(item) for item in row])
    return rows


def _optional_2d_int_list(value: TensorLike, name: str) -> Optional[list[list[int]]]:
    if value is None:
        return None
    return _as_2d_int_list(value, name)


def _optional_int_list(value: TensorLike, name: str) -> Optional[list[int]]:
    if value is None:
        return None
    value = _tolist(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a 1D sequence.")
    return [int(item) for item in value]


def _as_number_list(value: TensorLike, name: str) -> list[float]:
    value = _tolist(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a 1D sequence.")
    return [float(item) for item in value]


def _validate_rank_load_values(values: Sequence[float], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty.")
    if any(load < 0 for load in values):
        raise ValueError(f"{name} values must be non-negative.")


def _metadata_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise TypeError("allow_all_ranks integer metadata must be 0 or 1.")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise TypeError("allow_all_ranks metadata must be a bool, 0/1, or bool-like string.")


def _positive_or_none(value: object, name: str) -> Optional[int]:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    if value == 0:
        return None
    return value


def _check_index(index: int, length: int, label: str) -> None:
    if index >= length:
        raise ValueError(f"{label} id {index} is out of range for length {length}.")


def _tolist(value: TensorLike) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
