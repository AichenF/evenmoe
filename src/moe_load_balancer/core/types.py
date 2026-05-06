from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Mapping, Optional

TensorLike = Any


class RoutingStage(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"
    IDLE = "idle"
    SPECULATIVE = "speculative"

    @classmethod
    def from_str(cls, value: "RoutingStage | str") -> "RoutingStage":
        """Normalize adapter-supplied stage strings into the enum.

        Accepts the enum itself or a case-insensitive string. Raises ValueError
        on unknown string values and TypeError on unsupported input types so
        callers fail fast at the adapter boundary.
        """

        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError(
                "RoutingStage.from_str expects a RoutingStage or str, "
                f"got {type(value).__name__}."
            )
        return cls(value.lower())


@dataclass(frozen=True)
class RoutingPolicyConfig:
    """Framework-neutral policy selection and gating configuration.

    ``name="disabled"`` is the conservative default and should not run a
    routing policy. Any other name identifies a registered policy.
    ``enabled_stages=None`` means the policy is not gated by stage. If a set is
    provided, a request without stage information is not eligible by default.
    The policy implementation may still ignore ``RoutingRequest.stage`` after
    this common gating decision.
    """

    DISABLED: ClassVar[str] = "disabled"

    name: str = DISABLED
    enabled_stages: Optional[frozenset[RoutingStage]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.enabled_stages is None:
            return
        enabled_stages = frozenset(
            RoutingStage.from_str(stage) for stage in self.enabled_stages
        )
        if not enabled_stages:
            raise ValueError("RoutingPolicyConfig.enabled_stages cannot be empty.")
        object.__setattr__(self, "enabled_stages", enabled_stages)

    @classmethod
    def from_value(
        cls, value: "RoutingPolicyConfig | str"
    ) -> "RoutingPolicyConfig":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(name=value)
        raise TypeError(
            "RoutingPolicyConfig.from_value expects RoutingPolicyConfig or str, "
            f"got {type(value).__name__}."
        )

    @property
    def is_disabled(self) -> bool:
        return self.name == self.DISABLED

    def allows_stage(self, stage: Optional[RoutingStage]) -> bool:
        if self.enabled_stages is None:
            return True
        if stage is None:
            return False
        return stage in self.enabled_stages

    def should_run_for_stage(self, stage: Optional[RoutingStage]) -> bool:
        """Return whether this policy should run for this request stage.

        Ungated policies run even when stage is unknown. Stage-gated policies
        require a known stage that appears in ``enabled_stages``.
        """

        return (not self.is_disabled) and self.allows_stage(stage)


@dataclass(frozen=True)
class PlacementSnapshot:
    """Framework-neutral view of one layer's expert placement.

    Tensor shapes (when supplied):
        logical_to_physical_candidates: ``[num_logical_experts, max_replicas]``
            padded with ``-1`` past ``logical_to_physical_count[i]``.
        logical_to_physical_count:      ``[num_logical_experts]``
        default_physical_for_logical:   ``[num_logical_experts]`` baseline remap.
        physical_to_rank_map:           ``[num_physical_experts]`` EP rank id.
    """

    layer_id: int
    num_logical_experts: int
    num_physical_experts: int
    logical_to_physical_candidates: TensorLike
    logical_to_physical_count: TensorLike
    default_physical_for_logical: Optional[TensorLike] = None
    physical_to_rank_map: Optional[TensorLike] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlacementStats:
    """Input statistics for future L1 placement planning."""

    logical_count: Optional[TensorLike] = None
    physical_count: Optional[TensorLike] = None
    rank_load: Optional[TensorLike] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlacementRequest:
    """Reserved L1 planning request.

    Not consumed by the current SGLang POC (which only reads PlacementSnapshot
    from EPLB). Kept as a forward-looking surface for the future L1 planner;
    whether PlacementPlan is written back to ExpertLocationMetadata is pending.
    """

    current: Mapping[int, PlacementSnapshot]
    stats: PlacementStats
    hbm_budget_bytes: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlacementPlan:
    """Reserved L1 planning output (see PlacementRequest)."""

    snapshots: Mapping[int, PlacementSnapshot]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingLoadSnapshot:
    """Optional L2 load view shared by routing policies.

    This is intentionally broader than rank load. LP/LB-style policies may need
    per-expert or per-rank/per-expert statistics, while shared-expert Waterfill
    can often consume only rank load.
    """

    routed_rank_load: Optional[TensorLike] = None
    prior_rank_load: Optional[TensorLike] = None
    physical_expert_load: Optional[TensorLike] = None
    rank_expert_load: Optional[TensorLike] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class RoutingRequest:
    """Framework-neutral L2 routing request.

    Tensor shapes (when supplied):
        logical_topk_ids:   ``[num_tokens, top_k]`` logical expert ids.
        topk_weights:       ``[num_tokens, top_k]`` matching gate weights.
        routed_rank_load:   ``[ep_size]`` current-batch rank load.
        prior_rank_load:    ``[ep_size]`` EPLB-derived prior rank load.
        token_count:        scalar or ``[1]`` total live tokens.
        lb_token_mask:      ``[num_tokens]`` bool mask of LB-eligible tokens.

    ``stage`` is optional runtime context. It lets framework adapters describe
    the current request phase, but algorithms are not required to branch on it.
    Stage-based routing enablement belongs to
    ``policy.should_run_for_stage(request.stage)``.
    """

    layer_id: int
    logical_topk_ids: TensorLike
    topk_weights: TensorLike
    placement: PlacementSnapshot
    stage: Optional[RoutingStage] = None
    routed_rank_load: Optional[TensorLike] = None
    prior_rank_load: Optional[TensorLike] = None
    token_count: Optional[TensorLike] = None
    lb_token_mask: Optional[TensorLike] = None
    routed_scaling_factor: float = 1.0
    policy: RoutingPolicyConfig | str = field(default_factory=RoutingPolicyConfig)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", RoutingPolicyConfig.from_value(self.policy))


@dataclass(frozen=True)
class RoutingDecision:
    """Framework-neutral L2 routing decision.

    ``routed_physical_topk_ids`` lives in the physical-id namespace defined by
    the accompanying ``PlacementSnapshot`` (i.e. indices into
    ``logical_to_physical_candidates``-resolvable physical experts), not in
    SGLang/DeepEP interleaved id space — that mapping is the adapter's job.
    """

    routed_physical_topk_ids: TensorLike
    topk_weights: TensorLike
    shared_expert_rank: Optional[TensorLike] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
