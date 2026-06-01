from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
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
    An explicit set containing every stage is stricter than ``None`` because it
    still rejects requests with unknown stage.
    The policy implementation may still ignore ``RoutingRequest.stage`` after
    this common gating decision.

    ``metadata`` is for **per-policy stable configuration** (immutable scalars
    such as algorithm hyperparameters, layout assumptions, mode hints).
    Tensors must NOT be placed in ``metadata``: per-step tensor inputs flow
    through ``RoutingRequest`` top-level fields (``current``-batch state) or
    ``RoutingLoadHistoryProvider`` (long-lived load views). Putting a tensor
    here would create misleading immutability semantics because the frozen
    dataclass + ``MappingProxyType`` wrapper does not prevent in-place updates
    to the wrapped tensor, only to the dictionary slot.
    """

    DISABLED: ClassVar[str] = "disabled"

    name: str = DISABLED
    enabled_stages: Optional[frozenset[RoutingStage]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
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
    """Input statistics for L1 placement planning.

    For EPLB the canonical field is ``logical_count``: per-layer per-logical-
    expert activation count, summed over a recording window or a single batch.
    Shape ``[num_layers, num_logical_experts]`` (already-summed) or
    ``[num_steps, num_layers, num_logical_experts]`` (multi-step, summed by
    the planner). Future planners may consume ``physical_count`` / ``rank_load``.
    """

    logical_count: Optional[TensorLike] = None
    physical_count: Optional[TensorLike] = None
    rank_load: Optional[TensorLike] = None
    # Aggregate GPU expert-utilization rate over the most recent collection
    # window, as reported by the framework. Used by rebalance orchestration
    # (not by the L1 planner) to skip rebalances that would not meaningfully
    # change throughput. ``None`` means "not measured".
    average_utilization_rate: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlacementPolicyConfig:
    """Framework-neutral L1 placement policy selection.

    Mirrors :class:`RoutingPolicyConfig` for L1. ``name`` identifies the
    placement algorithm family (``"eplb"`` for the DeepSeek EPLB family,
    ``"disabled"`` to skip). ``metadata`` carries stable per-policy scalar
    config: algorithm variant (``"auto"`` / ``"deepseek"`` /
    ``"deepseek_hierarchical"`` / ...), layout scalars
    (``num_physical_experts``, ``num_local_physical_experts``,
    ``num_groups``, ``num_nodes``). Tensors MUST NOT go here for *scalars*; read-only tensor references that are
    policy parameters (e.g. ``active_ranks`` for the elasticity-
    aware EPLB variants) are acceptable. The boundary is "don't
    put load *statistics* in metadata; those flow through
    :class:`PlacementRequest.stats`."

    L1 has no stage gating (placement is computed offline or at low
    frequency, not per-batch), so this class is intentionally simpler than
    ``RoutingPolicyConfig`` and omits ``enabled_stages``.
    """

    DISABLED: ClassVar[str] = "disabled"

    name: str = DISABLED
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_value(
        cls, value: "PlacementPolicyConfig | str"
    ) -> "PlacementPolicyConfig":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(name=value)
        raise TypeError(
            "PlacementPolicyConfig.from_value expects PlacementPolicyConfig or str, "
            f"got {type(value).__name__}."
        )

    @property
    def is_disabled(self) -> bool:
        return self.name == self.DISABLED


@dataclass(frozen=True)
class RebalancePolicyConfig:
    """Framework-neutral configuration for the L1 rebalance loop.

    Mirrors :class:`PlacementPolicyConfig` / :class:`RoutingPolicyConfig` in
    that it groups tunables for a single concern (periodic rebalance
    triggering and the recording window that feeds it). MLB provides
    defaults; framework adapters may override per-deployment by passing
    overrides into the runtime/recorder/provider constructors.

    Fields:

    * ``recording_window_size`` -- how many forward passes the recorder
      keeps in its rolling buffer. ``logical_count`` returned from
      :meth:`StatsCollector.collect` is the sum across this window.
      Matches sglang's
      ``server_args.expert_distribution_recorder_buffer_size`` semantics.

    * ``min_utilization_threshold`` -- per-rebalance skip gate. If the
      recently-measured GPU expert utilization rate is above this value,
      the rebalance call is a no-op (no new plan published). ``None``
      means "never skip". Matches sglang's
      ``server_args.eplb_min_rebalancing_utilization_threshold``.

    The rebalance *period* itself (every N forward passes) is **not** in
    here -- that's owned by the orchestration layer (sglang ``EPLBManager``
    holds ``--eplb-rebalance-num-iterations``). MLB sees only the trigger
    edge; whether the trigger fires is the orchestrator's call.
    """

    recording_window_size: int = 100
    min_utilization_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        if self.recording_window_size <= 0:
            raise ValueError(
                f"recording_window_size must be > 0 (got "
                f"{self.recording_window_size!r}); infinite-buffer mode "
                f"is not supported."
            )
        thr = self.min_utilization_threshold
        if thr is not None:
            # Reject NaN (NaN != NaN) and non-numeric.
            if not isinstance(thr, (int, float)) or (
                isinstance(thr, float) and thr != thr
            ):
                raise ValueError(
                    f"min_utilization_threshold must be None or a finite "
                    f"float in [0.0, 1.0]; got {thr!r}."
                )
            if thr < 0.0 or thr > 1.0:
                raise ValueError(
                    f"min_utilization_threshold must be in [0.0, 1.0]; "
                    f"got {thr!r}."
                )


@dataclass(frozen=True, kw_only=True)
class PlacementRequest:
    """Framework-neutral L1 planning request.

    Carries the inputs a :class:`PlacementPlanner` needs to produce a
    placement plan. For EPLB the required inputs are ``stats.logical_count``
    plus the scalars in ``policy.metadata``: ``num_physical_experts``,
    ``num_local_physical_experts``, ``num_groups``, ``num_nodes``, and
    optionally ``algorithm`` (defaults to ``"auto"``).

    ``current`` is optional — EPLB does not consume the current placement
    (it produces an absolute new placement from stats only). Future planners
    (CRAFT / MoETuner) may use ``current`` for diff-based budgeting.
    """

    stats: PlacementStats
    policy: PlacementPolicyConfig | str = field(default_factory=PlacementPolicyConfig)
    current: Optional[Mapping[int, PlacementSnapshot]] = None
    hbm_budget_bytes: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", PlacementPolicyConfig.from_value(self.policy))


@dataclass(frozen=True)
class PlacementPlan:
    """Framework-neutral L1 planning output.

    ``snapshots`` is the per-layer placement view consumable by L2 (and by
    framework adapters for writeback). ``metadata`` carries raw algorithm
    outputs for adapters that need them verbatim — for EPLB this is
    ``{"physical_to_logical_map": Tensor, "logical_to_all_physical_map":
    Tensor, "expert_count": Tensor}``, so the SGLang adapter can construct
    ``ExpertLocationMetadata`` directly without round-tripping through
    ``snapshots``.
    """

    snapshots: Mapping[int, PlacementSnapshot]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingLoadHistory:
    """Long-lived L2 load view exposed by a ``RoutingLoadHistoryProvider``.

    This view captures load state that persists across batches (live tensor
    references the framework maintains continuously, EWMA / aggregated
    historical statistics, per-expert and per-rank-per-expert breakdowns).
    Per-batch transient state (``current_rank_load`` style) flows through
    ``RoutingRequest`` top-level fields instead, so policies that only need
    per-batch view do not have to subscribe to a provider.

    ``live_rank_load`` is intentionally distinct from request-local
    ``RoutingRequest.routed_rank_load`` / ``active_rank_token_count``: it is a
    reference-style channel for frameworks that maintain a long-lived live
    counter (e.g. SGLang EPLB recorder). Adapters expose the live tensor
    without copying so policies can read the latest value each routing call.

    LP / LB-style policies (LPLB, Metro) use ``historical_expert_load`` and
    ``historical_rank_expert_load`` for the per-expert and 2-D
    ``[num_ranks, num_experts_per_rank]`` views the LP solver needs.
    Shared-expert Waterfill does not consume this view at all.
    """

    live_rank_load: Optional[TensorLike] = None
    historical_rank_load: Optional[TensorLike] = None
    historical_expert_load: Optional[TensorLike] = None
    historical_rank_expert_load: Optional[TensorLike] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class RoutingRequest:
    """Framework-neutral L2 routing request.

    Tensor shapes (when supplied):
        logical_topk_ids:        ``[num_tokens, top_k]`` logical expert ids.
        topk_weights:            ``[num_tokens, top_k]`` matching gate weights.
        routed_rank_load:        ``[ep_size]`` per-rank routed expert assignment
                                 count (post-collective if applicable).
                                 Static: local routed counts (no collective).
                                 Dynamic: post-EP-all-reduce global routed.
        active_rank_token_count: ``[ep_size]`` per-rank active token count
                                 (post-collective). Set only in dynamic mode;
                                 ``None`` in static mode.
        token_count:             scalar or ``[1]`` total live tokens.
        lb_token_mask:           ``[num_tokens]`` bool mask of LB-eligible
                                 tokens.

    ``stage`` is optional runtime context. It lets framework adapters describe
    the current request phase, but algorithms are not required to branch on it.
    Stage-based routing enablement belongs to
    ``policy.should_run_for_stage(request.stage)``.

    ``placement`` is optional. Algorithms that require logical->physical
    translation (LPLB, Metro) must validate it is non-None. Algorithms that
    operate on already-physical IDs (shared-expert Waterfill) leave it ``None``.

    ``routed_rank_load`` and ``active_rank_token_count`` are the canonical raw
    inputs for shared-expert Waterfill. The presence of
    ``active_rank_token_count`` signals dynamic mode; its absence signals
    static mode. The policy combines them as needed (e.g. ``effective =
    routed + active`` for Waterfill dynamic). Adapters compute these from
    framework state (e.g. SGLang adapter via ``count_local_routed_per_rank``
    and an EP-group all-reduce); callers in CPU-only test paths can pass plain
    sequences.
    """

    layer_id: int
    logical_topk_ids: TensorLike
    topk_weights: TensorLike
    placement: Optional[PlacementSnapshot] = None
    stage: Optional[RoutingStage] = None
    routed_rank_load: Optional[TensorLike] = None
    active_rank_token_count: Optional[TensorLike] = None
    # ``[num_logical_experts]`` per-logical token count for this batch — local
    # or post-EP-all-reduce depending on the adapter / env. Consumed by L2
    # algorithms that re-balance across logical-expert replicas (LP-CF).
    global_logical_count: Optional[TensorLike] = None
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
