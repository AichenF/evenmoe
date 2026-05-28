from __future__ import annotations

from typing import Protocol, runtime_checkable

from moe_load_balancer.core.types import PlacementPlan, PlacementRequest, PlacementSnapshot, PlacementStats


@runtime_checkable
class PlacementProvider(Protocol):
    """Read-only placement source used by L2 routing.

    Adapters that own framework-specific placement state (e.g. SGLang's
    ``ExpertLocationMetadata``) implement this protocol to surface a
    framework-neutral per-layer view to L2 algorithms.
    """

    def snapshot(self, layer_id: int) -> PlacementSnapshot:
        ...


@runtime_checkable
class PlacementPlanner(Protocol):
    """Framework-neutral L1 placement planning contract.

    A planner computes a new expert placement plan from current statistics.
    The current canonical implementation is
    :class:`moe_load_balancer.policies.l1.eplb.DeepSeekEPLBPlanner`, which
    wraps the DeepSeek EPLB algorithm; future planners (CRAFT, MoETuner,
    replica-budget allocators) implement the same protocol.

    Equivalence contract: for the same ``request.stats.logical_count`` and
    ``request.policy.metadata`` scalars, ``plan(...)`` must return a
    ``PlacementPlan`` whose ``metadata`` carries the same raw EPLB output
    tensors as the upstream ``eplb_algorithms.rebalance_experts(...)``
    return value, so framework adapters can write back to native placement
    structures without losing information.

    The plan is expressed as:

    - ``plan.snapshots``: per-layer framework-neutral ``PlacementSnapshot``
      view, consumable by L2 routing or any non-adapter caller.
    - ``plan.metadata``: raw algorithm outputs for adapter writeback. For
      EPLB this contains ``physical_to_logical_map``,
      ``logical_to_all_physical_map`` and ``expert_count`` tensors.

    A planner that is gated off (``policy.is_disabled``) should return an
    empty ``PlacementPlan(snapshots={})`` with ``metadata={"skipped": True}``.
    """

    def plan(self, request: PlacementRequest) -> PlacementPlan:
        ...


@runtime_checkable
class StatsCollector(Protocol):
    """Framework-neutral L1 statistics-collection contract.

    Implementations (typically framework adapters such as
    :class:`SGLangStatsCollector`) drive their own data collection during
    inference. The L1 planner -- or the rebalance orchestration around it --
    calls :meth:`collect` once per rebalance window to obtain a
    framework-neutral :class:`PlacementStats` snapshot.

    Internal collection mechanics (per-token / per-dispatch hooks,
    rank-local accumulators, reduction across EP ranks) stay inside the
    implementation; the protocol surfaces only the aggregated output. This
    mirrors the L2 ``PlacementProvider`` pattern: framework-specific reads
    happen inside the adapter, framework-neutral types cross the boundary.

    Lifecycle methods (``start`` / ``stop``) follow the canonical
    ``ExpertDistributionRecorder.start_record` / ``stop_record`` shape;
    implementations that are always-on may treat them as no-ops.

    :meth:`collect` carries an *atomic snapshot-and-reset* semantics --
    after the call, the underlying accumulator is cleared so the next
    window starts fresh. This matches sglang's ``dump_record`` behavior
    and lets the L1 rebalance loop be a single read.
    """

    @property
    def recording(self) -> bool:
        """Whether the collector is currently accumulating data."""
        ...

    def start(self) -> None:
        """Begin (or resume) data collection."""
        ...

    def stop(self) -> None:
        """Pause data collection."""
        ...

    def collect(self) -> PlacementStats:
        """Snapshot accumulated stats into a framework-neutral
        ``PlacementStats`` and clear the underlying accumulator.

        Returns an empty ``PlacementStats()`` (all fields ``None``) if
        no data has been collected yet.
        """
        ...
