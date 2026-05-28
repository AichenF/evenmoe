"""Framework-neutral L1 placement provider.

After the L1 EPLB planner produces a new :class:`PlacementPlan`, this provider
exposes it to L2 routing (LPLB / LP-CF / Waterfill-with-placement variants)
through the canonical :class:`PlacementProvider` Protocol -- *without*
round-tripping through framework-specific structures.

This is the framework-neutral counterpart to
:class:`SGLangPlacementAdapter`. Where the sglang adapter reads
``ExpertLocationMetadata`` (sglang's own state), this provider reads MLB's
own ``_latest_plan`` -- so L2 can be written entirely against
``moe_load_balancer.core.types`` without importing any framework module.

Pull semantics with version-keyed cache invalidation:

* :meth:`snapshot` returns a per-layer ``PlacementSnapshot`` whose
  ``metadata["placement_version"]`` carries the provider's internal
  counter. L2 consumers that cache derived state across calls can use the
  version as an O(1) invalidation key (same shape as
  :class:`SGLangPlacementAdapter`).
* :meth:`update_with` swaps in a new plan atomically and bumps the
  version. Single-ref replacement; no locking required as long as readers
  treat each ``snapshot()`` call as a fresh pull.
"""

from __future__ import annotations

from typing import Optional

from moe_load_balancer.core.l1 import PlacementProvider
from moe_load_balancer.core.types import PlacementPlan, PlacementSnapshot


class EPLBPlacementProvider(PlacementProvider):
    """L1-side placement provider backed by the latest :class:`PlacementPlan`
    published by an EPLB planner.

    Construction is dependency-free. ``snapshot`` raises ``RuntimeError`` if
    no plan has been published yet (the rebalance loop should publish at
    least once before L2 starts pulling).
    """

    def __init__(self) -> None:
        self._plan: Optional[PlacementPlan] = None
        self._version: int = 0

    # ------------------------------------------------------------------
    # Publish side (L1 manager / runtime calls this)
    # ------------------------------------------------------------------

    def update_with(self, plan: PlacementPlan) -> None:
        """Swap in a new plan and bump the version counter.

        Atomic single-ref replacement (CPython ref assignment is atomic).
        Concurrent ``snapshot()`` calls either see the old plan with old
        version or the new plan with new version -- never a torn read.
        """
        self._plan = plan
        self._version += 1

    @property
    def version(self) -> int:
        return self._version

    def latest_plan(self) -> Optional[PlacementPlan]:
        """Direct access to the cached plan. Returns ``None`` if no plan
        has been published yet. Mostly for tests / introspection -- L2
        consumers should use :meth:`snapshot`."""
        return self._plan

    # ------------------------------------------------------------------
    # Pull side (L2 routing calls this via the PlacementProvider Protocol)
    # ------------------------------------------------------------------

    def snapshot(self, layer_id: int) -> PlacementSnapshot:
        plan = self._plan
        if plan is None:
            raise RuntimeError(
                "EPLBPlacementProvider.snapshot called before any plan was "
                "published. The L1 rebalance loop must call update_with() "
                "at least once (typically at cold-start) before L2 reads."
            )
        if layer_id not in plan.snapshots:
            raise KeyError(
                f"layer_id={layer_id} not in latest plan; available layers "
                f"are {sorted(plan.snapshots)}"
            )
        base = plan.snapshots[layer_id]
        # Re-emit the per-layer snapshot with the provider's version stamp
        # so L2 can use placement_version for cache invalidation. We copy
        # the metadata dict (immutable views) and overlay our version.
        merged_metadata = {
            **dict(base.metadata),
            "placement_version": self._version,
        }
        return PlacementSnapshot(
            layer_id=base.layer_id,
            num_logical_experts=base.num_logical_experts,
            num_physical_experts=base.num_physical_experts,
            logical_to_physical_candidates=base.logical_to_physical_candidates,
            logical_to_physical_count=base.logical_to_physical_count,
            default_physical_for_logical=base.default_physical_for_logical,
            physical_to_rank_map=base.physical_to_rank_map,
            metadata=merged_metadata,
        )
