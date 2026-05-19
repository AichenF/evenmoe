from __future__ import annotations

from typing import Protocol, runtime_checkable

from moe_load_balancer.core.types import PlacementPlan, PlacementRequest, PlacementSnapshot


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
