from __future__ import annotations

from typing import Protocol, runtime_checkable

from moe_load_balancer.core.types import PlacementPlan, PlacementRequest


@runtime_checkable
class PlacementPlanner(Protocol):
    """Framework-neutral L1 candidate-placement policy."""

    def plan(self, request: PlacementRequest) -> PlacementPlan: ...
