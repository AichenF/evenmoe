from __future__ import annotations

from typing import Protocol, runtime_checkable

from moe_load_balancer.core.types import PlacementPlan, PlacementRequest, PlacementSnapshot


@runtime_checkable
class PlacementProvider(Protocol):
    """Read-only placement source used by L2 routing."""

    def snapshot(self, layer_id: int) -> PlacementSnapshot:
        ...


@runtime_checkable
class PlacementPlanner(Protocol):
    """Future L1 planner contract.

    Reserved for the future L1 optimizer. The current SGLang POC reads EPLB
    placement only and does not write a PlacementPlan back to
    ExpertLocationMetadata; that writeback is a pending L1 integration
    decision.
    """

    def plan(self, request: PlacementRequest) -> PlacementPlan:
        ...
