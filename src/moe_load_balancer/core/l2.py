from __future__ import annotations

from typing import Protocol, runtime_checkable

from moe_load_balancer.core.types import (
    RoutingDecision,
    RoutingLoadHistory,
    RoutingPhase,
    RoutingPolicyConfig,
    RoutingRequest,
)


@runtime_checkable
class RoutingLoadHistoryProvider(Protocol):
    """Optional source for future policies that consume cross-batch history."""

    def load_history(self, layer_id: int) -> RoutingLoadHistory: ...


@runtime_checkable
class TokenRouter(Protocol):
    """One framework-neutral policy in the ordered L2 pipeline."""

    phase: RoutingPhase

    def route_tokens(
        self,
        request: RoutingRequest,
        policy: RoutingPolicyConfig,
    ) -> RoutingDecision: ...
