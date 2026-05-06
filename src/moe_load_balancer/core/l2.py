from __future__ import annotations

from typing import Protocol, runtime_checkable

from moe_load_balancer.core.types import RoutingDecision, RoutingLoadView, RoutingRequest


@runtime_checkable
class RoutingLoadProvider(Protocol):
    """Framework-neutral provider for long-lived L2 load state."""

    def load_view(self, layer_id: int) -> RoutingLoadView:
        ...


@runtime_checkable
class TokenRouter(Protocol):
    """Framework-neutral L2 token router."""

    def route_tokens(self, request: RoutingRequest) -> RoutingDecision:
        ...
