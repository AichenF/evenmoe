from __future__ import annotations

from typing import Protocol, runtime_checkable

from moe_load_balancer.core.types import RoutingDecision, RoutingRequest


@runtime_checkable
class TokenRouter(Protocol):
    """Framework-neutral L2 token router."""

    def route_tokens(self, request: RoutingRequest) -> RoutingDecision:
        ...
