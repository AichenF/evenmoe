from __future__ import annotations

from typing import Protocol, runtime_checkable

from moe_load_balancer.core.types import RoutingDecision, RoutingLoadHistory, RoutingRequest


# todo: deprecated RoutingLoadHistoryProvider protocol, history should be passed directly to TokenRouter
@runtime_checkable
class RoutingLoadHistoryProvider(Protocol):
    """Framework-neutral provider for long-lived L2 load state.

    Adapters that expose continuously-updated load tensors (live counters,
    EWMA / historical aggregates, per-expert breakdowns) implement this
    protocol so policies can read the latest view each routing call instead
    of receiving the same stable load through every ``RoutingRequest``.

    Per-batch transient load (e.g. shared-expert Waterfill's per-batch routed
    counts) is not exposed via this protocol; it goes through canonical
    ``RoutingRequest`` fields instead.
    """

    def load_history(self, layer_id: int) -> RoutingLoadHistory:
        ...


@runtime_checkable
class TokenRouter(Protocol):
    """Framework-neutral L2 token router."""

    def route_tokens(self, request: RoutingRequest) -> RoutingDecision:
        ...
