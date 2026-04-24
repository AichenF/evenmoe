from __future__ import annotations

from typing import Any

from moe_load_balancer.adapters.sglang.context import SGLangRoutingContext
from moe_load_balancer.core.l1 import PlacementProvider
from moe_load_balancer.core.l2 import TokenRouter
from moe_load_balancer.core.types import RoutingDecision, RoutingRequest


class SGLangRoutingAdapter:
    """TopK-level adapter that calls the framework-neutral TokenRouter."""

    def __init__(self, router: TokenRouter, placement_provider: PlacementProvider):
        self._router = router
        self._placement_provider = placement_provider

    def route_topk(
        self,
        *,
        layer_id: int,
        logical_topk_output: Any,
        context: SGLangRoutingContext,
    ) -> Any:
        """Run the adapter-mediated TopK routing flow.

        This public method is orchestration, not a pure conversion helper:

        1. Build a framework-neutral ``RoutingRequest`` from SGLang TopK data.
        2. Call the framework-neutral core ``TokenRouter``.
        3. Materialize the ``RoutingDecision`` back into SGLang's TopK output
           layout.
        """

        raise NotImplementedError(
            "Build RoutingRequest from logical TopK output, call core router, "
            "then materialize RoutingDecision into StandardTopKOutput."
        )

    def _build_routing_request(
        self,
        *,
        layer_id: int,
        logical_topk_output: Any,
        context: SGLangRoutingContext,
    ) -> RoutingRequest:
        """Convert SGLang TopK data into the core request type."""

        raise NotImplementedError

    def _materialize_routing_decision(
        self,
        *,
        logical_topk_output: Any,
        decision: RoutingDecision,
        context: SGLangRoutingContext,
    ) -> Any:
        """Write a core routing decision back into SGLang's TopK output layout."""

        raise NotImplementedError
