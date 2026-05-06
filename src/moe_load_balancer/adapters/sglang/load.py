from __future__ import annotations

from typing import Any

from moe_load_balancer.core.types import RoutingLoadView


class SGLangRoutingLoadProvider:
    """Exposes SGLang runtime load state to L2 policies.

    This provider is the adapter-owned place to hold SGLang-specific live
    references, such as ``ExpertLocationMetadata.rank_load``. Policies consume
    the framework-neutral ``RoutingLoadView`` returned here and do not need to
    receive the same prior load through every routing request.

    For PR #19290-style static Waterfill, this provider must expose the original
    live rank-load storage. It may wrap the reference in ``RoutingLoadView`` or
    select a layer view, but it must not copy values into a new buffer unless it
    also owns an explicit refresh path for that buffer.
    """

    def __init__(self, metadata_source: Any):
        self._metadata_source = metadata_source

    def load_view(self, layer_id: int) -> RoutingLoadView:
        raise NotImplementedError(
            "Expose ExpertLocationMetadata.rank_load or other SGLang load "
            "state as a no-copy RoutingLoadView."
        )
