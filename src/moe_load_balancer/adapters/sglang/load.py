from __future__ import annotations

from typing import Any

from moe_load_balancer.core.types import RoutingLoadView


class SGLangRoutingLoadProvider:
    """Exposes SGLang runtime load state to L2 policies.

    This provider is the adapter-owned place to hold SGLang-specific live
    references, such as ``ExpertLocationMetadata.rank_load``. Policies consume
    the framework-neutral ``RoutingLoadView`` returned here and do not need to
    receive the same prior load through every routing request.
    """

    def __init__(self, metadata_source: Any):
        self._metadata_source = metadata_source

    def load_view(self, layer_id: int) -> RoutingLoadView:
        raise NotImplementedError(
            "Read ExpertLocationMetadata.rank_load or other SGLang load state, "
            "then expose it as a RoutingLoadView."
        )
