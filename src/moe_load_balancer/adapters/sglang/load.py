from __future__ import annotations

from typing import Any

from moe_load_balancer.core.l2 import RoutingLoadProvider
from moe_load_balancer.core.types import RoutingLoadView


class SGLangRoutingLoadProvider(RoutingLoadProvider):
    """Exposes SGLang runtime load state to L2 policies.

    This provider is the adapter-owned place to hold SGLang-specific live
    references when an integration exposes long-lived load state. Policies
    consume the framework-neutral ``RoutingLoadView`` returned here and do not
    need to receive the same prior load through every routing request.

    Latest PR #19290 Waterfill no longer uses a long-lived SGLang
    ``ExpertLocationMetadata.rank_load`` input; its rank load is request-local
    and should be passed through ``RoutingRequest.current_rank_load``. Keep this
    provider for other L2 policies or future integrations that genuinely need a
    live/history load view.
    """

    def __init__(self, metadata_source: Any):
        self._metadata_source = metadata_source

    def load_view(self, layer_id: int) -> RoutingLoadView:
        raise NotImplementedError(
            "Expose SGLang long-lived load state as a no-copy RoutingLoadView "
            "when the selected policy requires it."
        )
