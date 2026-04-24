from __future__ import annotations

from typing import Any

from moe_load_balancer.core.types import PlacementSnapshot


class SGLangPlacementAdapter:
    """Converts SGLang EPLB metadata into PlacementSnapshot.

    SGLang-specific imports should stay inside this adapter implementation, not
    in core contracts.
    """

    def __init__(self, metadata_source: Any):
        self._metadata_source = metadata_source

    def snapshot(self, layer_id: int) -> PlacementSnapshot:
        raise NotImplementedError(
            "Convert ExpertLocationMetadata / ExpertLocationDispatchInfo here."
        )

