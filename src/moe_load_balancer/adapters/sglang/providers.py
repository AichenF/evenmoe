"""SGLang adapters that produce framework-neutral core types.

These are L1 / L2 *generic* providers shared across L2 algorithms. Algorithm-
specific runtime code (Waterfill / LPLB / Metro) lives in dedicated
subpackages alongside their L2 algorithm class.
"""

from __future__ import annotations

from typing import Any

from moe_load_balancer.core.l1 import PlacementProvider
from moe_load_balancer.core.types import PlacementSnapshot


class SGLangPlacementAdapter(PlacementProvider):
    """Convert SGLang EPLB metadata into a framework-neutral ``PlacementSnapshot``.

    Implements ``core.l1.PlacementProvider``. The current Stage 3 POC reads
    SGLang ``ExpertLocationMetadata`` / ``ExpertLocationDispatchInfo`` and
    derives the per-layer placement view. SGLang-specific imports live inside
    this adapter; the rest of the codebase (and other adapters) consumes only
    the resulting ``PlacementSnapshot``.

    This implementation is a placeholder for the current POC scope (where
    Waterfill does not consume placement). LPLB / Metro integrations will fill
    in the real ``snapshot`` body when they need it.
    """

    def __init__(self, metadata_source: Any):
        self._metadata_source = metadata_source

    def snapshot(self, layer_id: int) -> PlacementSnapshot:
        raise NotImplementedError(
            "Convert SGLang ExpertLocationMetadata / ExpertLocationDispatchInfo "
            "into PlacementSnapshot when an L2 algorithm needs it."
        )
