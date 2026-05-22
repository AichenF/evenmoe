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
    """Aliases SGLang ``ExpertLocationMetadata`` into ``PlacementSnapshot``.

    Per-layer tensors in the snapshot reference the metadata's live buffers
    (no copy), so in-place EPLB updates are reflected automatically. Policies
    that cache per-layer state across calls key invalidation on
    ``snapshot.metadata["placement_version"]``; the version counter is bumped
    by the SGLang glue via :meth:`bump_version` after every EPLB rebalance.
    """

    def __init__(self, metadata_source: Any, version: int = 0):
        self._metadata_source = metadata_source
        self._version = int(version)

    @property
    def version(self) -> int:
        return self._version

    def bump_version(self) -> None:
        self._version += 1

    def snapshot(self, layer_id: int) -> PlacementSnapshot:
        meta = self._metadata_source
        if meta is None:
            raise RuntimeError(
                "SGLangPlacementAdapter was constructed without an "
                "ExpertLocationMetadata; set metadata_source before requesting "
                "snapshots."
            )
        log2phy = meta.logical_to_all_physical_map[layer_id]
        log_counts = meta.logical_to_all_physical_map_num_valid[layer_id]
        rank_dispatch_map = getattr(meta, "logical_to_rank_dispatch_physical_map", None)
        default_map = (
            rank_dispatch_map[layer_id] if rank_dispatch_map is not None else None
        )
        return PlacementSnapshot(
            layer_id=layer_id,
            num_logical_experts=meta.num_logical_experts,
            num_physical_experts=meta.num_physical_experts,
            logical_to_physical_candidates=log2phy,
            logical_to_physical_count=log_counts,
            default_physical_for_logical=default_map,
            physical_to_rank_map=None,
            metadata={"placement_version": self._version},
        )
