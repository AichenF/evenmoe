"""SGLang adapters that produce framework-neutral core types.

These are L1 / L2 *generic* providers shared across L2 algorithms. Algorithm-
specific runtime code (Waterfill / LPLB / Metro) lives in dedicated
subpackages alongside their L2 algorithm class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from moe_load_balancer.core.l1 import PlacementProvider
from moe_load_balancer.core.types import PlacementSnapshot

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location import ExpertLocationMetadata


class SGLangPlacementAdapter(PlacementProvider):
    """Per-layer framework-neutral view of sglang's ``ExpertLocationMetadata``.

    Implements ``core.l1.PlacementProvider``. The adapter holds a live
    reference to an ``ExpertLocationMetadata`` instance and surfaces
    ``snapshot(layer_id)`` as the framework-neutral read access point. SGLang
    imports live inside this module; downstream L2 policies (LPLB, LP-CF,
    Metro, future Waterfill variants that need placement) consume only the
    resulting ``PlacementSnapshot``.

    Two complementary invalidation paths are supported:

    * **Live reference** -- when sglang's ``EPLBManager`` rewrites
      ``ExpertLocationMetadata`` in place, subsequent ``snapshot`` calls
      observe the new placement automatically (no adapter rebuild).
    * **Version counter** -- for L2 policies that cache per-layer state
      across calls and need an O(1) invalidation key, the SGLang glue
      calls :meth:`bump_version` after every EPLB rebalance; the counter
      is exposed via ``snapshot.metadata["placement_version"]``.

    Field mapping (per-layer slice):

    +-----------------------------------------+----------------------------------------------------+
    | PlacementSnapshot                       | ExpertLocationMetadata                             |
    +=========================================+====================================================+
    | ``logical_to_physical_candidates``      | ``logical_to_all_physical_map[layer_id]``          |
    | ``logical_to_physical_count``           | ``logical_to_all_physical_map_num_valid[layer_id]``|
    | ``default_physical_for_logical``        | ``logical_to_rank_dispatch_physical_map[layer_id]``|
    |                                         | (``None`` if sglang did not build the table)       |
    | ``physical_to_rank_map``                | derived once: ``arange // num_local_physical``     |
    | ``metadata["placement_version"]``       | per-adapter counter; bump via ``bump_version``     |
    | ``metadata["physical_to_logical_map"]`` | ``physical_to_logical_map[layer_id]``              |
    +-----------------------------------------+----------------------------------------------------+

    Construction is dependency-free (no sglang import, no field access on the
    metadata object), so the protocol-conformance check
    ``isinstance(adapter, PlacementProvider)`` works in CPU-only test
    environments without sglang installed. Field access happens on the first
    ``snapshot`` call. If ``metadata`` is ``None`` at snapshot time, a
    ``RuntimeError`` is raised.
    """

    def __init__(
        self,
        metadata: "Optional[ExpertLocationMetadata]" = None,
        version: int = 0,
    ) -> None:
        self._metadata = metadata
        self._version = int(version)
        self._physical_to_rank_map: Optional[torch.Tensor] = None

    @property
    def version(self) -> int:
        return self._version

    def bump_version(self) -> None:
        self._version += 1

    def snapshot(self, layer_id: int) -> PlacementSnapshot:
        m = self._metadata
        if m is None:
            raise RuntimeError(
                "SGLangPlacementAdapter was constructed without an "
                "ExpertLocationMetadata; set metadata before requesting "
                "snapshots."
            )
        rank_dispatch = getattr(m, "logical_to_rank_dispatch_physical_map", None)
        default_physical = (
            rank_dispatch[layer_id] if rank_dispatch is not None else None
        )
        return PlacementSnapshot(
            layer_id=layer_id,
            num_logical_experts=m.num_logical_experts,
            num_physical_experts=m.num_physical_experts,
            logical_to_physical_candidates=m.logical_to_all_physical_map[layer_id],
            logical_to_physical_count=m.logical_to_all_physical_map_num_valid[layer_id],
            default_physical_for_logical=default_physical,
            physical_to_rank_map=self._get_physical_to_rank_map(),
            metadata={
                "placement_version": self._version,
                "physical_to_logical_map": m.physical_to_logical_map[layer_id],
            },
        )

    def _get_physical_to_rank_map(self) -> torch.Tensor:
        """Lazily derive ``physical_to_rank_map`` once and cache it.

        The EP layout (``num_physical_experts``, ``ep_size``) is stable across
        layers and across rebalances, so this is computed once per adapter and
        reused. Built on the same device as ``physical_to_logical_map`` so
        downstream consumers can index without a host<->device hop.
        """
        if self._physical_to_rank_map is None:
            m = self._metadata
            num_physical = m.num_physical_experts
            num_local_physical = m.num_local_physical_experts
            device = m.physical_to_logical_map.device
            self._physical_to_rank_map = (
                torch.arange(num_physical, dtype=torch.int64, device=device)
                // num_local_physical
            )
        return self._physical_to_rank_map
