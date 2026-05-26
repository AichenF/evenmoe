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

    Implements ``core.l1.PlacementProvider``. The adapter holds a reference
    to a live ``ExpertLocationMetadata`` instance and surfaces
    ``snapshot(layer_id)`` as the framework-neutral read access point. SGLang
    imports live inside this module; downstream L2 policies (LPLB, Metro,
    future Waterfill variants that need placement) consume only the resulting
    ``PlacementSnapshot``.

    The reference is *live*: when sglang's ``EPLBManager`` periodically
    rebalances and rewrites ``ExpertLocationMetadata`` in place, subsequent
    ``snapshot`` calls see the new placement without reconstructing the
    adapter.

    Field mapping (per-layer slice):

    +---------------------------------------+----------------------------------------------------+
    | PlacementSnapshot                     | ExpertLocationMetadata                             |
    +=======================================+====================================================+
    | ``logical_to_physical_candidates``    | ``logical_to_all_physical_map[layer_id]``          |
    | ``logical_to_physical_count``         | ``logical_to_all_physical_map_num_valid[layer_id]``|
    | ``default_physical_for_logical``      | ``logical_to_rank_dispatch_physical_map[layer_id]``|
    |                                       | (``None`` if sglang did not build the table)       |
    | ``physical_to_rank_map``              | derived once: ``arange // num_local_physical``     |
    | ``metadata["physical_to_logical_map"]`` | ``physical_to_logical_map[layer_id]``            |
    +---------------------------------------+----------------------------------------------------+

    Construction is dependency-free (no sglang import, no field access on the
    metadata object), so the protocol-conformance check
    ``isinstance(adapter, PlacementProvider)`` works in CPU-only test environments
    without sglang installed. Field access happens on the first ``snapshot``
    call.
    """

    def __init__(self, metadata: "ExpertLocationMetadata") -> None:
        self._metadata = metadata
        self._physical_to_rank_map: Optional[torch.Tensor] = None

    def snapshot(self, layer_id: int) -> PlacementSnapshot:
        m = self._metadata
        rank_dispatch = m.logical_to_rank_dispatch_physical_map
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
                "physical_to_logical_map": m.physical_to_logical_map[layer_id],
            },
        )

    def _get_physical_to_rank_map(self) -> torch.Tensor:
        """Lazily derive ``physical_to_rank_map`` once and cache it.

        The EP layout (``num_physical_experts``, ``ep_size``) is stable across
        layers and across rebalances, so this is computed once per adapter and
        reused. Built on the same device as ``physical_to_logical_map`` so
        downstream consumers can index without a host↔device hop.
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
