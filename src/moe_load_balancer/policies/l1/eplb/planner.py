"""DeepSeek EPLB L1 planner — framework-neutral.

Wraps :mod:`moe_load_balancer.policies.l1.eplb.algorithms` behind the
:class:`moe_load_balancer.core.l1.PlacementPlanner` protocol. Style mirrors
:class:`moe_load_balancer.policies.l2.waterfill.WaterfillL2Router`: stateless
class, scalar inputs come from ``request.policy.metadata``, tensor inputs
from ``request.stats``, no framework-specific types.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from moe_load_balancer.core.l1 import PlacementPlanner
from moe_load_balancer.core.types import (
    PlacementPlan,
    PlacementPolicyConfig,
    PlacementRequest,
    PlacementSnapshot,
)
from moe_load_balancer.policies.l1.eplb.algorithms import (
    EplbAlgorithm,
    compute_algorithm,
    rebalance_experts,
)

_REQUIRED_METADATA_SCALARS = (
    "num_physical_experts",
    "num_local_physical_experts",
    "num_groups",
    "num_nodes",
)


class DeepSeekEPLBPlanner(PlacementPlanner):
    """L1 placement planner backed by the DeepSeek EPLB algorithm family.

    Implements ``core.PlacementPlanner``. ``plan(request)`` reads scalars
    from ``request.policy.metadata`` (``num_physical_experts``,
    ``num_local_physical_experts``, ``num_groups``, ``num_nodes``,
    optional ``algorithm`` and ``active_ranks``) and the
    ``logical_count`` tensor from ``request.stats``, dispatches to the
    matching variant via the vendored algorithms package, and returns a
    ``PlacementPlan`` whose:

    - ``snapshots`` is a per-layer framework-neutral ``PlacementSnapshot``
      view, suitable for L2 consumers.
    - ``metadata`` carries the raw three EPLB output tensors
      (``physical_to_logical_map``, ``logical_to_all_physical_map``,
      ``expert_count``) verbatim, so framework adapters can write back to
      native placement structures (e.g. SGLang ``ExpertLocationMetadata``)
      without round-tripping through ``snapshots``.

    Equivalence with sglang: a request whose ``policy.metadata`` mirrors
    sglang's ``_init_common`` scalars and whose ``stats.logical_count`` is
    the same tensor that sglang's ``init_by_eplb`` would have passed to
    ``eplb_algorithms.rebalance_experts`` produces a bit-identical plan,
    because the algorithm body is the same code path.
    """

    def __init__(self, *, default_algorithm: str = "auto"):
        """``default_algorithm`` is used only when
        ``request.policy.metadata`` does not provide an explicit
        ``"algorithm"`` key. Acceptable values match SGLang's
        ``--eplb-algorithm`` flag: ``"auto"`` or any ``EplbAlgorithm`` name.
        """

        self._default_algorithm = default_algorithm

    def plan(self, request: PlacementRequest) -> PlacementPlan:
        policy = request.policy
        if policy.is_disabled:
            return PlacementPlan(snapshots={}, metadata={"skipped": True})

        m = policy.metadata
        missing = [k for k in _REQUIRED_METADATA_SCALARS if k not in m]
        if missing:
            raise ValueError(
                "DeepSeekEPLBPlanner requires policy.metadata to carry "
                f"{_REQUIRED_METADATA_SCALARS}; missing {missing}."
            )

        logical_count = request.stats.logical_count
        if logical_count is None:
            raise ValueError(
                "DeepSeekEPLBPlanner requires request.stats.logical_count."
            )
        if not isinstance(logical_count, Tensor):
            logical_count = torch.tensor(logical_count)
        if logical_count.dim() == 2:
            # promote [layers, num_logical] to [1, layers, num_logical] so the
            # `tokens_per_expert.sum(dim=0)` inside the dispatcher is a no-op.
            logical_count = logical_count.unsqueeze(0)

        raw_algorithm = m.get("algorithm", self._default_algorithm)
        algorithm = compute_algorithm(
            raw_algorithm=raw_algorithm,
            num_groups=m["num_groups"],
            num_nodes=m["num_nodes"],
        )

        active_ranks: Optional[Tensor] = m.get("active_ranks")

        phy2log, log2allphy, expert_count = rebalance_experts(
            tokens_per_expert=logical_count,
            num_physical_experts=m["num_physical_experts"],
            num_local_physical_experts=m["num_local_physical_experts"],
            num_groups=m["num_groups"],
            num_nodes=m["num_nodes"],
            algorithm=algorithm,
            active_ranks=active_ranks,
        )

        snapshots = _pack_snapshots(phy2log, log2allphy, expert_count)
        return PlacementPlan(
            snapshots=snapshots,
            metadata={
                "physical_to_logical_map": phy2log,
                "logical_to_all_physical_map": log2allphy,
                "expert_count": expert_count,
                "algorithm": algorithm.name,
            },
        )


def _pack_snapshots(
    phy2log: Tensor,
    log2allphy: Tensor,
    expert_count: Tensor,
) -> dict[int, PlacementSnapshot]:
    """Pack EPLB output tensors into per-layer framework-neutral snapshots.

    Cheap CPU-side fan-out: each layer gets a slice view, no copy. The
    adapter typically ignores this and uses ``PlacementPlan.metadata`` for
    writeback, but L2 consumers (Waterfill / LPLB / Metro) read this.
    """

    num_layers, num_physical = phy2log.shape
    _, num_logical, _ = log2allphy.shape
    out: dict[int, PlacementSnapshot] = {}
    for layer_id in range(num_layers):
        out[layer_id] = PlacementSnapshot(
            layer_id=layer_id,
            num_logical_experts=num_logical,
            num_physical_experts=num_physical,
            logical_to_physical_candidates=log2allphy[layer_id],
            logical_to_physical_count=expert_count[layer_id],
        )
    return out
