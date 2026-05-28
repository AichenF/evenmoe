"""SGLang adapter for the EPLB L1 planner.

The runtime exposes a single method, :meth:`SGLangEPLBRuntime.init_by_eplb`,
with the exact signature SGLang's
``ExpertLocationMetadata.init_by_eplb(server_args, model_config, logical_count)``
expects, so the upstream call site becomes a one-line redirect through this
adapter:

.. code-block:: python

    # python/sglang/srt/eplb/expert_location.py
    from moe_load_balancer.adapters.sglang.eplb import SGLangEPLBRuntime

    @staticmethod
    def init_by_eplb(server_args, model_config, logical_count):
        return SGLangEPLBRuntime().init_by_eplb(server_args, model_config, logical_count)

The runtime is the analogue of
:class:`moe_load_balancer.adapters.sglang.waterfill.WaterfillRuntime` for L1
— it converts SGLang-specific objects into a framework-neutral
``PlacementRequest``, calls the framework-neutral planner, then writes the
plan back through SGLang's ``ExpertLocationMetadata._init_raw`` so all
downstream sglang-specific fields (``logical_to_rank_dispatch_physical_map``,
device placement, CPU mirrors) are produced by the upstream code path.

Equivalence guarantee: as long as the planner produces the same three EPLB
output tensors as upstream ``eplb_algorithms.rebalance_experts``, the
resulting ``ExpertLocationMetadata`` is bit-identical to the one upstream
``init_by_eplb`` would have produced.
"""

from __future__ import annotations

from typing import Optional

import torch

from moe_load_balancer.core.l1 import PlacementPlanner, StatsCollector
from moe_load_balancer.core.types import (
    PlacementPolicyConfig,
    PlacementRequest,
    PlacementStats,
)
from moe_load_balancer.policies.l1.eplb import DeepSeekEPLBPlanner


class SGLangEPLBRuntime:
    """SGLang-side runtime for the EPLB L1 planner.

    Default-constructs a :class:`DeepSeekEPLBPlanner`; pass a custom planner
    to swap in CRAFT / MoETuner / etc. without touching the SGLang glue.
    """

    def __init__(self, *, planner: Optional[PlacementPlanner] = None):
        self._planner = planner if planner is not None else DeepSeekEPLBPlanner()

    def init_by_eplb(
        self,
        server_args,
        model_config,
        logical_count,
    ):
        """Adapter entry point — matches
        ``ExpertLocationMetadata.init_by_eplb`` exactly.

        Returns an SGLang ``ExpertLocationMetadata`` or ``None`` if the model
        config does not expose expert-location info.
        """

        # Local imports keep the SGLang dependency contained to the call path,
        # so importing this module never triggers an sglang import for
        # environments that only run the framework-neutral contract tests.
        from sglang.srt.eplb.expert_location import ExpertLocationMetadata

        if not isinstance(logical_count, torch.Tensor):
            logical_count = torch.tensor(logical_count)
        if len(logical_count.shape) == 2:
            logical_count = logical_count.unsqueeze(0)
        logical_count = logical_count.to(server_args.device)

        common = ExpertLocationMetadata._init_common(server_args, model_config)
        if common is None:
            return None

        num_groups = common["model_config_for_expert_location"].num_groups
        num_nodes = server_args.nnodes
        num_physical_experts = common["num_physical_experts"]
        num_local_physical_experts = num_physical_experts // common["ep_size"]

        active_ranks = _maybe_active_ranks(server_args)

        request = PlacementRequest(
            stats=PlacementStats(logical_count=logical_count),
            policy=PlacementPolicyConfig(
                name="eplb",
                metadata={
                    "algorithm": server_args.eplb_algorithm,
                    "num_physical_experts": num_physical_experts,
                    "num_local_physical_experts": num_local_physical_experts,
                    "num_groups": num_groups,
                    "num_nodes": num_nodes,
                    **({"active_ranks": active_ranks} if active_ranks is not None else {}),
                },
            ),
        )

        plan = self._planner.plan(request)
        if plan.metadata.get("skipped"):
            return None

        physical_to_logical_map = plan.metadata["physical_to_logical_map"]
        logical_to_all_physical_map = plan.metadata["logical_to_all_physical_map"]

        return ExpertLocationMetadata._init_raw(
            server_args=server_args,
            ep_size=common["ep_size"],
            physical_to_logical_map=physical_to_logical_map.to(server_args.device),
            logical_to_all_physical_map=logical_to_all_physical_map.to(
                server_args.device
            ),
        )

    def plan_placement(
        self,
        server_args,
        model_config,
        collector: StatsCollector,
    ):
        """Periodic-rebalance entry point.

        The natural counterpart of :meth:`init_by_eplb` for the rebalance
        loop: instead of receiving a pre-fetched ``logical_count`` tensor,
        pull stats via the framework-neutral
        :class:`~moe_load_balancer.core.l1.StatsCollector` protocol.

        Today this is a thin wrapper around :meth:`init_by_eplb`. Once
        sglang's ``EPLBManager.rebalance`` is rewired to call through this
        method, the rebalance-skip decision (based on
        ``stats.average_utilization_rate``) will live here instead of in
        ``EPLBManager``, so the orchestration becomes framework-neutral.

        Returns an ``ExpertLocationMetadata`` (or ``None`` if the
        collector has not yet observed any stats).
        """
        stats = collector.collect()
        if stats.logical_count is None:
            return None
        return self.init_by_eplb(server_args, model_config, stats.logical_count)


def _maybe_active_ranks(server_args) -> Optional[torch.Tensor]:
    """Pull ``active_ranks`` from SGLang's ``ElasticEPStateManager``, if any.

    The elasticity_aware variants need this; other variants ignore it.
    Returns ``None`` if Elastic-EP isn't configured, matching SGLang's own
    branching inside ``eplb_algorithms.rebalance_experts``.
    """

    algorithm = getattr(server_args, "eplb_algorithm", "auto")
    if not str(algorithm).startswith("elasticity_aware"):
        return None
    try:
        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
    except ImportError:
        return None
    instance = ElasticEPStateManager.instance()
    if instance is not None:
        return instance.active_ranks
    return ElasticEPStateManager.healthy_rank_state()
