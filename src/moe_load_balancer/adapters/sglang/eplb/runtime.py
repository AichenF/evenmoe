"""SGLang adapter for the EPLB L1 planner.

Two surfaces:

* **Static-method shim**: sglang's
  ``ExpertLocationMetadata.init_by_eplb(server_args, model_config, logical_count)``
  redirects through :meth:`SGLangEPLBRuntime.init_by_eplb`. This is the
  cold-start path -- called once at server startup with the .pt file's
  ``logical_count``.

* **Stateful facade**: a single ``SGLangEPLBRuntime`` instance owns the
  framework-neutral pieces (:class:`EPLBRecorder`,
  :class:`EPLBPlacementProvider`, :class:`RebalancePolicyConfig`) that
  sglang's ``_StatAccumulator`` and ``EPLBManager`` share across the
  per-forward push path and the periodic rebalance path. Sglang server
  startup builds the instance with deployment-specific overrides and
  blesses it via :func:`set_default_runtime`; the static-method shim
  picks up that same instance.

The runtime is the L1 analogue of
:class:`moe_load_balancer.adapters.sglang.waterfill.WaterfillRuntime`: it
converts SGLang-specific objects into framework-neutral types
(:class:`PlacementRequest`, :class:`PlacementStats`), calls the
framework-neutral planner, publishes the resulting plan to the framework-
neutral provider so L2 can read it, then writes back through SGLang's
``ExpertLocationMetadata._init_raw`` for sglang-specific fields.

Equivalence guarantee: as long as the planner produces the same three
EPLB output tensors as upstream ``eplb_algorithms.rebalance_experts``,
the resulting ``ExpertLocationMetadata`` is bit-identical to the one
upstream ``init_by_eplb`` would have produced.
"""

from __future__ import annotations

from typing import Optional

import torch

from moe_load_balancer.core.l1 import PlacementPlanner, StatsCollector
from moe_load_balancer.core.types import (
    PlacementPolicyConfig,
    PlacementRequest,
    PlacementStats,
    RebalancePolicyConfig,
)
from moe_load_balancer.policies.l1.eplb import DeepSeekEPLBPlanner
from moe_load_balancer.policies.l1.eplb.provider import EPLBPlacementProvider
from moe_load_balancer.policies.l1.eplb.recorder import EPLBRecorder


class SGLangEPLBRuntime:
    """SGLang-side runtime for the EPLB L1 planner.

    Stateful facade -- holds one :class:`EPLBRecorder` and one
    :class:`EPLBPlacementProvider` so the per-forward push path
    (``_StatAccumulator.append`` -> bridge -> recorder) and the periodic
    rebalance path (``EPLBManager.rebalance`` -> runtime -> planner ->
    provider) share state through the same instance.

    Args:
      planner: Optional override for the L1 planner. Defaults to
        :class:`DeepSeekEPLBPlanner`.
      rebalance_policy: Optional MLB defaults override. Used to construct
        the internal recorder; framework adapters typically pass an
        instance reflecting sglang ``server_args`` so the window size and
        utilization threshold match per-deployment.
      recorder / provider: Optional pre-constructed instances. Used by
        tests or when sglang model_runner wants to own these objects
        directly and inject them.
    """

    def __init__(
        self,
        *,
        planner: Optional[PlacementPlanner] = None,
        rebalance_policy: Optional[RebalancePolicyConfig] = None,
        recorder: Optional[EPLBRecorder] = None,
        provider: Optional[EPLBPlacementProvider] = None,
    ):
        self._planner = planner if planner is not None else DeepSeekEPLBPlanner()
        self._rebalance_policy = rebalance_policy or RebalancePolicyConfig()
        self._recorder = recorder if recorder is not None else EPLBRecorder(
            self._rebalance_policy
        )
        self._provider = provider if provider is not None else EPLBPlacementProvider()

    # ------------------------------------------------------------------
    # Exposed handles (sglang and L2 use these)
    # ------------------------------------------------------------------

    @property
    def recorder(self) -> EPLBRecorder:
        """The shared EPLBRecorder. SGLang's stats bridge feeds this;
        :meth:`plan_placement` collects from it."""
        return self._recorder

    @property
    def provider(self) -> EPLBPlacementProvider:
        """The shared EPLBPlacementProvider. L2 routing reads from this
        instead of (or in addition to)
        :class:`SGLangPlacementAdapter`."""
        return self._provider

    @property
    def rebalance_policy(self) -> RebalancePolicyConfig:
        return self._rebalance_policy

    def get_stats_bridge(self, *, rank: int = 0):
        """Convenience constructor for the sglang-facing bridge.

        Returns a :class:`SGLangStatsBridge` wired to this runtime's
        recorder. SGLang's ``_StatAccumulator.__init__`` calls this once
        and stores the bridge for per-forward appends and per-rebalance
        dump.
        """
        # Local import keeps the sglang adapter package internally
        # decoupled (stats_bridge -> recorder, not bridge -> runtime).
        from moe_load_balancer.adapters.sglang.eplb.stats_bridge import (
            SGLangStatsBridge,
        )

        return SGLangStatsBridge(self._recorder, rank=rank)

    # ------------------------------------------------------------------
    # Cold-start / single-shot path -- matches sglang init_by_eplb signature
    # ------------------------------------------------------------------

    def init_by_eplb(
        self,
        server_args,
        model_config,
        logical_count,
    ):
        """Adapter entry point — matches
        ``ExpertLocationMetadata.init_by_eplb`` exactly.

        Computes a placement plan from ``logical_count``, publishes it to
        the shared provider (so L2 sees it on subsequent reads), and
        returns the SGLang ``ExpertLocationMetadata`` for sglang's own
        state. Returns ``None`` if the model config does not expose
        expert-location info.
        """
        # Local imports keep the SGLang dependency contained to the call
        # path, so importing this module never triggers an sglang import
        # for environments that only run the framework-neutral contract
        # tests.
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
                    **(
                        {"active_ranks": active_ranks}
                        if active_ranks is not None
                        else {}
                    ),
                },
            ),
        )

        plan = self._planner.plan(request)
        if plan.metadata.get("skipped"):
            return None

        # Publish the plan to the shared provider so L2 can read it.
        self._provider.update_with(plan)

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
        collector: Optional[StatsCollector] = None,
    ):
        """Periodic-rebalance entry point.

        Instead of receiving a pre-fetched ``logical_count`` tensor (as
        :meth:`init_by_eplb` does), pull stats via the framework-neutral
        :class:`StatsCollector` protocol. Defaults to this runtime's own
        :attr:`recorder` if no collector is passed -- the typical case
        when sglang's ``_StatAccumulator`` has been feeding the same
        recorder via the stats bridge.

        Future iteration: read
        ``stats.average_utilization_rate`` and apply
        ``rebalance_policy.min_utilization_threshold`` for a framework-
        neutral rebalance-skip decision (currently sglang ``EPLBManager``
        owns this check). Once moved, the orchestration becomes framework-
        neutral.

        Returns an ``ExpertLocationMetadata`` (or ``None`` if no stats
        have been observed yet).
        """
        source = collector if collector is not None else self._recorder
        stats = source.collect()
        if stats.logical_count is None:
            return None
        return self.init_by_eplb(server_args, model_config, stats.logical_count)


# ----------------------------------------------------------------------
# Module-level default runtime registry
# ----------------------------------------------------------------------
#
# Sglang's static-method shim
# (``ExpertLocationMetadata.init_by_eplb -> SGLangEPLBRuntime``) cannot
# carry an instance reference -- the static method has no self. Yet we
# want cold-start AND periodic-rebalance AND L2 reads to share state.
#
# Sglang's startup code calls :func:`set_default_runtime` once with the
# fully-configured runtime; subsequent calls to :func:`get_default_runtime`
# (including the one in the static-method shim) pick it up. Falls back to
# an unconfigured default instance for tests / standalone usage.
#
# This is NOT an autonomous singleton -- sglang owns construction and
# blesses the instance. MLB only provides the storage slot.

_default_runtime: Optional[SGLangEPLBRuntime] = None


def set_default_runtime(runtime: SGLangEPLBRuntime) -> None:
    """Register *runtime* as the default for static-method dispatch.

    Sglang server startup calls this once with the configured instance.
    Subsequent ``get_default_runtime()`` calls return the same object.
    """
    global _default_runtime
    _default_runtime = runtime


def get_default_runtime() -> SGLangEPLBRuntime:
    """Return the registered default runtime, creating an unconfigured
    instance on first call if sglang has not registered one yet.

    The fallback path is mostly for tests / contract-only environments;
    production paths always register explicitly via
    :func:`set_default_runtime`.
    """
    global _default_runtime
    if _default_runtime is None:
        _default_runtime = SGLangEPLBRuntime()
    return _default_runtime


def clear_default_runtime() -> None:
    """Reset the default runtime. Useful for tests."""
    global _default_runtime
    _default_runtime = None


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
