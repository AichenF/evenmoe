"""SGLang-side runtime for the EPLB L1 planner.

After the Phase 1.5 inversion, this module **does not import sglang at any
level** -- not at module load, not at call time, not even via
``TYPE_CHECKING``. The adapter exposes two pure APIs that take plain
scalars and tensors:

* :meth:`SGLangEPLBRuntime.drain_stats` -- snapshot-and-reset the internal
  recorder, returning a dict of plain tensors / scalars.
* :meth:`SGLangEPLBRuntime.compute_placement` -- pure computation: plain
  ``logical_count`` + scalars in, ``physical_to_logical_map`` and
  ``logical_to_all_physical_map`` tensors out.

SGLang code is responsible for all sglang-shape work on either side of
those two calls:

* Extracting plain scalars from ``server_args`` / ``model_config`` /
  ``ExpertLocationMetadata`` before calling ``compute_placement``.
* Calling ``torch.distributed.all_reduce`` on the drained ``logical_count``
  before passing it back into ``compute_placement`` (cross-rank
  aggregation is rightly a framework concern).
* Calling ``ExpertLocationMetadata._init_raw(...)`` on the result tensors
  to write back into sglang's authoritative state.

This shape is reusable across frameworks: vLLM / TRT-LLM adapters call
the same ``compute_placement`` with their own scalar extraction.
"""

from __future__ import annotations

from typing import Optional

from torch import Tensor

from moe_load_balancer.core.l1 import PlacementPlanner
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
    """Stateful facade for sglang's EPLB integration.

    Holds one :class:`EPLBRecorder` and one :class:`EPLBPlacementProvider`
    so the per-forward push path (sglang ``_StatAccumulator.append`` ->
    bridge -> recorder) and the periodic rebalance path (sglang
    ``EPLBManager.rebalance`` -> :meth:`drain_stats` ->
    :meth:`compute_placement`) share state through the same instance.
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
    # Exposed handles
    # ------------------------------------------------------------------

    @property
    def recorder(self) -> EPLBRecorder:
        """The shared EPLBRecorder. SGLang's stats bridge feeds this;
        :meth:`drain_stats` reads from the same recorder."""
        return self._recorder

    @property
    def provider(self) -> EPLBPlacementProvider:
        """The shared EPLBPlacementProvider. L2 routing reads from this
        once Phase 2 L2 migration lands."""
        return self._provider

    @property
    def rebalance_policy(self) -> RebalancePolicyConfig:
        return self._rebalance_policy

    def get_stats_bridge(self, *, rank: int = 0):
        """Convenience constructor for the sglang-facing bridge.

        Returns a :class:`SGLangStatsBridge` wired to this runtime's
        recorder. SGLang's ``_StatAccumulator.__init__`` calls this once.
        """
        from moe_load_balancer.adapters.sglang.eplb.stats_bridge import (
            SGLangStatsBridge,
        )

        return SGLangStatsBridge(self._recorder, rank=rank)

    # ------------------------------------------------------------------
    # The two-call rebalance API (sglang calls these from EPLBManager)
    # ------------------------------------------------------------------

    def drain_stats(self) -> Optional[dict]:
        """Drain the internal recorder atomically.

        Returns ``None`` if no observations have been recorded yet.
        Otherwise returns::

            {
                "logical_count": Tensor,                      # per-rank view
                "average_utilization_rate": float | None,
            }

        The ``logical_count`` is not cross-rank reduced -- callers that
        need a global view must call ``torch.distributed.all_reduce``
        themselves before feeding it into :meth:`compute_placement`.
        """
        stats = self._recorder.collect()
        if stats.logical_count is None:
            return None
        return {
            "logical_count": stats.logical_count,
            "average_utilization_rate": stats.average_utilization_rate,
        }

    def drain_for_rebalance(
        self,
        *,
        observed_utilization_rate: Optional[float] = None,
    ) -> Optional[dict]:
        """Drain stats and apply the rebalance-skip policy in one call.

        Combines :meth:`drain_stats` with the skip-gate decision (MLB owns
        the threshold; sglang owns the metric). Used by sglang
        ``EPLBManager.rebalance`` as the single entry point for the
        periodic-rebalance path.

        Returns ``None`` if **either**:

        * the recorder has not observed any stats yet, OR
        * ``observed_utilization_rate`` exceeds
          ``rebalance_policy.min_utilization_threshold`` (current placement
          is already good enough; skip the rebalance to save the
          all_reduce + algorithm + hot-swap cost).

        Otherwise returns the same dict shape as :meth:`drain_stats`::

            {
                "logical_count": Tensor,                    # per-rank view
                "average_utilization_rate": float | None,
            }

        The caller cannot distinguish "no data" from "skipped by
        threshold" -- both yield a no-op rebalance. If finer-grained
        logging is needed, call :meth:`drain_stats` directly and check the
        threshold yourself.

        Args:
          observed_utilization_rate: Current windowed GPU expert
            utilization rate (typically computed by the framework's own
            metric path; MLB does not compute it). Pass ``None`` if the
            metric is unavailable -- the skip-gate degrades to
            "always rebalance" in that case.
        """
        stats = self._recorder.collect()
        if stats.logical_count is None:
            return None

        threshold = self._rebalance_policy.min_utilization_threshold
        if (
            threshold is not None
            and observed_utilization_rate is not None
            and observed_utilization_rate > threshold
        ):
            return None  # current placement already balanced; skip

        return {
            "logical_count": stats.logical_count,
            "average_utilization_rate": observed_utilization_rate,
        }

    def compute_placement(
        self,
        *,
        logical_count: Tensor,
        num_physical_experts: int,
        num_local_physical_experts: int,
        num_groups: int,
        num_nodes: int,
        algorithm: str = "auto",
        active_ranks: Optional[Tensor] = None,
    ) -> Optional[dict]:
        """Pure compute: produce a new EPLB placement from a logical
        token-count tensor + scalars.

        All inputs are plain tensors / Python scalars; no sglang-shape
        objects. The output is a plain dict of result tensors that the
        caller writes back into framework-specific state.

        Args:
          logical_count: ``[num_layers, num_logical_experts]`` already
            cross-rank reduced.
          num_physical_experts / num_local_physical_experts /
            num_groups / num_nodes: EP layout scalars.
          algorithm: ``"auto"`` or any ``EplbAlgorithm`` name (matches
            sglang's ``--eplb-algorithm`` flag).
          active_ranks: optional ``[ep_size]`` mask for
            ``elasticity_aware_*`` variants.

        Returns ``None`` if the planner is disabled or skipped; otherwise::

            {
                "physical_to_logical_map":      Tensor,
                "logical_to_all_physical_map":  Tensor,
            }

        Side effect: publishes the resulting plan to the internal
        :class:`EPLBPlacementProvider`, so L2 routing reads see the new
        plan on subsequent ``snapshot()`` calls.
        """
        request = PlacementRequest(
            stats=PlacementStats(logical_count=logical_count),
            policy=PlacementPolicyConfig(
                name="eplb",
                metadata={
                    "algorithm": algorithm,
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

        self._provider.update_with(plan)
        return {
            "physical_to_logical_map": plan.metadata["physical_to_logical_map"],
            "logical_to_all_physical_map": plan.metadata["logical_to_all_physical_map"],
        }


# ----------------------------------------------------------------------
# Module-level default runtime registry
# ----------------------------------------------------------------------
#
# SGLang's _StatAccumulator.__init__ calls set_default_runtime(...) once
# with its configured instance; sglang's EPLBManager.rebalance() (and
# any future direct callers) use get_default_runtime() to find that
# instance. Not an autonomous singleton -- sglang owns construction;
# MLB only provides the storage slot.

_default_runtime: Optional[SGLangEPLBRuntime] = None


def set_default_runtime(runtime: SGLangEPLBRuntime) -> None:
    """Register *runtime* as the default for `get_default_runtime`."""
    global _default_runtime
    _default_runtime = runtime


def get_default_runtime() -> SGLangEPLBRuntime:
    """Return the registered default runtime.

    Raises :class:`RuntimeError` if no runtime has been registered yet
    via :func:`set_default_runtime`. There is **no silent lazy default**:
    in production sglang's ``_StatAccumulator.__init__`` (or the
    cold-start ``init_by_eplb`` path) is responsible for registering a
    properly-configured runtime; in tests, call
    ``set_default_runtime(SGLangEPLBRuntime())`` explicitly.

    Rationale: a lazy default would silently mask wiring bugs — if
    ``_StatAccumulator`` failed to register due to an upstream ordering
    issue, ``init_by_eplb`` would happily publish to an unconfigured
    runtime whose ``RebalancePolicyConfig`` doesn\'t match
    ``server_args``, and downstream L2 readers would see plans that
    never re-appear after ``_StatAccumulator`` later overwrites the
    slot.
    """
    global _default_runtime
    if _default_runtime is None:
        raise RuntimeError(
            "No default SGLangEPLBRuntime registered. "
            "sglang.srt.eplb.expert_distribution._StatAccumulator.__init__ "
            "or sglang.srt.eplb.expert_location.ExpertLocationMetadata.init_by_eplb "
            "must call set_default_runtime(...) before any caller invokes "
            "get_default_runtime(). For tests, register an unconfigured "
            "instance explicitly: set_default_runtime(SGLangEPLBRuntime())."
        )
    return _default_runtime


def ensure_default_runtime(
    *,
    rebalance_policy=None,
    planner=None,
    recorder=None,
    provider=None,
) -> SGLangEPLBRuntime:
    """First-caller-wins helper: return the registered runtime if any,
    otherwise build one with the given kwargs and register it.

    Both ``_StatAccumulator.__init__`` and the ``init_by_eplb`` static
    method use this to avoid clobbering each other's instance when
    sglang's startup order varies. The first one to run installs the
    configured runtime; the second one sees it and reuses.
    """
    global _default_runtime
    if _default_runtime is not None:
        return _default_runtime
    _default_runtime = SGLangEPLBRuntime(
        planner=planner,
        rebalance_policy=rebalance_policy,
        recorder=recorder,
        provider=provider,
    )
    return _default_runtime


def clear_default_runtime() -> None:
    """Reset the default runtime. Useful for tests."""
    global _default_runtime
    _default_runtime = None
