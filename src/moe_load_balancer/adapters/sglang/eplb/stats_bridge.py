"""Bridge between sglang's per-forward stats and MLB's :class:`EPLBRecorder`.

Designed to replace sglang ``_StatAccumulator._global_physical_count_of_buffered_step``
(an internal :class:`_Buffer`) with an MLB-owned recorder, so EPLB's
buffer + physical-to-logical conversion logic lives in MLB instead of
being duplicated.

The bridge exposes two surfaces:

* **Recorder-facing (push)**:
  :meth:`append_per_forward` ingests the per-forward
  ``global_physical_count`` (already all-reduced across EP ranks by sglang)
  and forwards it to the MLB recorder.

* **sglang-facing (pull, for `_StatAccumulator.dump()`)**:
  :meth:`dump_for_sglang` returns the dict shape sglang's ``EPLBManager``
  expects:
  ``{"rank": int, "logical_count": Tensor, "average_utilization_rate_over_window": Optional[float]}``
  -- ``logical_count`` comes from the MLB recorder; ``rank`` and
  ``average_utilization_rate_over_window`` are sglang-side concerns
  (utilization is computed by sglang's existing
  ``_UtilizationRateAccumulatorMixin``; this bridge just passes the value
  through).

The sglang patch is minimal: hand sglang's ``_StatAccumulator`` an
:class:`SGLangStatsBridge` instance at construction; route ``append`` and
``dump`` to the bridge. Utilization rate / Prometheus heatmap stay in
sglang.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from moe_load_balancer.policies.l1.eplb.recorder import EPLBRecorder

if TYPE_CHECKING:
    from torch import Tensor


class SGLangStatsBridge:
    """SGLang-side proxy that replaces ``_StatAccumulator``'s ``_Buffer``.

    Holds a reference to an MLB :class:`EPLBRecorder` (caller-supplied so
    the same recorder instance is shared between the bridge -- which
    sglang feeds -- and the L1 rebalance loop in
    :class:`SGLangEPLBRuntime`, which reads from the same recorder).

    Args:
      recorder: The MLB recorder to forward observations to.
      rank: The EP rank index (sglang's ``self._rank``), included in
        ``dump_for_sglang`` so the dict shape matches what sglang's
        ``EPLBManager.rebalance`` consumes.
    """

    def __init__(self, recorder: EPLBRecorder, rank: int = 0) -> None:
        self._recorder = recorder
        self._rank = int(rank)

    # ------------------------------------------------------------------
    # Push side (sglang _StatAccumulator.append calls this)
    # ------------------------------------------------------------------

    def append_per_forward(
        self,
        global_physical_count: "Tensor",
        physical_to_logical_map: "Tensor",
    ) -> None:
        """Forward one per-forward ``global_physical_count`` to MLB.

        Both tensors come from sglang:

        * ``global_physical_count`` -- per-physical-expert counts produced
          by the single-pass gatherer (e.g. select_experts, deepep_dispatch).
          Already all-reduced across EP ranks in sglang.
        * ``physical_to_logical_map`` -- current placement,
          ``[num_layers, num_physical_experts]``, pulled from
          ``self._expert_location_metadata.physical_to_logical_map``.
        """
        self._recorder.append(
            global_physical_count=global_physical_count,
            physical_to_logical_map=physical_to_logical_map,
        )

    # ------------------------------------------------------------------
    # Pull side (sglang _StatAccumulator.dump calls this)
    # ------------------------------------------------------------------

    def dump_for_sglang(
        self,
        *,
        average_utilization_rate_over_window: Optional[float] = None,
    ) -> dict:
        """Drain the MLB recorder and emit the dict shape sglang's
        ``EPLBManager.rebalance`` expects.

        The ``average_utilization_rate_over_window`` field is *passed
        through* from the sglang-side
        ``_UtilizationRateAccumulatorMixin._get_global_average_utilization_rate()``
        (this bridge does not compute it -- sglang owns the utilization
        path and its Prometheus / log metrics).

        Field mapping:

          MLB PlacementStats             sglang dump dict
          ------------------             ----------------
          logical_count                  "logical_count"
          (from arg)                     "average_utilization_rate_over_window"
          (from constructor)             "rank"
        """
        stats = self._recorder.collect()
        return {
            "rank": self._rank,
            "logical_count": stats.logical_count,
            "average_utilization_rate_over_window": average_utilization_rate_over_window,
        }

    # ------------------------------------------------------------------
    # Lifecycle pass-through (so sglang can drive recorder state too)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Drain the recorder without consuming the result (matches sglang
        ``_StatAccumulator.reset`` semantics)."""
        # collect() drains; discard the result.
        self._recorder.collect()

    @property
    def recorder(self) -> EPLBRecorder:
        """Direct handle to the underlying recorder. Useful when the L1
        rebalance loop (:class:`SGLangEPLBRuntime`) wants to read stats
        without going through the sglang-facing dict shape."""
        return self._recorder
