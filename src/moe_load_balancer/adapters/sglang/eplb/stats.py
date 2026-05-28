"""SGLang adapter for the framework-neutral :class:`StatsCollector` protocol.

Wraps :class:`sglang.srt.eplb.expert_distribution.ExpertDistributionRecorder`
and exposes its aggregated output as a framework-neutral
:class:`PlacementStats`. This is the L1 counterpart of
:class:`SGLangPlacementAdapter` (which adapts ``ExpertLocationMetadata`` into
``PlacementSnapshot`` for L2 routing).

Field mapping for the canonical ``stat`` / ``stat_approx`` accumulator mode:

+----------------------------------------+---------------------------------------+
| SGLang ``dump_record(output_mode=...)``| ``PlacementStats``                    |
+========================================+=======================================+
| ``logical_count``                      | ``logical_count``                     |
| ``average_utilization_rate_over_window``| ``average_utilization_rate``         |
| ``rank`` (collector-internal)          | -- (dropped)                          |
+----------------------------------------+---------------------------------------+

Construction is dependency-free (no sglang import at module load): if no
recorder is passed in, ``collect`` / ``start`` / ``stop`` / ``recording``
lazily look up sglang's global singleton
``get_global_expert_distribution_recorder()`` at call time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from moe_load_balancer.core.l1 import StatsCollector
from moe_load_balancer.core.types import PlacementStats

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_distribution import ExpertDistributionRecorder


class SGLangStatsCollector(StatsCollector):
    """Framework-neutral view over sglang's ``ExpertDistributionRecorder``.

    Two construction modes:

    * **Explicit recorder** -- ``SGLangStatsCollector(recorder=rec)``. The
      adapter holds the reference and routes all calls to it. Useful when
      a model runner already owns a recorder instance, or for tests that
      pass a synthetic recorder-shaped object.

    * **Global singleton** (default) -- ``SGLangStatsCollector()``. Each
      call resolves sglang's module-level recorder via
      ``get_global_expert_distribution_recorder()``. This matches the
      pattern ``EPLBManager`` uses today and keeps the adapter usable from
      code paths that don't have a direct recorder handle.

    The sglang import is deferred to call time so importing this module
    never triggers sglang import in CPU-only test environments.
    """

    def __init__(
        self,
        recorder: "Optional[ExpertDistributionRecorder]" = None,
    ) -> None:
        self._explicit_recorder = recorder

    def _resolve(self) -> "ExpertDistributionRecorder":
        if self._explicit_recorder is not None:
            return self._explicit_recorder
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )

        return get_global_expert_distribution_recorder()

    @property
    def recording(self) -> bool:
        return bool(self._resolve().recording)

    def start(self) -> None:
        rec = self._resolve()
        if not rec.recording:
            rec.start_record()

    def stop(self) -> None:
        rec = self._resolve()
        if rec.recording:
            rec.stop_record()

    def collect(self) -> PlacementStats:
        payload = self._resolve().dump_record(output_mode="object")
        if payload is None:
            return PlacementStats()
        return PlacementStats(
            logical_count=payload.get("logical_count"),
            average_utilization_rate=payload.get(
                "average_utilization_rate_over_window"
            ),
        )
