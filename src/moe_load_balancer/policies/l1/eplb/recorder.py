"""Framework-neutral EPLB recorder.

Owns the rolling window of per-forward ``global_physical_count`` observations
and produces a framework-neutral :class:`PlacementStats` snapshot on demand.

This is the L1 record half of the EPLB loop. Framework adapters (e.g.
:class:`SGLangStatsBridge`) feed per-forward observations via :meth:`append`;
the rebalance orchestration calls :meth:`collect` once per rebalance window
to obtain the aggregated :class:`PlacementStats` that the planner consumes.

Implements :class:`moe_load_balancer.core.l1.StatsCollector`.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import torch
from torch import Tensor

from moe_load_balancer.core.l1 import StatsCollector
from moe_load_balancer.core.types import PlacementStats, RebalancePolicyConfig


class EPLBRecorder(StatsCollector):
    """Rolling-window recorder for EPLB load statistics.

    Each :meth:`append` call ingests one forward pass worth of
    ``global_physical_count`` (already all-reduced across EP ranks; produced
    by the framework adapter). The recorder buffers the most recent
    ``recording_window_size`` observations as a deque.

    :meth:`collect` aggregates the buffered window:

    1. Sums ``global_physical_count`` across the window dimension to a
       ``[num_layers, num_physical_experts]`` tensor.
    2. Converts physical->logical via the most recent
       ``physical_to_logical_map`` (captured at :meth:`append` time) using
       the canonical DeepSeek mapping
       (``logical_count[layer, l] = sum(physical_count[layer, p] for p
       where physical_to_logical_map[layer, p] == l)``).
    3. Returns a :class:`PlacementStats` with ``logical_count`` filled and
       ``average_utilization_rate`` left as ``None`` (the framework
       adapter populates it if available; EPLBRecorder does not compute
       utilization).

    The recorder is framework-agnostic: no sglang import, no distributed
    communication. The communication (all-reduce across EP ranks) belongs
    to the framework adapter producing the ``global_physical_count``.
    """

    def __init__(
        self,
        rebalance_policy: Optional[RebalancePolicyConfig] = None,
    ) -> None:
        self._policy = rebalance_policy or RebalancePolicyConfig()
        self._buffer: Deque[Tensor] = deque(
            maxlen=self._policy.recording_window_size
        )
        self._latest_physical_to_logical_map: Optional[Tensor] = None
        self._recording = True

    # ------------------------------------------------------------------
    # StatsCollector Protocol
    # ------------------------------------------------------------------

    @property
    def recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        self._recording = True

    def stop(self) -> None:
        self._recording = False

    def collect(self) -> PlacementStats:
        if not self._buffer or self._latest_physical_to_logical_map is None:
            return PlacementStats()

        # Stack the window into a single tensor and sum across the window
        # axis: result shape [num_layers, num_physical_experts].
        stacked = torch.stack(tuple(self._buffer), dim=0)
        physical_count = stacked.sum(dim=0)

        logical_count = _physical_to_logical(
            physical_count, self._latest_physical_to_logical_map
        )

        # Drain the buffer so the next window starts fresh. This is a
        # *single-threaded* snapshot-and-reset: append + collect are
        # NOT guarded by a lock, so concurrent calls would race. SGLang's
        # scheduler invokes per-forward append and periodic collect on
        # the same event-loop thread per rank, which makes the race
        # impossible in production; tests run the sequence linearly.
        self._buffer.clear()

        return PlacementStats(
            logical_count=logical_count,
            physical_count=physical_count,
            average_utilization_rate=None,
        )

    # ------------------------------------------------------------------
    # Append API for framework adapters
    # ------------------------------------------------------------------

    def append(
        self,
        global_physical_count: Tensor,
        physical_to_logical_map: Tensor,
    ) -> None:
        """Ingest one forward pass of stats.

        Args:
          global_physical_count: ``[num_layers, num_physical_experts]``
            per-physical-expert token counts, already all-reduced across
            EP ranks by the framework adapter.
          physical_to_logical_map: ``[num_layers, num_physical_experts]``
            current placement, used at :meth:`collect` time to convert
            physical counts into logical counts. The *most recent* map at
            collect time wins -- placement updates between collects are
            applied with the freshest map.
        """
        if not self._recording:
            return

        # Defensive snapshot: sglang gatherers (e.g.
        # ``_LayerBasedGpuSinglePassGatherer``) reuse a single
        # ``self._data`` tensor across forward passes and zero it in
        # place on ``reset()`` before the next pass starts. If we put
        # that tensor reference straight into the deque, every entry
        # aliases the same memory and the next ``reset()`` wipes our
        # whole window. We must take an independent copy.
        #
        # ``.to(device, copy=True)`` does both the device co-location
        # (CPU gatherer -> GPU placement; needed for sglang
        # ``all_reduce(logical_count)`` under NCCL) AND the independent
        # snapshot in one shot. If devices match the result is still a
        # fresh tensor; if they differ ``.to`` allocates on the target
        # device. Cost per forward: one [num_layers,num_physical] int32
        # copy (~tens of KB on typical EP configs).
        global_physical_count = global_physical_count.to(
            physical_to_logical_map.device, copy=True
        )
        self._buffer.append(global_physical_count)
        # Keep only the latest placement map -- per-step storage would
        # be wasteful and ambiguous (the active placement at collect
        # time is what matters for the conversion).
        self._latest_physical_to_logical_map = physical_to_logical_map

    # ------------------------------------------------------------------
    # Introspection (useful for tests + sglang dump_for_sglang)
    # ------------------------------------------------------------------

    def window_size(self) -> int:
        """Current number of buffered observations (<= recording_window_size)."""
        return len(self._buffer)


def _physical_to_logical(
    physical_count: Tensor,
    physical_to_logical_map: Tensor,
) -> Tensor:
    """Convert ``[L, P]`` physical counts -> ``[L, num_logical]`` logical
    counts using the placement map.

    For each layer, ``logical_count[l] = sum(physical_count[p] for p where
    physical_to_logical_map[p] == l)``.

    Mirrors sglang's
    ``_convert_global_physical_count_to_logical_count`` so the resulting
    tensor is bit-identical for the same inputs.

    The two inputs may live on different devices (sglang gatherers in
    DeepEP normal mode produce ``global_physical_count`` on CPU while
    ``ExpertLocationMetadata.physical_to_logical_map`` is on the EP
    device); we co-locate them on ``physical_count``'s device so the
    accumulation happens wherever sglang's gatherer chose to land its
    output.
    """
    num_layers, num_physical = physical_count.shape
    if physical_to_logical_map.device != physical_count.device:
        physical_to_logical_map = physical_to_logical_map.to(physical_count.device)
    num_logical = int(physical_to_logical_map.max().item()) + 1
    logical = torch.zeros(
        (num_layers, num_logical),
        dtype=physical_count.dtype,
        device=physical_count.device,
    )
    # scatter_add along the physical axis -> logical axis
    logical.scatter_add_(
        dim=1,
        index=physical_to_logical_map.to(torch.int64),
        src=physical_count,
    )
    return logical
