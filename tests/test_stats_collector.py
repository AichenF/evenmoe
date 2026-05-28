"""Tests for the framework-neutral StatsCollector protocol and the
SGLang adapter that satisfies it.

These tests intentionally do not import sglang itself: the SGLang adapter
takes an explicit recorder-shaped object, and the global-singleton path is
exercised via monkey-patching ``get_global_expert_distribution_recorder``
where needed. CPU-only test environments without sglang installed pass.
"""

from __future__ import annotations

import pytest

from moe_load_balancer import PlacementStats, StatsCollector
from moe_load_balancer.adapters.sglang.eplb import SGLangStatsCollector


class _RecorderShaped:
    """Minimal stand-in for sglang's ExpertDistributionRecorder.

    Exposes only the attributes/methods the adapter actually touches:
    ``recording`` (property/attribute), ``start_record``, ``stop_record``,
    ``dump_record(output_mode="object")``.
    """

    def __init__(self, payload=None, recording=False):
        self.recording = recording
        self._payload = payload
        self.start_count = 0
        self.stop_count = 0
        self.dump_count = 0
        self.last_output_mode = None

    def start_record(self):
        self.recording = True
        self.start_count += 1

    def stop_record(self):
        self.recording = False
        self.stop_count += 1

    def dump_record(self, output_mode="object"):
        self.dump_count += 1
        self.last_output_mode = output_mode
        return self._payload


@pytest.mark.unit
class TestStatsCollectorProtocol:
    """Structural conformance for the runtime_checkable protocol."""

    def test_synthetic_implementation_satisfies_protocol(self):
        class _Synth:
            recording = False

            def start(self):
                pass

            def stop(self):
                pass

            def collect(self):
                return PlacementStats()

        assert isinstance(_Synth(), StatsCollector)

    def test_class_missing_collect_fails_isinstance(self):
        class _BadNoCollect:
            recording = False

            def start(self):
                pass

            def stop(self):
                pass

        assert not isinstance(_BadNoCollect(), StatsCollector)

    def test_sglang_adapter_satisfies_protocol_without_sglang_installed(self):
        # Construction does not touch sglang; isinstance only checks method
        # shape, so this passes in CPU-only environments without sglang.
        adapter = SGLangStatsCollector(recorder=_RecorderShaped())
        assert isinstance(adapter, StatsCollector)


@pytest.mark.unit
class TestPlacementStatsAggregateField:
    def test_average_utilization_rate_defaults_none(self):
        s = PlacementStats()
        assert s.average_utilization_rate is None

    def test_average_utilization_rate_round_trips(self):
        s = PlacementStats(average_utilization_rate=0.42)
        assert s.average_utilization_rate == pytest.approx(0.42)


@pytest.mark.unit
class TestSGLangStatsCollectorExplicit:
    """Adapter against an explicit recorder-shaped object (sglang not needed)."""

    def test_collect_maps_dump_record_to_placement_stats(self):
        import torch

        payload = {
            "rank": 0,
            "logical_count": torch.zeros((4, 8), dtype=torch.int32),
            "average_utilization_rate_over_window": 0.42,
        }
        rec = _RecorderShaped(payload=payload, recording=True)
        adapter = SGLangStatsCollector(recorder=rec)

        stats = adapter.collect()

        assert stats.logical_count is not None
        assert stats.logical_count.shape == (4, 8)
        assert stats.average_utilization_rate == pytest.approx(0.42)
        assert rec.dump_count == 1
        assert rec.last_output_mode == "object"

    def test_collect_handles_none_payload(self):
        rec = _RecorderShaped(payload=None, recording=False)
        adapter = SGLangStatsCollector(recorder=rec)

        stats = adapter.collect()

        assert stats.logical_count is None
        assert stats.average_utilization_rate is None

    def test_collect_handles_missing_utilization_field(self):
        # Older sglang builds, or alternate accumulator modes, may omit
        # the optional utilization field.
        import torch

        rec = _RecorderShaped(
            payload={"logical_count": torch.zeros((2, 4), dtype=torch.int32)},
        )
        adapter = SGLangStatsCollector(recorder=rec)

        stats = adapter.collect()

        assert stats.logical_count is not None
        assert stats.average_utilization_rate is None

    def test_recording_property_delegates_to_recorder(self):
        rec = _RecorderShaped(recording=True)
        adapter = SGLangStatsCollector(recorder=rec)
        assert adapter.recording is True

        rec.recording = False
        assert adapter.recording is False

    def test_start_no_op_when_already_recording(self):
        rec = _RecorderShaped(recording=True)
        adapter = SGLangStatsCollector(recorder=rec)
        adapter.start()
        assert rec.start_count == 0  # not re-started

    def test_start_invokes_start_record_when_idle(self):
        rec = _RecorderShaped(recording=False)
        adapter = SGLangStatsCollector(recorder=rec)
        adapter.start()
        assert rec.start_count == 1
        assert rec.recording is True

    def test_stop_no_op_when_already_stopped(self):
        rec = _RecorderShaped(recording=False)
        adapter = SGLangStatsCollector(recorder=rec)
        adapter.stop()
        assert rec.stop_count == 0

    def test_stop_invokes_stop_record_when_recording(self):
        rec = _RecorderShaped(recording=True)
        adapter = SGLangStatsCollector(recorder=rec)
        adapter.stop()
        assert rec.stop_count == 1
        assert rec.recording is False


@pytest.mark.unit
class TestSGLangEPLBRuntimePlanPlacement:
    """``plan_placement`` should pull stats via the protocol and delegate
    to the existing ``init_by_eplb`` path. We exercise the protocol-call
    half here (sglang itself is mocked out via the recorder)."""

    def test_returns_none_when_collector_has_no_logical_count(self):
        from moe_load_balancer.adapters.sglang.eplb import SGLangEPLBRuntime

        rec = _RecorderShaped(payload=None)
        collector = SGLangStatsCollector(recorder=rec)
        runtime = SGLangEPLBRuntime()

        # server_args / model_config aren't touched on the early-return path.
        result = runtime.plan_placement(
            server_args=object(), model_config=object(), collector=collector,
        )
        assert result is None
        assert rec.dump_count == 1  # collector.collect() was called

    def test_calls_init_by_eplb_when_logical_count_present(self, monkeypatch):
        import torch
        from moe_load_balancer.adapters.sglang.eplb import SGLangEPLBRuntime

        lc = torch.zeros((4, 8), dtype=torch.int32)
        rec = _RecorderShaped(
            payload={"logical_count": lc, "average_utilization_rate_over_window": None},
        )
        collector = SGLangStatsCollector(recorder=rec)
        runtime = SGLangEPLBRuntime()

        captured = {}

        def fake_init_by_eplb(server_args, model_config, logical_count):
            captured["server_args"] = server_args
            captured["model_config"] = model_config
            captured["logical_count"] = logical_count
            return "SENTINEL_PLACEMENT"

        monkeypatch.setattr(runtime, "init_by_eplb", fake_init_by_eplb)

        sa, mc = object(), object()
        result = runtime.plan_placement(server_args=sa, model_config=mc, collector=collector)

        assert result == "SENTINEL_PLACEMENT"
        assert captured["server_args"] is sa
        assert captured["model_config"] is mc
        assert captured["logical_count"] is lc
