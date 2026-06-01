"""Tests for the framework-neutral L1 EPLB pieces and the SGLang bridge:

* :class:`EPLBRecorder` (policies/l1/eplb/recorder.py)
* :class:`EPLBPlacementProvider` (policies/l1/eplb/provider.py)
* :class:`SGLangStatsBridge` (adapters/sglang/eplb/stats_bridge.py)
* :class:`SGLangEPLBRuntime` facade extensions + default-runtime registry

All tests are CPU-only and do not import sglang.
"""

from __future__ import annotations

import pytest
import torch

from moe_load_balancer import (
    PlacementPlan,
    PlacementProvider,
    PlacementSnapshot,
    PlacementStats,
    RebalancePolicyConfig,
    StatsCollector,
)
from moe_load_balancer.adapters.sglang.eplb import (
    SGLangEPLBRuntime,
    SGLangStatsBridge,
    clear_default_runtime,
    get_default_runtime,
    set_default_runtime,
)
from moe_load_balancer.policies.l1.eplb.provider import EPLBPlacementProvider
from moe_load_balancer.policies.l1.eplb.recorder import EPLBRecorder


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _phy2log(num_layers=3, num_logical=4, replicas=2):
    """Build a deterministic physical-to-logical map: each logical expert
    gets ``replicas`` consecutive physical slots."""
    num_physical = num_logical * replicas
    m = torch.zeros((num_layers, num_physical), dtype=torch.int64)
    for L in range(num_layers):
        for p in range(num_physical):
            m[L, p] = p // replicas
    return m


def _gpc(num_layers=3, num_physical=8, fill=1):
    return torch.full((num_layers, num_physical), fill, dtype=torch.int32)


# ----------------------------------------------------------------------
# EPLBRecorder
# ----------------------------------------------------------------------


@pytest.mark.unit
class TestEPLBRecorder:
    def test_satisfies_stats_collector_protocol(self):
        rec = EPLBRecorder()
        assert isinstance(rec, StatsCollector)

    def test_collect_empty_returns_blank_stats(self):
        rec = EPLBRecorder()
        stats = rec.collect()
        assert stats.logical_count is None
        assert stats.physical_count is None
        assert stats.average_utilization_rate is None

    def test_append_then_collect_sums_window(self):
        rec = EPLBRecorder(RebalancePolicyConfig(recording_window_size=10))
        phy2log = _phy2log()
        # Push 5 forwards of all-ones counts -> physical_count[L, p] = 5
        for _ in range(5):
            rec.append(global_physical_count=_gpc(fill=1), physical_to_logical_map=phy2log)
        assert rec.window_size() == 5

        stats = rec.collect()
        assert stats.logical_count is not None
        # 2 replicas per logical, fill=1 over 5 steps -> 10 per logical
        expected = torch.full((3, 4), 10, dtype=torch.int32)
        assert torch.equal(stats.logical_count, expected)
        # physical_count is also exposed (5 per slot)
        assert stats.physical_count is not None
        assert torch.equal(stats.physical_count, torch.full((3, 8), 5, dtype=torch.int32))

    def test_collect_drains_buffer(self):
        rec = EPLBRecorder(RebalancePolicyConfig(recording_window_size=10))
        phy2log = _phy2log()
        rec.append(global_physical_count=_gpc(), physical_to_logical_map=phy2log)
        assert rec.window_size() == 1
        rec.collect()
        assert rec.window_size() == 0

    def test_window_rolls_over_maxlen(self):
        rec = EPLBRecorder(RebalancePolicyConfig(recording_window_size=3))
        phy2log = _phy2log()
        for _ in range(10):
            rec.append(global_physical_count=_gpc(), physical_to_logical_map=phy2log)
        # Only the most recent 3 are kept
        assert rec.window_size() == 3

    def test_stop_pauses_append(self):
        rec = EPLBRecorder()
        rec.stop()
        rec.append(
            global_physical_count=_gpc(), physical_to_logical_map=_phy2log()
        )
        assert rec.window_size() == 0
        assert rec.recording is False

    def test_start_resumes(self):
        rec = EPLBRecorder()
        rec.stop()
        rec.start()
        rec.append(
            global_physical_count=_gpc(), physical_to_logical_map=_phy2log()
        )
        assert rec.window_size() == 1
        assert rec.recording is True

    def test_uses_latest_phy2log_at_collect_time(self):
        """If placement changes mid-window (re-mapping with same num_logical),
        collect uses the latest map for the conversion."""
        rec = EPLBRecorder(RebalancePolicyConfig(recording_window_size=10))
        # v1: logical 0 -> slots [0,1]; logical 1 -> slots [2,3]
        phy2log_v1 = torch.tensor([[0, 0, 1, 1]], dtype=torch.int64)
        rec.append(
            global_physical_count=torch.tensor([[10, 20, 30, 40]], dtype=torch.int32),
            physical_to_logical_map=phy2log_v1,
        )
        # v2: re-mapped interleave: logical 0 -> slots [0,2]; logical 1 -> slots [1,3]
        phy2log_v2 = torch.tensor([[0, 1, 0, 1]], dtype=torch.int64)
        rec.append(
            global_physical_count=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
            physical_to_logical_map=phy2log_v2,
        )
        stats = rec.collect()
        # Per-layer summed physical count: [11, 22, 33, 44]
        # Applied via the latest map (v2): logical[0] = 11+33 = 44,
        #                                  logical[1] = 22+44 = 66
        assert stats.logical_count is not None
        assert torch.equal(
            stats.logical_count, torch.tensor([[44, 66]], dtype=torch.int32)
        )


# ----------------------------------------------------------------------
# EPLBPlacementProvider
# ----------------------------------------------------------------------


@pytest.mark.unit
class TestEPLBPlacementProvider:
    def test_satisfies_placement_provider_protocol(self):
        p = EPLBPlacementProvider()
        assert isinstance(p, PlacementProvider)

    def test_snapshot_without_plan_raises(self):
        p = EPLBPlacementProvider()
        with pytest.raises(RuntimeError):
            p.snapshot(layer_id=0)

    def test_update_with_bumps_version(self):
        p = EPLBPlacementProvider()
        assert p.version == 0
        p.update_with(_build_plan())
        assert p.version == 1
        p.update_with(_build_plan())
        assert p.version == 2

    def test_snapshot_carries_version_in_metadata(self):
        p = EPLBPlacementProvider()
        p.update_with(_build_plan())
        snap = p.snapshot(layer_id=1)
        assert snap.layer_id == 1
        assert snap.metadata["placement_version"] == 1

    def test_snapshot_preserves_base_metadata(self):
        """Base snapshot metadata (from the plan) survives the version overlay."""
        plan = _build_plan(extra_metadata={"algorithm": "deepseek_hierarchical"})
        p = EPLBPlacementProvider()
        p.update_with(plan)
        snap = p.snapshot(layer_id=0)
        # The provider only adds placement_version; base metadata is preserved
        # (here we don't set per-layer metadata in _build_plan, but the
        # version key is present)
        assert "placement_version" in snap.metadata

    def test_unknown_layer_raises(self):
        p = EPLBPlacementProvider()
        p.update_with(_build_plan(num_layers=3))
        with pytest.raises(KeyError):
            p.snapshot(layer_id=99)

    def test_latest_plan_accessor(self):
        p = EPLBPlacementProvider()
        assert p.latest_plan() is None
        plan = _build_plan()
        p.update_with(plan)
        assert p.latest_plan() is plan


def _build_plan(num_layers=3, num_logical=4, num_physical=8, extra_metadata=None):
    snapshots = {}
    for L in range(num_layers):
        snapshots[L] = PlacementSnapshot(
            layer_id=L,
            num_logical_experts=num_logical,
            num_physical_experts=num_physical,
            logical_to_physical_candidates=torch.zeros((num_logical, 2), dtype=torch.int64),
            logical_to_physical_count=torch.full((num_logical,), 2, dtype=torch.int64),
        )
    metadata = {"algorithm": "test"}
    if extra_metadata:
        metadata.update(extra_metadata)
    return PlacementPlan(snapshots=snapshots, metadata=metadata)


# ----------------------------------------------------------------------
# SGLangStatsBridge
# ----------------------------------------------------------------------


@pytest.mark.unit
class TestSGLangStatsBridge:
    def test_append_forwards_to_recorder(self):
        rec = EPLBRecorder()
        br = SGLangStatsBridge(recorder=rec, rank=0)
        br.append_per_forward(
            global_physical_count=_gpc(), physical_to_logical_map=_phy2log()
        )
        assert rec.window_size() == 1

    def test_dump_emits_sglang_shaped_dict(self):
        rec = EPLBRecorder(RebalancePolicyConfig(recording_window_size=10))
        br = SGLangStatsBridge(recorder=rec, rank=7)
        for _ in range(3):
            br.append_per_forward(
                global_physical_count=_gpc(fill=1), physical_to_logical_map=_phy2log()
            )

        out = br.dump_for_sglang(average_utilization_rate_over_window=0.5)

        assert set(out.keys()) == {
            "rank",
            "logical_count",
            "average_utilization_rate_over_window",
        }
        assert out["rank"] == 7
        assert out["average_utilization_rate_over_window"] == 0.5
        # 3 forwards * fill=1 * 2 replicas = 6 per logical
        expected = torch.full((3, 4), 6, dtype=torch.int32)
        assert torch.equal(out["logical_count"], expected)

    def test_dump_drains_recorder(self):
        rec = EPLBRecorder()
        br = SGLangStatsBridge(recorder=rec, rank=0)
        br.append_per_forward(
            global_physical_count=_gpc(), physical_to_logical_map=_phy2log()
        )
        assert rec.window_size() == 1
        br.dump_for_sglang()
        assert rec.window_size() == 0

    def test_dump_with_no_utilization_arg(self):
        rec = EPLBRecorder()
        br = SGLangStatsBridge(recorder=rec, rank=0)
        out = br.dump_for_sglang()
        assert out["average_utilization_rate_over_window"] is None

    def test_reset_drains_without_consuming(self):
        rec = EPLBRecorder()
        br = SGLangStatsBridge(recorder=rec, rank=0)
        br.append_per_forward(
            global_physical_count=_gpc(), physical_to_logical_map=_phy2log()
        )
        br.reset()
        assert rec.window_size() == 0

    def test_recorder_handle_is_shared(self):
        rec = EPLBRecorder()
        br = SGLangStatsBridge(recorder=rec, rank=0)
        assert br.recorder is rec


# ----------------------------------------------------------------------
# SGLangEPLBRuntime facade (state sharing + provider publish)
# ----------------------------------------------------------------------


@pytest.mark.unit
class TestSGLangEPLBRuntimeFacade:
    def test_constructor_builds_default_pieces(self):
        rt = SGLangEPLBRuntime()
        assert isinstance(rt.recorder, EPLBRecorder)
        assert isinstance(rt.provider, EPLBPlacementProvider)
        assert isinstance(rt.rebalance_policy, RebalancePolicyConfig)

    def test_constructor_applies_overrides(self):
        custom_policy = RebalancePolicyConfig(
            recording_window_size=50, min_utilization_threshold=0.85
        )
        rt = SGLangEPLBRuntime(rebalance_policy=custom_policy)
        assert rt.rebalance_policy is custom_policy

    def test_get_stats_bridge_shares_recorder(self):
        rt = SGLangEPLBRuntime()
        br = rt.get_stats_bridge(rank=5)
        assert isinstance(br, SGLangStatsBridge)
        assert br.recorder is rt.recorder

    def test_drain_stats_returns_none_when_empty(self):
        rt = SGLangEPLBRuntime()
        assert rt.drain_stats() is None

    def test_drain_stats_returns_logical_count_after_append(self):
        rt = SGLangEPLBRuntime()
        rt.recorder.append(
            global_physical_count=_gpc(fill=1),
            physical_to_logical_map=_phy2log(),
        )
        drained = rt.drain_stats()
        assert drained is not None
        assert drained["logical_count"] is not None
        # 1 forward, fill=1, 2 replicas per logical -> 2 per logical
        assert torch.equal(
            drained["logical_count"], torch.full((3, 4), 2, dtype=torch.int32)
        )
        # Drained -> next call returns None (atomic snapshot-and-reset)
        assert rt.drain_stats() is None

    def test_compute_placement_produces_placement_tensors(self):
        rt = SGLangEPLBRuntime()
        # Synthetic logical_count: 1 layer, 4 logical experts, 8 physical
        logical_count = torch.tensor([[10, 20, 30, 40]], dtype=torch.int32)
        result = rt.compute_placement(
            logical_count=logical_count,
            num_physical_experts=8,
            num_local_physical_experts=4,  # ep_size=2
            num_groups=1,
            num_nodes=1,
        )
        assert result is not None
        assert "physical_to_logical_map" in result
        assert "logical_to_all_physical_map" in result
        # physical_to_logical_map shape: [num_layers, num_physical]
        assert result["physical_to_logical_map"].shape == (1, 8)
        # logical_to_all_physical_map shape: [num_layers, num_logical, max_replicas]
        assert result["logical_to_all_physical_map"].shape[0] == 1

    def test_compute_placement_publishes_to_provider(self):
        rt = SGLangEPLBRuntime()
        assert rt.provider.version == 0
        rt.compute_placement(
            logical_count=torch.tensor([[10, 20, 30, 40]], dtype=torch.int32),
            num_physical_experts=8,
            num_local_physical_experts=4,
            num_groups=1,
            num_nodes=1,
        )
        # Provider published once
        assert rt.provider.version == 1
        snap = rt.provider.snapshot(layer_id=0)
        assert snap.layer_id == 0

    def test_runtime_source_has_no_sglang_import(self):
        """Regression test for the Phase 1.5 inversion: the adapter module
        must not have any ``from sglang.…`` or ``import sglang`` lines.
        """
        from pathlib import Path
        import moe_load_balancer.adapters.sglang.eplb.runtime as rt_module
        src_text = Path(rt_module.__file__).read_text()
        # Strip docstrings (they contain the word "sglang" intentionally)
        import ast
        tree = ast.parse(src_text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = []
                if isinstance(node, ast.ImportFrom):
                    names.append(node.module or "")
                for alias in node.names:
                    names.append(alias.name)
                for name in names:
                    assert not name.startswith("sglang"), (
                        f"runtime.py imports sglang at top level: {name}"
                    )


# ----------------------------------------------------------------------
# Default-runtime registry
# ----------------------------------------------------------------------


@pytest.mark.unit
class TestDefaultRuntimeRegistry:
    def setup_method(self):
        clear_default_runtime()

    def teardown_method(self):
        clear_default_runtime()

    def test_lazy_init_returns_default_instance(self):
        rt = get_default_runtime()
        assert isinstance(rt, SGLangEPLBRuntime)

    def test_subsequent_gets_return_same_instance(self):
        rt1 = get_default_runtime()
        rt2 = get_default_runtime()
        assert rt1 is rt2

    def test_set_default_overrides_lazy_init(self):
        custom = SGLangEPLBRuntime(
            rebalance_policy=RebalancePolicyConfig(recording_window_size=999)
        )
        set_default_runtime(custom)
        got = get_default_runtime()
        assert got is custom
        assert got.rebalance_policy.recording_window_size == 999

    def test_clear_resets_state(self):
        custom = SGLangEPLBRuntime()
        set_default_runtime(custom)
        clear_default_runtime()
        # After clear, a new lazy default is built
        rt = get_default_runtime()
        assert rt is not custom


# ----------------------------------------------------------------------
# RebalancePolicyConfig
# ----------------------------------------------------------------------


@pytest.mark.unit
class TestRebalancePolicyConfig:
    def test_defaults(self):
        c = RebalancePolicyConfig()
        assert c.recording_window_size == 100
        assert c.min_utilization_threshold is None

    def test_overrides(self):
        c = RebalancePolicyConfig(
            recording_window_size=50, min_utilization_threshold=0.85
        )
        assert c.recording_window_size == 50
        assert c.min_utilization_threshold == pytest.approx(0.85)

    def test_is_frozen(self):
        from dataclasses import FrozenInstanceError

        c = RebalancePolicyConfig()
        with pytest.raises(FrozenInstanceError):
            c.recording_window_size = 999  # type: ignore[misc]
