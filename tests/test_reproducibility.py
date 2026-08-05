from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from moe_load_balancer.reproducibility import (
    build_eplb_reproduction_manifest,
    compare_eplb_reproduction_manifests,
    tensor_descriptor,
    verify_eplb_reproduction_manifest,
    write_eplb_reproduction_manifest,
)


def _write_inputs(
    root: Path,
    logical_count: torch.Tensor,
    *,
    recording_metadata: dict | None = None,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "record_dataset": root / "record.jsonl",
        "benchmark_dataset": root / "benchmark.jsonl",
        "schedule": root / "schedule.json",
        "recording": root / "logical_count.pt",
        "manifest": root / "eplb_manifest.json",
    }
    paths["record_dataset"].write_text('{"prompt":"record"}\n', encoding="utf-8")
    paths["benchmark_dataset"].write_text('{"prompt":"benchmark"}\n', encoding="utf-8")
    paths["schedule"].write_text('{"indices":[0]}\n', encoding="utf-8")
    torch.save(
        {"logical_count": logical_count, **(recording_metadata or {})},
        paths["recording"],
    )
    return paths


def _manifest(paths: dict[str, Path]) -> dict:
    return build_eplb_reproduction_manifest(
        manifest_path=paths["manifest"],
        record_dataset=paths["record_dataset"],
        benchmark_dataset=paths["benchmark_dataset"],
        schedule=paths["schedule"],
        recording=paths["recording"],
        num_physical_experts=8,
        num_local_physical_experts=4,
        num_groups=1,
        num_nodes=1,
        algorithm="deepseek",
        identity={"model": "unit-test"},
    )


@pytest.mark.unit
def test_tensor_descriptor_depends_on_shape_dtype_and_values():
    original = tensor_descriptor(torch.tensor([[1, 2], [3, 4]], dtype=torch.int64))
    same = tensor_descriptor(torch.tensor([[1, 2], [3, 4]], dtype=torch.int64))
    reshaped = tensor_descriptor(torch.tensor([1, 2, 3, 4], dtype=torch.int64))
    different_dtype = tensor_descriptor(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))

    assert original == same
    assert original["sha256"] != reshaped["sha256"]
    assert original["sha256"] != different_dtype["sha256"]


@pytest.mark.unit
def test_manifest_seals_and_verifies_exact_recording_and_placement(tmp_path: Path):
    paths = _write_inputs(tmp_path, torch.tensor([[[20, 3, 9, 4]]]))
    manifest = _manifest(paths)
    write_eplb_reproduction_manifest(manifest, paths["manifest"])

    assert verify_eplb_reproduction_manifest(paths["manifest"]) == manifest
    write_eplb_reproduction_manifest(manifest, paths["manifest"])

    stored = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert stored["recording"]["logical_count"]["shape"] == [1, 1, 4]
    assert stored["policy"] == {"requested": "deepseek", "resolved": "deepseek"}
    assert len(stored["placement_plan"]["fingerprint_sha256"]) == 64


@pytest.mark.unit
def test_manifest_rejects_mutated_inputs_and_non_identical_overwrite(tmp_path: Path):
    paths = _write_inputs(tmp_path, torch.tensor([[[20, 3, 9, 4]]]))
    manifest = _manifest(paths)
    write_eplb_reproduction_manifest(manifest, paths["manifest"])
    paths["schedule"].write_text('{"indices":[]}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="verification failed"):
        verify_eplb_reproduction_manifest(paths["manifest"])
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_eplb_reproduction_manifest({**manifest, "identity": {}}, paths["manifest"])


@pytest.mark.unit
def test_manifest_compare_ignores_paths_but_requires_same_raw_recording(tmp_path: Path):
    left_paths = _write_inputs(tmp_path / "left", torch.tensor([[[20, 3, 9, 4]]]))
    right_paths = _write_inputs(
        tmp_path / "right",
        torch.tensor([[[20, 3, 9, 4]]]),
        recording_metadata={"recorder_run": "different"},
    )
    left = _manifest(left_paths)
    right = _manifest(right_paths)
    write_eplb_reproduction_manifest(left, left_paths["manifest"])
    write_eplb_reproduction_manifest(right, right_paths["manifest"])

    result = compare_eplb_reproduction_manifests(left_paths["manifest"], right_paths["manifest"])
    assert result["comparisons"]["logical_count"] is True
    assert result["comparisons"]["placement_plan"] is True
    # Raw byte identity remains a stricter cohort requirement than
    # logical-count equality because recorder metadata may also differ.
    assert result["comparisons"]["recording"] is False
    assert result["match"] is False
