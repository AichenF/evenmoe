"""Immutable EPLB recording manifests for reproducible placement experiments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from moe_load_balancer.core.orchestrator import MoELoadBalancer
from moe_load_balancer.core.types import (
    PlacementPolicyConfig,
    PlacementRequest,
    PlacementStats,
    PlacementTopology,
)

EPLB_REPRO_SCHEMA_VERSION = 1
EPLB_REPRO_ARTIFACT_KIND = "mlb.eplb.reproduction"


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_descriptor(tensor: Tensor) -> dict[str, Any]:
    """Describe and hash a tensor using canonical CPU-contiguous bytes."""

    value = tensor.detach().cpu().contiguous()
    header = {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "numel": value.numel(),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {**header, "sha256": digest.hexdigest()}


def _load_logical_count(path: Path) -> Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    logical_count = value.get("logical_count") if isinstance(value, dict) else value
    if not isinstance(logical_count, Tensor):
        raise ValueError(f"{path} does not contain a logical_count tensor")
    if logical_count.ndim not in (2, 3):
        raise ValueError(
            f"logical_count must have shape [layers, experts] or "
            f"[observations, layers, experts], got {tuple(logical_count.shape)}"
        )
    if logical_count.numel() == 0:
        raise ValueError("logical_count must not be empty")
    if logical_count.is_floating_point() and not torch.isfinite(logical_count).all():
        raise ValueError("logical_count contains NaN or Inf")
    if (logical_count < 0).any():
        raise ValueError("logical_count contains negative values")
    return logical_count


def _portable_path(path: Path, manifest_path: Path) -> str:
    absolute = path.resolve()
    try:
        return os.path.relpath(absolute, manifest_path.parent.resolve())
    except ValueError:
        return str(absolute)


def _resolve_path(stored_path: str, manifest_path: Path) -> Path:
    path = Path(stored_path)
    return path if path.is_absolute() else manifest_path.parent / path


def _file_entry(path: Path, manifest_path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _portable_path(path, manifest_path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def build_eplb_reproduction_manifest(
    *,
    manifest_path: Path,
    record_dataset: Path,
    benchmark_dataset: Path,
    schedule: Path,
    recording: Path,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    algorithm: str = "auto",
    identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a manifest that binds workload bytes to one deterministic plan."""

    files = {
        "record_dataset": _file_entry(record_dataset, manifest_path),
        "benchmark_dataset": _file_entry(benchmark_dataset, manifest_path),
        "schedule": _file_entry(schedule, manifest_path),
        "recording": _file_entry(recording, manifest_path),
    }
    logical_count = _load_logical_count(recording)
    request = PlacementRequest(
        stats=PlacementStats(logical_count=logical_count),
        topology=PlacementTopology(
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
        ),
        policy=PlacementPolicyConfig(name=algorithm),
    )
    plan = MoELoadBalancer().plan_placement(request)
    plan_tensors = {
        "physical_to_logical_map": tensor_descriptor(plan.physical_to_logical_map),
        "logical_to_all_physical_map": tensor_descriptor(plan.logical_to_all_physical_map),
        "logical_to_physical_count": tensor_descriptor(plan.logical_to_physical_count),
    }
    fingerprint_input = {
        "logical_count": tensor_descriptor(logical_count),
        "topology": {
            "num_physical_experts": num_physical_experts,
            "num_local_physical_experts": num_local_physical_experts,
            "num_groups": num_groups,
            "num_nodes": num_nodes,
        },
        "policy": {
            "requested": algorithm,
            "resolved": plan.metadata["resolved_policy"],
        },
        "tensors": plan_tensors,
    }
    placement_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": EPLB_REPRO_SCHEMA_VERSION,
        "artifact_kind": EPLB_REPRO_ARTIFACT_KIND,
        "identity": dict(sorted((identity or {}).items())),
        "files": files,
        "recording": {"logical_count": tensor_descriptor(logical_count)},
        "topology": fingerprint_input["topology"],
        "policy": fingerprint_input["policy"],
        "placement_plan": {
            "fingerprint_sha256": placement_fingerprint,
            **plan_tensors,
        },
    }


def write_eplb_reproduction_manifest(manifest: Mapping[str, Any], manifest_path: Path) -> None:
    """Create an immutable manifest, allowing only an identical reseal."""

    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        if manifest_path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(
                f"refusing to overwrite non-identical EPLB manifest: {manifest_path}"
            )
        return
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_eplb_reproduction_manifest(manifest_path: Path) -> dict[str, Any]:
    """Recompute every hash and placement map described by a manifest."""

    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected.get("schema_version") != EPLB_REPRO_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version in {manifest_path}")
    if expected.get("artifact_kind") != EPLB_REPRO_ARTIFACT_KIND:
        raise ValueError(f"unexpected artifact_kind in {manifest_path}")
    files = expected["files"]
    topology = expected["topology"]
    actual = build_eplb_reproduction_manifest(
        manifest_path=manifest_path,
        record_dataset=_resolve_path(files["record_dataset"]["path"], manifest_path),
        benchmark_dataset=_resolve_path(files["benchmark_dataset"]["path"], manifest_path),
        schedule=_resolve_path(files["schedule"]["path"], manifest_path),
        recording=_resolve_path(files["recording"]["path"], manifest_path),
        num_physical_experts=topology["num_physical_experts"],
        num_local_physical_experts=topology["num_local_physical_experts"],
        num_groups=topology["num_groups"],
        num_nodes=topology["num_nodes"],
        algorithm=expected["policy"]["requested"],
        identity=expected.get("identity", {}),
    )
    if actual != expected:
        raise ValueError(f"EPLB reproduction manifest verification failed: {manifest_path}")
    return actual


def compare_eplb_reproduction_manifests(left_path: Path, right_path: Path) -> dict[str, Any]:
    """Compare the workload, raw recording and derived placement of two manifests."""

    left = verify_eplb_reproduction_manifest(left_path)
    right = verify_eplb_reproduction_manifest(right_path)
    comparisons = {
        name: left["files"][name]["sha256"] == right["files"][name]["sha256"]
        for name in ("record_dataset", "benchmark_dataset", "schedule", "recording")
    }
    comparisons["logical_count"] = (
        left["recording"]["logical_count"]["sha256"]
        == right["recording"]["logical_count"]["sha256"]
    )
    comparisons["topology"] = left["topology"] == right["topology"]
    comparisons["policy"] = left["policy"] == right["policy"]
    comparisons["placement_plan"] = (
        left["placement_plan"]["fingerprint_sha256"]
        == right["placement_plan"]["fingerprint_sha256"]
    )
    return {"match": all(comparisons.values()), "comparisons": comparisons}
