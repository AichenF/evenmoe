#!/usr/bin/env python3
"""Seal and verify immutable artifacts for EPLB performance experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moe_load_balancer.reproducibility import (
    build_eplb_reproduction_manifest,
    compare_eplb_reproduction_manifests,
    verify_eplb_reproduction_manifest,
    write_eplb_reproduction_manifest,
)


def _identity(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or key in result:
            raise ValueError(f"identity must be a unique KEY=VALUE pair: {value!r}")
        result[key] = item
    return result


def _add_manifest_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--record-dataset", type=Path, required=True)
    parser.add_argument("--benchmark-dataset", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--num-physical-experts", type=int, required=True)
    parser.add_argument("--num-local-physical-experts", type=int, required=True)
    parser.add_argument("--num-groups", type=int, required=True)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--algorithm", default="auto")
    parser.add_argument("--identity", action="append", default=[], metavar="KEY=VALUE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal")
    _add_manifest_inputs(seal)
    verify = subparsers.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    compare = subparsers.add_parser("compare")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "seal":
        manifest = build_eplb_reproduction_manifest(
            manifest_path=args.manifest,
            record_dataset=args.record_dataset,
            benchmark_dataset=args.benchmark_dataset,
            schedule=args.schedule,
            recording=args.recording,
            num_physical_experts=args.num_physical_experts,
            num_local_physical_experts=args.num_local_physical_experts,
            num_groups=args.num_groups,
            num_nodes=args.num_nodes,
            algorithm=args.algorithm,
            identity=_identity(args.identity),
        )
        write_eplb_reproduction_manifest(manifest, args.manifest)
        print(json.dumps(manifest, sort_keys=True))
    elif args.command == "verify":
        manifest = verify_eplb_reproduction_manifest(args.manifest)
        print(
            json.dumps(
                {
                    "verified": True,
                    "manifest": str(args.manifest),
                    "placement_fingerprint_sha256": manifest["placement_plan"][
                        "fingerprint_sha256"
                    ],
                },
                sort_keys=True,
            )
        )
    else:
        result = compare_eplb_reproduction_manifests(args.left, args.right)
        print(json.dumps(result, sort_keys=True))
        if not result["match"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
