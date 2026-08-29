"""Thin Python wrapper around MLB's UltraEP-derived CUDA placement kernels."""

from __future__ import annotations

from typing import Any

import torch

from .config import UltraEPPlacementTuning


class UltraEPPlacementBackend:
    """Stateless placement/reroute kernel launcher.

    The extension is imported lazily so the rest of MLB stays usable on hosts
    without CUDA.  Runtime state and expert communication intentionally live
    outside this class.
    """

    def __init__(self, extension: Any | None = None) -> None:
        self._extension = extension

    @property
    def extension(self):
        if self._extension is None:
            try:
                from . import _C
            except ImportError as exc:  # pragma: no cover - optional CUDA build
                raise ImportError(
                    "UltraEP placement kernels are not built. Reinstall MLB with "
                    "MLB_BUILD_ULTRAEP_PLACEMENT=1 and --no-build-isolation."
                ) from exc
            self._extension = _C
        return self._extension

    def solve(
        self,
        *,
        expert_loads: torch.Tensor,
        expert_loads_per_rank: torch.Tensor,
        physical_to_logical_map: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_counts: torch.Tensor,
        logical_instance_quota: torch.Tensor,
        logical_instance_quota_prefix: torch.Tensor,
        rank_quota_prefix: torch.Tensor,
        num_ranks: int,
        num_local_master_experts: int,
        num_local_redundant_experts: int,
        num_nvl_ranks: int,
        rank: int,
        tuning: UltraEPPlacementTuning,
    ) -> None:
        self.extension.solve_placement(
            expert_loads,
            expert_loads_per_rank,
            physical_to_logical_map,
            logical_to_physical_map,
            logical_replica_counts,
            logical_instance_quota,
            logical_instance_quota_prefix,
            rank_quota_prefix,
            num_ranks,
            num_local_master_experts,
            num_local_redundant_experts,
            num_nvl_ranks,
            tuning.balance_threshold,
            tuning.min_tokens_per_replica,
            tuning.allow_zero_master_quota,
            tuning.locality_aware,
            tuning.oracle_eps,
            tuning.kernel_stage,
            rank,
        )

    def reroute_sparse(
        self,
        *,
        logical_topk_ids: torch.Tensor,
        routed_topk_ids: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_counts: torch.Tensor,
        rank_quota_prefix: torch.Tensor,
        counters: torch.Tensor,
    ) -> None:
        self.extension.reroute_sparse(
            logical_topk_ids,
            routed_topk_ids,
            logical_to_physical_map,
            logical_replica_counts,
            rank_quota_prefix,
            counters,
        )
