"""Joint min-max LPLB routing for redundant expert replicas.

The framework supplies an all-reduced logical expert count.  MLB owns the LP
matrix, per-layer cached state, the solver invocation, and probability-based
physical dispatch.  A small injected backend performs the fused IPM solve so
the policy does not import a serving framework or its process groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from moe_load_balancer.core.l2 import TokenRouter
from moe_load_balancer.core.types import (
    PlacementSnapshot,
    RoutingDecision,
    RoutingPhase,
    RoutingPolicyConfig,
    RoutingRequest,
)


class LPLBIPMBackend(Protocol):
    """Framework-injected execution backend for the small standard-form LP."""

    def warmup(self, nc: int, nv: int, *, device: torch.device) -> None: ...

    def solve(
        self,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        *,
        result: Tensor,
    ) -> Tensor: ...


class TorchReferenceIPMBackend:
    """Portable numerical reference; production integrations inject fused CUDA."""

    def __init__(self, *, num_iterations: int = 5) -> None:
        self.num_iterations = num_iterations

    def warmup(self, nc: int, nv: int, *, device: torch.device) -> None:
        del nc, nv, device

    def solve(
        self,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        *,
        result: Tensor,
    ) -> Tensor:
        nc, nv = a.shape
        x = torch.ones(nv, dtype=torch.float32, device=a.device)
        d_max = torch.zeros((), dtype=torch.float32, device=a.device)
        eye = torch.eye(nc, dtype=torch.float32, device=a.device)
        for _ in range(self.num_iterations):
            ax2 = a * (x * x).unsqueeze(0)
            kkt = ax2 @ a.t() + 1e-12 * eye
            delta = torch.linalg.solve(kkt, ax2 @ c)
            direction = x * (c - a.t() @ delta)
            d_max = direction.max()
            alpha = torch.where(d_max > 1e-9, 0.999 / d_max, 1.0)
            x = x * (1.0 - alpha * direction)

        residual = (a @ x - b).abs().max()
        converged = (
            (d_max < 0.1)
            & (x[-1] >= 0)
            & (x[-1] < 1e-4)
            & (residual < 0.05)
        )
        result.copy_(torch.where(converged, x, torch.full_like(x, 0.5)))
        return result


@dataclass
class _LayerState:
    placement_key: tuple
    candidates: Tensor
    copy_count: Tensor
    a_full: Tensor
    b: Tensor
    c: Tensor
    a_base_row_sum: Tensor
    log_single: Tensor
    log_replicated: Tensor
    phy_single: Tensor
    phy_replicated: Tensor
    b1: Tensor
    t1: Tensor
    x: Tensor
    probability: Tensor
    probability_source: Tensor


def _placement_key(snapshot: PlacementSnapshot) -> tuple:
    candidates = snapshot.logical_to_physical_candidates
    copy_count = snapshot.logical_to_physical_count
    physical_to_logical = snapshot.physical_to_logical_map
    return (
        candidates.data_ptr(),
        tuple(candidates.shape),
        getattr(candidates, "_version", None),
        copy_count.data_ptr(),
        tuple(copy_count.shape),
        getattr(copy_count, "_version", None),
        physical_to_logical.data_ptr(),
        tuple(physical_to_logical.shape),
        getattr(physical_to_logical, "_version", None),
    )


class LPLBL2Router(TokenRouter):
    """Solve one joint min-max LP per layer and dispatch to valid replicas."""

    phase = RoutingPhase.PHYSICAL_ROUTING

    def __init__(
        self,
        *,
        ep_size: int,
        num_physical_experts: int,
        ipm_backend: LPLBIPMBackend,
    ) -> None:
        if ep_size <= 0 or num_physical_experts <= 0:
            raise ValueError("ep_size and num_physical_experts must be positive.")
        if num_physical_experts % ep_size != 0:
            raise ValueError("num_physical_experts must be divisible by ep_size.")
        self.ep_size = ep_size
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_physical_experts // ep_size
        self.ipm_backend = ipm_backend
        self._layer_states: dict[int, _LayerState] = {}

    def prepare_layer(self, snapshot: PlacementSnapshot) -> None:
        """Build and warm the cached LP state outside the first real request."""

        self._get_state(snapshot)

    def route_tokens(
        self,
        request: RoutingRequest,
        policy: RoutingPolicyConfig,
    ) -> RoutingDecision:
        del policy
        if request.placement is None:
            raise ValueError("LPLBL2Router requires RoutingRequest.placement.")
        if request.global_logical_count is None:
            raise ValueError("LPLBL2Router requires global_logical_count.")
        topk_ids = request.logical_topk_ids
        if not isinstance(topk_ids, Tensor):
            raise TypeError("LPLBL2Router requires torch.Tensor logical_topk_ids.")
        if topk_ids.numel() == 0:
            return RoutingDecision(
                routed_physical_topk_ids=topk_ids,
                topk_weights=request.topk_weights,
                metadata={"policy": "lplb", "empty": True},
            )

        state = self._get_state(request.placement)
        global_counts = request.global_logical_count.to(
            device=topk_ids.device, dtype=torch.float32
        )
        self._prepare_inputs(state, global_counts)
        self.ipm_backend.solve(
            state.a_full,
            state.b,
            state.c,
            result=state.x,
        )
        self._extract_probability(state)
        physical = self._dispatch(topk_ids, state.probability, state.candidates)
        return RoutingDecision(
            routed_physical_topk_ids=physical,
            topk_weights=request.topk_weights,
            metadata={"policy": "lplb"},
        )

    def _get_state(self, snapshot: PlacementSnapshot) -> _LayerState:
        if snapshot.num_physical_experts != self.num_physical_experts:
            raise ValueError("Placement num_physical_experts does not match LPLB router.")
        if not isinstance(snapshot.logical_to_physical_candidates, Tensor):
            raise TypeError("LPLB placement candidates must be torch.Tensor.")
        key = _placement_key(snapshot)
        cached = self._layer_states.get(snapshot.layer_id)
        if cached is not None and cached.placement_key == key:
            return cached

        copy_count = snapshot.logical_to_physical_count.to(torch.int64).contiguous()
        raw_candidates = snapshot.logical_to_physical_candidates
        if copy_count.numel() == 0:
            raise ValueError("LPLB placement copy counts must not be empty.")
        # Frameworks may store this table with a fixed, heavily padded final
        # dimension (SGLang uses num_physical_experts).  Compact it only when
        # placement state changes; the cache hit above keeps this GPU-to-host
        # scalar read out of the per-forward hot path.
        max_copies = int(copy_count.max().item())
        if max_copies <= 0 or max_copies > raw_candidates.shape[1]:
            raise ValueError(
                "LPLB placement copy counts are inconsistent with candidate slots."
            )
        candidates = raw_candidates[:, :max_copies].to(torch.int64).contiguous()
        physical_to_logical = snapshot.physical_to_logical_map.to(torch.int64).contiguous()
        device = candidates.device
        num_logical = candidates.shape[0]

        log_single = torch.nonzero(copy_count == 1).flatten().to(torch.int64)
        log_replicated = torch.nonzero(copy_count > 1).flatten().to(torch.int64)
        phy_single = candidates[log_single, 0]
        phy_replicated = torch.nonzero(copy_count[physical_to_logical] > 1).flatten()
        num_single = log_single.numel()
        num_replicated_logical = log_replicated.numel()
        num_replicated_physical = phy_replicated.numel()

        rank_by_physical = torch.arange(
            self.num_physical_experts, device=device, dtype=torch.int64
        ) // self.num_local_physical_experts
        b_full = torch.nn.functional.one_hot(
            rank_by_physical, num_classes=self.ep_size
        ).t().to(torch.float32)
        b1 = b_full[:, phy_single].contiguous()
        b2 = b_full[:, phy_replicated]

        c_matrix = torch.zeros(
            (num_replicated_logical, num_replicated_physical),
            dtype=torch.float32,
            device=device,
        )
        replicated_phy_to_log = physical_to_logical[phy_replicated]
        for row, logical_id in enumerate(log_replicated):
            c_matrix[row, replicated_phy_to_log == logical_id] = 1.0

        zeros_top_g = torch.zeros(
            (num_replicated_logical, self.ep_size),
            dtype=torch.float32,
            device=device,
        )
        zeros_top_1 = torch.zeros(
            (num_replicated_logical, 1), dtype=torch.float32, device=device
        )
        a_top = torch.hstack((c_matrix, zeros_top_g, zeros_top_1))
        a_bottom = torch.hstack(
            (
                b2,
                torch.eye(self.ep_size, dtype=torch.float32, device=device),
                -torch.ones((self.ep_size, 1), dtype=torch.float32, device=device),
            )
        )
        a_base = torch.vstack((a_top, a_bottom)).contiguous()
        nc = a_base.shape[0]
        nv = a_base.shape[1] + 1
        a_full = torch.empty((nc, nv), dtype=torch.float32, device=device)
        a_full[:, :-1].copy_(a_base)
        objective = torch.zeros(nv, dtype=torch.float32, device=device)
        objective[-2] = 1.0
        objective[-1] = 1000.0

        state = _LayerState(
            placement_key=key,
            candidates=candidates,
            copy_count=copy_count,
            a_full=a_full,
            b=torch.empty(nc, dtype=torch.float32, device=device),
            c=objective,
            a_base_row_sum=a_base.sum(dim=1),
            log_single=log_single,
            log_replicated=log_replicated,
            phy_single=phy_single,
            phy_replicated=phy_replicated,
            b1=b1,
            t1=torch.empty(num_single, dtype=torch.float32, device=device),
            x=torch.empty(nv, dtype=torch.float32, device=device),
            probability=torch.empty(
                (num_logical, max_copies), dtype=torch.float32, device=device
            ),
            probability_source=self._build_probability_source(
                candidates,
                phy_single,
                phy_replicated,
                self.num_physical_experts,
            ),
        )
        self.ipm_backend.warmup(nc, nv, device=device)
        self._layer_states[snapshot.layer_id] = state
        return state

    @staticmethod
    def _prepare_inputs(state: _LayerState, global_counts: Tensor) -> None:
        if global_counts.is_cuda:
            from moe_load_balancer.policies.l2._lplb_cuda import prep_lp_inputs_cuda

            prep_lp_inputs_cuda(
                state.a_full,
                state.b,
                state.t1,
                global_counts,
                state.log_single,
                state.log_replicated,
                state.b1,
                state.a_base_row_sum,
            )
            return

        counts = global_counts / global_counts.sum().clamp(min=1.0)
        state.t1.copy_(counts[state.log_single])
        num_replicated = state.log_replicated.numel()
        state.b[:num_replicated].copy_(counts[state.log_replicated])
        state.b[num_replicated:].copy_(-(state.b1 @ state.t1))
        state.a_full[:, -1].copy_(state.b - state.a_base_row_sum)

    @staticmethod
    def _extract_probability(state: _LayerState) -> None:
        if state.x.is_cuda:
            from moe_load_balancer.policies.l2._lplb_cuda import (
                extract_probability_cuda,
            )

            extract_probability_cuda(
                state.probability,
                state.x,
                state.t1,
                state.probability_source,
            )
            return

        num_replicated_physical = state.phy_replicated.numel()
        num_physical = state.phy_single.numel() + num_replicated_physical
        physical_probability = torch.zeros(
            num_physical + 1,
            dtype=torch.float32,
            device=state.x.device,
        )
        physical_probability[state.phy_replicated] = state.x[
            :num_replicated_physical
        ].clamp(min=0)
        physical_probability[state.phy_single] = state.t1
        # torch.take intentionally maps candidate -1 to the final zero sink.
        state.probability.copy_(torch.take(physical_probability, state.candidates))

    @staticmethod
    def _build_probability_source(
        candidates: Tensor,
        phy_single: Tensor,
        phy_replicated: Tensor,
        num_physical_experts: int,
    ) -> Tensor:
        """Map each placement slot directly to its preallocated LP source.

        Non-negative values index the replicated-expert prefix of ``x``;
        values <= -2 encode ``t1`` indices; -1 is the padded zero sink.
        This placement-only table lets the CUDA post step avoid allocating a
        physical-probability scratch tensor on every layer forward.
        """

        source_by_physical = torch.full(
            (num_physical_experts + 1,),
            -1,
            dtype=torch.int64,
            device=candidates.device,
        )
        source_by_physical[phy_replicated] = torch.arange(
            phy_replicated.numel(), dtype=torch.int64, device=candidates.device
        )
        source_by_physical[phy_single] = -(
            torch.arange(
                phy_single.numel(), dtype=torch.int64, device=candidates.device
            )
            + 2
        )
        # Candidate -1 intentionally indexes the final zero-sink element.
        return torch.take(source_by_physical, candidates).contiguous()

    def _dispatch(self, topk_ids: Tensor, probability: Tensor, candidates: Tensor) -> Tensor:
        if not topk_ids.is_cuda:
            flat = topk_ids.reshape(-1).to(torch.int64)
            probs = probability[flat]
            maps = candidates[flat]
            row_sum = probs.sum(dim=-1, keepdim=True)
            probs = torch.where(row_sum > 0, probs, (maps >= 0).to(probs.dtype))
            chosen = torch.multinomial(probs, 1).squeeze(-1)
            return maps.gather(1, chosen[:, None]).view(topk_ids.shape).to(topk_ids.dtype)

        flat = topk_ids.reshape(-1).contiguous().to(torch.int32)
        n = flat.numel()
        # One LPLB router is shared by every layer.  Per-call outputs let the
        # PyTorch allocator track their CUDA-stream lifetime; a process-wide
        # scratch tensor can be overwritten by overlap scheduling before its
        # asynchronous consumer finishes.
        output = torch.empty(n, dtype=torch.int32, device=topk_ids.device)
        random_values = torch.empty(n, dtype=torch.float32, device=topk_ids.device)
        random_values.uniform_()
        from moe_load_balancer.policies.l2._lplb_cuda import dispatch_probability_cuda

        dispatch_probability_cuda(
            output,
            flat,
            probability,
            candidates,
            random_values,
        )
        return output.view(topk_ids.shape).to(topk_ids.dtype)
