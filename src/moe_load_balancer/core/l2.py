from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from moe_load_balancer.core.types import (
    RoutingDecision,
    RoutingLoadHistory,
    RoutingPhase,
    RoutingPolicyConfig,
    RoutingRequest,
    TensorLike,
)


@runtime_checkable
class RoutingLoadHistoryProvider(Protocol):
    """Optional source for future policies that consume cross-batch history."""

    def load_history(self, layer_id: int) -> RoutingLoadHistory: ...


@runtime_checkable
class TokenRouter(Protocol):
    """One framework-neutral policy in the ordered L2 pipeline."""

    phase: RoutingPhase

    def route_tokens(
        self,
        request: RoutingRequest,
        policy: RoutingPolicyConfig,
    ) -> RoutingDecision: ...


@runtime_checkable
class LPLBKernels(Protocol):
    """Injected execution interface for the framework-neutral LPLB policy.

    The policy owns the LP and placement logic. Implementations of this
    protocol own only the tensor kernels used by the four execution stages.
    Mutable output buffers are supplied by the policy so it can isolate them
    per CUDA graph or eager stream.
    """

    def warmup(
        self,
        nc: int,
        nv: int,
        num_iters: int,
        device: TensorLike,
    ) -> None: ...

    def prep_lp_inputs(
        self,
        a_full: TensorLike,
        b: TensorLike,
        t1: TensorLike,
        global_counts: TensorLike,
        log_single: TensorLike,
        log_replicated: TensorLike,
        b1: TensorLike,
        a_base_row_sum: TensorLike,
    ) -> None: ...

    def solve_ipm(
        self,
        a: TensorLike,
        b: TensorLike,
        c: TensorLike,
        num_iters: int = 5,
        result: Optional[TensorLike] = None,
    ) -> TensorLike: ...

    def extract_log2phy_prob(
        self,
        log2phy_prob: TensorLike,
        x: TensorLike,
        t1: TensorLike,
        phy_single: TensorLike,
        phy_replicated: TensorLike,
        log2phy: TensorLike,
    ) -> None: ...

    def dispatch_probability(
        self,
        topk_ids: TensorLike,
        log2phy_prob: TensorLike,
        log2phy_map: TensorLike,
    ) -> TensorLike: ...
