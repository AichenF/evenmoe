from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from moe_load_balancer.core.types import (
    RoutingDecision,
    RoutingLoadHistory,
    RoutingRequest,
    TensorLike,
)


@runtime_checkable
class RoutingLoadHistoryProvider(Protocol):
    """Framework-neutral provider for long-lived L2 load state.

    Adapters that expose continuously-updated load tensors (live counters,
    EWMA / historical aggregates, per-expert breakdowns) implement this
    protocol so policies can read the latest view each routing call instead
    of receiving the same stable load through every ``RoutingRequest``.

    Per-batch transient load (e.g. shared-expert Waterfill's per-batch routed
    counts) is not exposed via this protocol; it goes through canonical
    ``RoutingRequest`` fields instead.
    """

    def load_history(self, layer_id: int) -> RoutingLoadHistory:
        ...


@runtime_checkable
class TokenRouter(Protocol):
    """Framework-neutral L2 token router."""

    def route_tokens(self, request: RoutingRequest) -> RoutingDecision:
        ...


@runtime_checkable
class LPLBKernels(Protocol):
    """Framework-neutral kernel interface for the LPLB four-stage pipeline.

    ``LPLBL2Router`` takes an ``LPLBKernels`` implementation by injection.
    The SDK ships one implementation backed by the vendored CUDA pipeline
    at :mod:`moe_load_balancer.kernels.lplb.cuda_solver`; framework adapters
    instantiate it and pass it to the router. A different adapter is free
    to supply its own implementation without changing the policy or the
    core contract.

    **Pipeline**: ``warmup`` (once per ``(nc, nv)`` shape) → per-batch
    ``prep_lp_inputs → solve_ipm → extract_log2phy_prob → dispatch_probability``.

    **Buffer ownership**: ``prep_lp_inputs`` and ``extract_log2phy_prob``
    write into caller-supplied output buffers (``A_full``, ``b``, ``t1``,
    ``log2phy_prob``). ``solve_ipm`` writes into ``result`` when supplied,
    otherwise allocates. ``dispatch_probability`` always allocates a fresh
    output tensor of the same shape as ``topk_ids``. All input/output
    tensors live on CUDA in the supported implementations; the protocol
    itself does not constrain the device so CPU oracle implementations
    can satisfy the same interface for unit testing.

    Tensor dtypes (when constrained by the supported implementation):
        - ``A_full``, ``b``, ``c``, ``A_base_row_sum``, ``log2phy_prob``,
          ``x``, ``t1``: float32.
        - ``global_counts``: float32 (cast from int32 counts upstream).
        - ``log_single``, ``log_replicated``, ``phy_single``,
          ``phy_replicated``, ``log2phy``, ``log2phy_map``: int64
          (``log2phy_map`` is converted to int32 on the fly inside
          ``dispatch_probability``).
        - ``B1``: float32.
        - ``topk_ids``: int32 or int64 (implementation must accept both
          and preserve the input dtype in the output).
    """

    def warmup(self, nc: int, nv: int, num_iters: int, device: TensorLike) -> None:
        """Pre-compile / pre-allocate any per-shape resources for ``(nc, nv)``.

        ``device`` accepts whatever the underlying framework uses (a
        ``torch.device``, a string like ``"cuda"``, or framework-specific
        opaque handle). For implementations with no shape-specific warmup
        path (e.g. CPU torch reference), this is a no-op.
        """
        ...

    def prep_lp_inputs(
        self,
        A_full: TensorLike,
        b: TensorLike,
        t1: TensorLike,
        global_counts: TensorLike,
        log_single: TensorLike,
        log_replicated: TensorLike,
        B1: TensorLike,
        A_base_row_sum: TensorLike,
    ) -> None:
        """Build the per-batch LP inputs.

        Writes into the caller-provided ``A_full`` (Big-M / last column
        only), ``b`` (RHS vector), and ``t1`` (per-single-copy global
        count) buffers. ``A_full``'s first ``NV-1`` columns must already
        hold the static ``A_base`` from solver init — this kernel does
        not touch them.
        """
        ...

    def solve_ipm(
        self,
        A: TensorLike,
        b: TensorLike,
        c: TensorLike,
        num_iters: int = 5,
        result: Optional[TensorLike] = None,
    ) -> TensorLike:
        """Run the interior-point method LP solve on ``A x = b``, minimize ``c^T x``.

        Returns the solution vector ``x`` of shape ``(NV,)``. When ``result``
        is supplied, the solver writes into it and returns the same tensor;
        otherwise a fresh tensor is allocated. On non-convergence the
        reference implementation writes ``0.5`` to every entry, matching the
        upstream LPLBSolver fallback behavior.
        """
        ...

    def extract_log2phy_prob(
        self,
        log2phy_prob: TensorLike,
        x: TensorLike,
        t1: TensorLike,
        phy_single: TensorLike,
        phy_replicated: TensorLike,
        log2phy: TensorLike,
    ) -> None:
        """Convert the IPM solution into a per-logical-expert probability table.

        Writes into the caller-provided ``log2phy_prob`` buffer of shape
        ``(num_logical, max_copies)``. Single-copy logical experts get
        a deterministic mass of 1.0 on their unique physical slot;
        replicated logicals get the LP-derived multinomial.
        """
        ...

    def dispatch_probability(
        self,
        topk_ids: TensorLike,
        log2phy_prob: TensorLike,
        log2phy_map: TensorLike,
    ) -> TensorLike:
        """Sample a physical replica for each token's logical top-k slot.

        ``topk_ids`` has shape ``(num_tokens, top_k)`` and carries logical
        expert IDs from the router. The kernel samples per (token, slot)
        from the categorical distribution ``log2phy_prob[logical_id, :]``,
        mapping the chosen slot through ``log2phy_map`` to the physical
        expert ID. Returns a tensor of the same shape and dtype as
        ``topk_ids``.
        """
        ...
