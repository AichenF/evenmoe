from __future__ import annotations

from typing import Protocol, runtime_checkable

from moe_load_balancer.core.types import TensorLike, WeightTransferRequest


@runtime_checkable
class ExpertWeightTransfer(Protocol):
    """Framework-neutral L3 policy: physically move expert weight data to
    match a placement L1 has already decided, before L2 dispatches tokens
    against it.

    Distinct from L1 (decides *what* the placement should be) and L2
    (routes tokens against a placement already reflected in memory) -- L3
    is what makes memory actually match the decision. Optional: most L1
    policies rely on the framework's own generic weight mover instead
    (vLLM's point-to-point ``rearrange_expert_weights_inplace``, for
    instance) and never need this at all. A policy that does needs it
    because, like UltraEP, its L1 and L2 solve depend on placement staying
    fresh far more often than a generic mover can afford to run.
    """

    def register_weights(self, expert_weights: TensorLike) -> None:
        """One-time setup: register this rank's own per-layer expert weight
        tensors so :meth:`transfer` has real memory to move data between.

        ``expert_weights`` is one entry per layer, in whatever per-layer
        shape the framework's own model representation already uses (for
        instance, vLLM's ``MixtureOfExperts.expert_weights``) -- this
        protocol does not require converting it to an MLB-owned layout
        first.
        """
        ...

    def transfer(self, request: WeightTransferRequest) -> None:
        """Physically move weight data for one layer to match a newly
        decided placement.

        Blocks until the transfer is confirmed complete and safe for
        compute to read: synchronization (waiting on whatever event or
        collective the underlying transfer mechanism needs) is this
        method's own responsibility, never left for the caller to get
        right or wrong.
        """
        ...
