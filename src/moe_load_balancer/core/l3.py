from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

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

    def register_weights(
        self,
        expert_weights: TensorLike,
        *,
        ep_size: int,
        num_local_master_experts: int,
        num_local_redundant_experts: int,
        process_group: Any,
    ) -> None:
        """One-time setup: register this rank's own per-layer expert weight
        tensors so :meth:`transfer` has real memory to move data between.

        ``expert_weights`` is one entry per layer, in whatever per-layer
        shape the framework's own model representation already uses (for
        instance, vLLM's ``MixtureOfExperts.expert_weights``) -- this
        protocol does not require converting it to an MLB-owned layout
        first. Master vs. redundant is not derivable from that shape alone
        (both live in the same per-layer tensor, split by convention, not
        by anything the tensor's own shape records), so the three topology
        keywords carry what :class:`~moe_load_balancer.core.types.PlacementTopology`
        would if this were called with one -- named after the same
        parameters an L1 quota policy already computes internally, not
        reusing that type itself, because there is no placement request to
        attach one to at registration time.

        ``process_group`` is a raw ``torch.distributed`` process group
        (vLLM's ``get_ep_group().device_group``, SGLang's own equivalent,
        ...). Unlike :class:`~moe_load_balancer.core.l2.RoutingCollectives`,
        this is not wrapped behind an MLB-defined interface: the only
        implementation today hands it straight to a third-party transfer
        library that needs the real group object for its own bootstrap and
        one-sided transfers, so there is no MLB-owned operation to abstract
        here -- just a handle the framework owns and this protocol passes
        through unopened.
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
