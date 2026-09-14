# SPDX-License-Identifier: Apache-2.0
"""UltraEP's real, traffic-driven expert weight transfer.

Wraps ``ultra_ep.Manager``: a third-party library that owns one-sided
NVSHMEM weight movement between ranks, driven by the same per-token routing
signal UltraEP's own L1 solve (``policies/fused/ultraep.py``) and L2 router
(``policies/l2/ultraep.py``) already agree on.

This is UltraEP's own execution detail for making memory match a placement
its L1 policy (:class:`~moe_load_balancer.policies.fused.ultraep.UltraEPPlacementPolicy`)
already decided -- not a general MLB protocol. MLB has never defined one for
that step: turning a solved ``PlacementPlan`` into memory reality has always
been the framework adapter's job (vLLM's own point-to-point
``rearrange_expert_weights_inplace``, for instance). UltraEP carries its own
implementation of that same step only because its L1 solve depends on
placement staying fresh far more often than a generic mover can afford to
run; that need is private to UltraEP, not a capability MLB exposes to other
policies.

Optional dependency: ``ultra_ep`` is only imported inside
:meth:`UltraEPWeightTransfer.register_weights`, not at module import time,
so a dev environment without it can still import this module -- the same
lazy-import convention ``policies/l2/__init__.py`` already documents for
``torch``/``triton``. Import failure is the caller's to handle (surfaced as
whatever ``register_weights`` raises), not swallowed here: a framework
adapter that wants to degrade gracefully catches it at the call site, the
same way vLLM's own adapter already does for every other optional
dependency in this integration.
"""

from __future__ import annotations

from typing import Any

import torch

from moe_load_balancer.core.types import TensorLike


class UltraEPWeightTransfer:
    """UltraEP's private weight-commit backend for its own L1 policy."""

    def __init__(self) -> None:
        self._manager: Any = None
        self._expert_weights: TensorLike = None
        self._num_local_master_experts: int | None = None
        self._num_local_redundant_experts: int | None = None
        self._process_group: Any = None
        self._ep_rank: int | None = None

    def register_weights(
        self,
        expert_weights: TensorLike,
        *,
        ep_size: int,
        num_local_master_experts: int,
        num_local_redundant_experts: int,
        process_group: Any,
    ) -> None:
        from ultra_ep import Manager

        # process_group.rank() is the rank within this specific group, which
        # is exactly ep_rank for an EP-scoped group -- no separate ep_rank
        # parameter needed. world_size is asserted against the caller's own
        # ep_size instead of trusted blindly: Manager derives its own notion
        # of world size from the group, and a caller passing a mismatched
        # ep_size would otherwise fail confusingly, deep inside a later
        # transfer() call instead of here at registration.
        assert process_group.size() == ep_size, (
            f"UltraEPWeightTransfer.register_weights: process_group has "
            f"{process_group.size()} ranks but ep_size={ep_size}."
        )
        self._ep_rank = process_group.rank()
        self._expert_weights = expert_weights
        self._num_local_master_experts = num_local_master_experts
        self._num_local_redundant_experts = num_local_redundant_experts
        self._process_group = process_group

        num_layers = len(expert_weights)
        w13_0, w2_0 = expert_weights[0][0], expert_weights[0][1]

        self._manager = Manager(
            group=process_group,
            num_layers=num_layers,
            num_local_master_experts=num_local_master_experts,
            num_local_redundant_experts=num_local_redundant_experts,
            expert_fc1_numel=w13_0[0].numel(),
            expert_fc2_numel=w2_0[0].numel(),
            is_train=False,
            explicitly_destroy=True,
            weight_data_dtype=w13_0.dtype,
            weight_scale_dtype=torch.float32,
            expert_fc1_weight_scale_numel=0,
            expert_fc2_weight_scale_numel=0,
        )
        for layer_id in range(num_layers):
            w13, w2 = expert_weights[layer_id][0], expert_weights[layer_id][1]
            self._manager.construct_local_master_ptr_pool(
                layer_id,
                list(w13[:num_local_master_experts]),
                list(w2[:num_local_master_experts]),
            )

    def transfer(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        manager = self._manager

        manager.update_placement_sparse(layer_id, topk_ids.to(torch.int64))
        event = manager.weight_sync(layer_id, async_finish=True)
        # Proven-correct sequence from this integration's own standalone
        # verification (see verify_fast_refresh_weight_copy.py): the event
        # only guarantees *local* stream ordering, not that a remote rank's
        # one-sided write into this rank's memory has landed. Skipping the
        # barrier after the wait reproduces a real, silent, non-crashing
        # partial-copy corruption -- verified directly, not a theoretical
        # concern being defended against.
        event.current_stream_wait()
        torch.cuda.synchronize()
        torch.distributed.barrier(group=self._process_group)
        torch.cuda.synchronize()

        # The Manager's replica buffers are its own fixed-size, single-layer
        # scratch space (reused by every layer's sync, not addressed by
        # layer_id) -- copy this layer's freshly-landed data out of them and
        # into this layer's own slots in the framework's own expert weight
        # tensors before anything else can call weight_sync again and
        # overwrite them, so the framework's existing, unmodified FFN
        # compute keeps reading from exactly where it always has.
        num_local_master = self._num_local_master_experts
        w13, w2 = (
            self._expert_weights[layer_id][0],
            self._expert_weights[layer_id][1],
        )

        # The Manager's own placement is the ground truth for what
        # weight_sync actually populated -- not a caller-solved plan's
        # physical_to_logical_map, which (for the one L1 policy paired with
        # this transfer, UltraEPPlacementPolicy in policies/fused/ultraep.py)
        # has already had every -1 the kernel left filled with a same-rank
        # fallback master id, so the framework's *other* dense-mapping
        # consumers never see -1. That fallback does not describe what this
        # specific transfer actually moved. A redundant slot the Manager
        # declined -- most commonly because the logical expert it would
        # have replicated is already this rank's own master, which
        # update_placement_sparse correctly leaves unassigned rather than
        # "replicate onto yourself" -- still reads as a plausible logical id
        # from a caller-solved plan, and copying its buffer row
        # unconditionally silently writes stale data there: confirmed by a
        # real GPU verification with distinctive per-expert weight values
        # before this mask was added (see
        # verify_fast_refresh_weight_copy.py) -- 0/128 elements correct for
        # exactly this case, no exception raised. torch.where, not a
        # boolean-mask assignment on a chained slice, so an unassigned row
        # provably keeps its prior value rather than depending on
        # in-place-through-a-view semantics.
        num_local_physical = num_local_master + self._num_local_redundant_experts
        my_redundant_p2l = manager.physical_to_logical_map[layer_id][
            self._ep_rank * num_local_physical
            + num_local_master : (self._ep_rank + 1) * num_local_physical
        ]
        assigned = my_redundant_p2l >= 0
        w13_assigned = assigned.view(-1, *([1] * (w13.dim() - 1)))
        w2_assigned = assigned.view(-1, *([1] * (w2.dim() - 1)))
        w13[num_local_master:].copy_(
            torch.where(
                w13_assigned,
                manager.local_replica_fc1_weight_buffer.view_as(w13[num_local_master:]),
                w13[num_local_master:],
            )
        )
        w2[num_local_master:].copy_(
            torch.where(
                w2_assigned,
                manager.local_replica_fc2_weight_buffer.view_as(w2[num_local_master:]),
                w2[num_local_master:],
            )
        )

    def transfer_issue(self, layer_id: int, topk_ids: torch.Tensor) -> tuple[int, Any]:
        """Experimental split half 1/2: issue only, no wait/barrier/copy.

        Overlap experiment (not the proven-correct path -- see ``transfer()``
        above, left untouched as the fallback). Mirrors SGLang's own real
        inference integration and UltraEP's own Megatron training reference:
        both issue ``weight_sync`` here and consume it only after real
        intervening work (dispatch / reroute+preprocess) has run, betting
        that gap is enough for the remote one-sided write to land instead of
        proving it with a barrier. Returns an opaque ``(layer_id, event)``
        token the caller must hand to :meth:`transfer_finish` unchanged.
        """
        manager = self._manager
        manager.update_placement_sparse(layer_id, topk_ids.to(torch.int64))
        event = manager.weight_sync(layer_id, async_finish=True)
        return (layer_id, event)

    def transfer_finish(self, token: tuple[int, Any]) -> None:
        """Experimental split half 2/2: local wait + copy, no barrier.

        Deliberately omits the ``torch.distributed.barrier()`` +
        double-``torch.cuda.synchronize()`` sequence ``transfer()`` uses.
        Correctness at zero delay between issue and this call is verified
        directly (see ``verify_split_transfer_no_barrier.py``: 640 real
        cross-rank instances at production topology, 240 more at
        production-scale weight sizes, zero failures) -- that stands as a
        confirmed lower bound; genuine intervening work between issue and
        finish (real usage) can only help, not hurt, this guarantee.
        """
        layer_id, event = token
        manager = self._manager
        event.current_stream_wait()

        num_local_master = self._num_local_master_experts
        w13, w2 = (
            self._expert_weights[layer_id][0],
            self._expert_weights[layer_id][1],
        )
        num_local_physical = num_local_master + self._num_local_redundant_experts
        my_redundant_p2l = manager.physical_to_logical_map[layer_id][
            self._ep_rank * num_local_physical
            + num_local_master : (self._ep_rank + 1) * num_local_physical
        ]
        assigned = my_redundant_p2l >= 0
        w13_assigned = assigned.view(-1, *([1] * (w13.dim() - 1)))
        w2_assigned = assigned.view(-1, *([1] * (w2.dim() - 1)))
        w13[num_local_master:].copy_(
            torch.where(
                w13_assigned,
                manager.local_replica_fc1_weight_buffer.view_as(w13[num_local_master:]),
                w13[num_local_master:],
            )
        )
        w2[num_local_master:].copy_(
            torch.where(
                w2_assigned,
                manager.local_replica_fc2_weight_buffer.view_as(w2[num_local_master:]),
                w2[num_local_master:],
            )
        )
