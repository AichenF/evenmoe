"""Framework-neutral local count helpers used by L2 routing policies."""

from __future__ import annotations


def count_logical_experts(
    logical_topk_ids: object,
    num_logical_experts: int,
    dtype: object = None,
) -> object:
    """Count valid logical TopK slots without counting ``-1`` padding.

    ``dtype`` only controls the returned tensor's dtype (default float32,
    unchanged): the count itself always accumulates in float32 -- the CUDA
    path is a Triton kernel whose ``atomic_add`` increment is a float literal,
    so counting directly into an integer buffer is not something this has
    been verified to do correctly, unlike a final cast.
    """

    if num_logical_experts <= 0:
        raise ValueError("num_logical_experts must be positive.")

    import torch

    if not isinstance(logical_topk_ids, torch.Tensor):
        raise TypeError("Logical expert counting requires a torch.Tensor.")
    counts = torch.zeros(
        num_logical_experts,
        dtype=torch.float32,
        device=logical_topk_ids.device,
    )
    if logical_topk_ids.is_cuda:
        from moe_load_balancer.kernels.logical_count import count_logical_experts_cuda

        count_logical_experts_cuda(logical_topk_ids, counts)
        return counts if dtype is None else counts.to(dtype)

    valid = (logical_topk_ids >= 0) & (logical_topk_ids < num_logical_experts)
    if valid.any():
        counts.copy_(
            torch.bincount(
                logical_topk_ids[valid].to(torch.int64),
                minlength=num_logical_experts,
            ).to(torch.float32)
        )
    return counts if dtype is None else counts.to(dtype)
