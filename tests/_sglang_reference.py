"""SGLang's replica-selection code, lifted verbatim as a comparison reference.

Copied rather than imported so the alignment tests do not depend on a SGLang
install, and so a drift on either side shows up as a test failure rather than
as a silently different policy sharing a name. Source:
sglang/srt/eplb/expert_location.py and expert_location_dispatch.py.
"""

import random
from typing import List
import torch


def _logical_to_all_physical_raw(
    logical_to_all_physical_map, layer_id: int, logical_expert_id: int
) -> List[int]:
    return [
        physical_expert_id
        for physical_expert_id in logical_to_all_physical_map[
            layer_id, logical_expert_id
        ].tolist()
        if physical_expert_id != -1
    ]



def _compute_gpu_id_of_physical_expert(
    physical_expert_id: int, num_local_gpu_physical_experts: int
) -> int:
    return physical_expert_id // num_local_gpu_physical_experts



def _compute_node_id_of_physical_expert(
    physical_expert_id: int, num_local_host_physical_experts: int
) -> int:
    return physical_expert_id // num_local_host_physical_experts



def _find_nearest_expert(
    candidate_physical_expert_ids: List[int],
    num_local_gpu_physical_experts: int,
    moe_ep_rank: int,
    num_gpus_per_node: int,
    num_local_node_physical_experts: int,
) -> int:
    # 1. If only one candidate, return it directly
    if len(candidate_physical_expert_ids) == 1:
        return candidate_physical_expert_ids[0]

    # 2. Prefer same-GPU experts
    same_gpu_physical_expert_ids = [
        physical_expert_id
        for physical_expert_id in candidate_physical_expert_ids
        if _compute_gpu_id_of_physical_expert(
            physical_expert_id, num_local_gpu_physical_experts
        )
        == moe_ep_rank
    ]
    if len(same_gpu_physical_expert_ids) > 0:
        return same_gpu_physical_expert_ids[0]

    # 3. Otherwise, prefer same-node experts
    node_rank = moe_ep_rank // num_gpus_per_node
    same_node_physical_expert_ids = [
        physical_expert_id
        for physical_expert_id in candidate_physical_expert_ids
        if _compute_node_id_of_physical_expert(
            physical_expert_id, num_local_node_physical_experts
        )
        == node_rank
    ]
    if len(same_node_physical_expert_ids) > 0:
        return same_node_physical_expert_ids[0]

    # 4. At last, leave it as -1 to indicate not found.
    return -1



def _fair_choices(arr: List, k: int, r: random.Random) -> List:
    quotient, remainder = divmod(k, len(arr))
    ans = arr * quotient + r.sample(arr, k=remainder)
    r.shuffle(ans)
    return ans



def compute_logical_to_rank_dispatch_physical_map(
    server_args,
    logical_to_all_physical_map: torch.Tensor,
    ep_size: int,
    num_physical_experts: int,
    ep_rank: int,
    seed: int = 42,
):
    r = random.Random(seed)

    device = logical_to_all_physical_map.device
    logical_to_all_physical_map = logical_to_all_physical_map.cpu()

    num_local_gpu_physical_experts = num_physical_experts // ep_size
    num_gpus_per_node = server_args.ep_size // server_args.nnodes
    num_local_node_physical_experts = num_local_gpu_physical_experts * num_gpus_per_node
    num_layers, num_logical_experts, _ = logical_to_all_physical_map.shape
    dtype = logical_to_all_physical_map.dtype

    result_list = [
        [[-1] * num_logical_experts for _ in range(num_layers)] for _ in range(ep_size)
    ]

    for layer_id in range(num_layers):
        for logical_expert_id in range(num_logical_experts):
            candidate_physical_expert_ids = _logical_to_all_physical_raw(
                logical_to_all_physical_map, layer_id, logical_expert_id
            )

            remaining_ranks = []
            for moe_ep_rank in range(ep_size):
                val = _find_nearest_expert(
                    candidate_physical_expert_ids=candidate_physical_expert_ids,
                    num_local_gpu_physical_experts=num_local_gpu_physical_experts,
                    moe_ep_rank=moe_ep_rank,
                    num_gpus_per_node=num_gpus_per_node,
                    num_local_node_physical_experts=num_local_node_physical_experts,
                )

                result_list[moe_ep_rank][layer_id][logical_expert_id] = val
                if val == -1:
                    remaining_ranks.append(moe_ep_rank)

            if remaining_ranks:
                choices = _fair_choices(
                    candidate_physical_expert_ids, k=len(remaining_ranks), r=r
                )
                for moe_ep_rank, choice in zip(remaining_ranks, choices, strict=True):
                    result_list[moe_ep_rank][layer_id][logical_expert_id] = choice

    logical_to_rank_dispatch_physical_map = torch.tensor(result_list, dtype=dtype)
    assert torch.all(logical_to_rank_dispatch_physical_map != -1)

    return logical_to_rank_dispatch_physical_map[ep_rank, :, :].to(device)



def topk_ids_logical_to_physical_dynamic(topk_ids, candidates, counts):
    """SGLang's `_topk_ids_logical_to_physical_dynamic`, verbatim."""
    original_shape = topk_ids.shape
    original_dtype = topk_ids.dtype
    flat = topk_ids.flatten()
    chosen = (
        torch.randint(0, 65536, flat.shape, dtype=torch.int32, device=flat.device)
        % counts[flat]
    )
    out = candidates[flat, chosen]
    return out.view(original_shape).to(original_dtype)


def topk_ids_logical_to_physical_static(topk_ids, rank_dispatch_map):
    """SGLang's `_topk_ids_logical_to_physical_static`, verbatim."""
    return rank_dispatch_map[topk_ids]
