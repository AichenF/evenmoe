#include <ATen/cuda/CUDAContext.h>
#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <string_view>
#include <utility>

#include "kernels/api.cuh"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

namespace {

void check_cuda_tensor(const torch::Tensor& tensor,
                       const c10::ScalarType dtype,
                       const std::string_view name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(const torch::Tensor& reference,
                       const torch::Tensor& tensor,
                       const std::string_view name) {
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on ", reference.device());
}

void solve_placement_impl(torch::Tensor expert_loads,
                          torch::Tensor expert_loads_per_rank,
                          torch::Tensor physical_to_logical_map,
                          torch::Tensor logical_to_physical_map,
                          torch::Tensor logical_replica_counts,
                          torch::Tensor logical_instance_quota,
                          torch::Tensor logical_instance_quota_prefix,
                          torch::Tensor rank_quota_prefix,
                          int num_ranks,
                          int num_local_master_experts,
                          int num_local_redundant_experts,
                          int num_nvl_ranks,
                          float balance_threshold,
                          int32_t min_tokens_per_replica,
                          bool allow_zero_master_quota,
                          bool locality_aware,
                          float oracle_eps,
                          int kernel_stage,
                          int rank) {
    check_cuda_tensor(expert_loads, c10::kInt, "expert_loads");
    check_cuda_tensor(expert_loads_per_rank, c10::kInt, "expert_loads_per_rank");
    check_cuda_tensor(physical_to_logical_map, c10::kInt, "physical_to_logical_map");
    check_cuda_tensor(logical_to_physical_map, c10::kInt, "logical_to_physical_map");
    check_cuda_tensor(logical_replica_counts, c10::kInt, "logical_replica_counts");
    check_cuda_tensor(logical_instance_quota, c10::kInt, "logical_instance_quota");
    check_cuda_tensor(logical_instance_quota_prefix, c10::kInt, "logical_instance_quota_prefix");
    check_cuda_tensor(rank_quota_prefix, c10::kInt, "rank_quota_prefix");

    for (const auto& named_tensor : {
             std::pair{"expert_loads_per_rank", expert_loads_per_rank},
             std::pair{"physical_to_logical_map", physical_to_logical_map},
             std::pair{"logical_to_physical_map", logical_to_physical_map},
             std::pair{"logical_replica_counts", logical_replica_counts},
             std::pair{"logical_instance_quota", logical_instance_quota},
             std::pair{"logical_instance_quota_prefix", logical_instance_quota_prefix},
             std::pair{"rank_quota_prefix", rank_quota_prefix},
         }) {
        check_same_device(expert_loads, named_tensor.second, named_tensor.first);
    }

    TORCH_CHECK(num_ranks > 1, "num_ranks must be greater than one");
    TORCH_CHECK(num_local_master_experts > 0, "num_local_master_experts must be positive");
    TORCH_CHECK(num_local_redundant_experts > 0,
                "num_local_redundant_experts must be positive");
    TORCH_CHECK(num_nvl_ranks > 0 && num_ranks % num_nvl_ranks == 0,
                "num_nvl_ranks must be positive and divide num_ranks");
    TORCH_CHECK(rank >= 0 && rank < num_ranks, "rank is outside the EP group");
    TORCH_CHECK(kernel_stage == 0 || kernel_stage == 1, "kernel_stage must be 0 or 1");
    TORCH_CHECK(min_tokens_per_replica >= 0, "min_tokens_per_replica must be non-negative");

    const int64_t logical_64 =
        static_cast<int64_t>(num_ranks) * num_local_master_experts;
    const int64_t physical_64 =
        static_cast<int64_t>(num_ranks) *
        (static_cast<int64_t>(num_local_master_experts) + num_local_redundant_experts);
    TORCH_CHECK(logical_64 <= std::numeric_limits<int>::max(),
                "the logical expert count exceeds the CUDA kernel limit");
    TORCH_CHECK(physical_64 <= std::numeric_limits<int>::max(),
                "the physical expert count exceeds the CUDA kernel limit");
    const int logical = static_cast<int>(logical_64);
    TORCH_CHECK(expert_loads.dim() == 1 && expert_loads.numel() == logical_64,
                "expert_loads must have shape [num_ranks * num_local_master_experts]");
    TORCH_CHECK(expert_loads_per_rank.sizes() ==
                    torch::IntArrayRef({static_cast<int64_t>(num_ranks), logical_64}),
                "expert_loads_per_rank has the wrong shape");
    TORCH_CHECK(physical_to_logical_map.dim() == 1 &&
                    physical_to_logical_map.numel() == physical_64,
                "physical_to_logical_map has the wrong shape");
    TORCH_CHECK(logical_to_physical_map.sizes() ==
                    torch::IntArrayRef({logical_64, num_ranks}),
                "logical_to_physical_map has the wrong shape");
    TORCH_CHECK(logical_replica_counts.dim() == 1 &&
                    logical_replica_counts.numel() == logical_64,
                "logical_replica_counts has the wrong shape");
    TORCH_CHECK(logical_instance_quota.sizes() ==
                    torch::IntArrayRef({logical_64, num_ranks}),
                "logical_instance_quota has the wrong shape");
    TORCH_CHECK(logical_instance_quota_prefix.sizes() ==
                    torch::IntArrayRef({logical_64, num_ranks}),
                "logical_instance_quota_prefix has the wrong shape");
    TORCH_CHECK(rank_quota_prefix.sizes() ==
                    torch::IntArrayRef({logical_64, num_ranks}),
                "rank_quota_prefix has the wrong shape");

    const c10::cuda::CUDAGuard device_guard(expert_loads.device());
    const auto stream = at::cuda::getCurrentCUDAStream(expert_loads.get_device());
    ultra_ep::kernels::solve_placement(expert_loads.data_ptr<int32_t>(),
                                       expert_loads_per_rank.data_ptr<int32_t>(),
                                       physical_to_logical_map.data_ptr<int32_t>(),
                                       logical_to_physical_map.data_ptr<int32_t>(),
                                       logical_replica_counts.data_ptr<int32_t>(),
                                       logical_instance_quota.data_ptr<int32_t>(),
                                       logical_instance_quota_prefix.data_ptr<int32_t>(),
                                       rank_quota_prefix.data_ptr<int32_t>(),
                                       stream.stream(),
                                       logical,
                                       num_ranks,
                                       num_local_master_experts,
                                       num_local_redundant_experts,
                                       num_nvl_ranks,
                                       num_ranks,
                                       balance_threshold,
                                       min_tokens_per_replica,
                                       allow_zero_master_quota,
                                       locality_aware,
                                       oracle_eps,
                                       kernel_stage,
                                       rank);
}

void solve_placement(torch::Tensor expert_loads,
                     torch::Tensor expert_loads_per_rank,
                     torch::Tensor physical_to_logical_map,
                     torch::Tensor logical_to_physical_map,
                     torch::Tensor logical_replica_counts,
                     torch::Tensor logical_instance_quota,
                     torch::Tensor logical_instance_quota_prefix,
                     torch::Tensor rank_quota_prefix,
                     int num_ranks,
                     int num_local_master_experts,
                     int num_local_redundant_experts,
                     int num_nvl_ranks,
                     float balance_threshold,
                     int32_t min_tokens_per_replica,
                     bool allow_zero_master_quota,
                     bool locality_aware,
                     float oracle_eps,
                     int kernel_stage,
                     int rank) {
    solve_placement_impl(expert_loads,
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
                         balance_threshold,
                         min_tokens_per_replica,
                         allow_zero_master_quota,
                         locality_aware,
                         oracle_eps,
                         kernel_stage,
                         rank);
}

void reroute_sparse(torch::Tensor logical_topk_ids,
                    torch::Tensor routed_topk_ids,
                    torch::Tensor logical_to_physical_map,
                    torch::Tensor logical_replica_counts,
                    torch::Tensor rank_quota_prefix,
                    torch::Tensor counters) {
    const auto topk_dtype = logical_topk_ids.scalar_type();
    TORCH_CHECK(topk_dtype == c10::kInt || topk_dtype == c10::kLong,
                "logical_topk_ids must use torch.int32 or torch.int64");
    check_cuda_tensor(logical_topk_ids, topk_dtype, "logical_topk_ids");
    check_cuda_tensor(routed_topk_ids, c10::kLong, "routed_topk_ids");
    at::assert_no_overlap(logical_topk_ids, routed_topk_ids);
    check_cuda_tensor(logical_to_physical_map, c10::kInt, "logical_to_physical_map");
    check_cuda_tensor(logical_replica_counts, c10::kInt, "logical_replica_counts");
    check_cuda_tensor(rank_quota_prefix, c10::kInt, "rank_quota_prefix");
    check_cuda_tensor(counters, c10::kInt, "counters");
    for (const auto& named_tensor : {
             std::pair{"routed_topk_ids", routed_topk_ids},
             std::pair{"logical_to_physical_map", logical_to_physical_map},
             std::pair{"logical_replica_counts", logical_replica_counts},
             std::pair{"rank_quota_prefix", rank_quota_prefix},
             std::pair{"counters", counters},
         }) {
        check_same_device(logical_topk_ids, named_tensor.second, named_tensor.first);
    }

    TORCH_CHECK(logical_topk_ids.dim() == 2,
                "logical_topk_ids must have shape [num_tokens, top_k]");
    TORCH_CHECK(routed_topk_ids.sizes() == logical_topk_ids.sizes(),
                "routed_topk_ids must match logical_topk_ids shape");
    TORCH_CHECK(logical_to_physical_map.dim() == 2,
                "logical_to_physical_map must have shape [num_experts, max_replicas]");
    const int64_t logical_64 = logical_to_physical_map.size(0);
    const int64_t max_replicas_64 = logical_to_physical_map.size(1);
    const int64_t num_tokens_64 = logical_topk_ids.size(0);
    const int64_t top_k_64 = logical_topk_ids.size(1);
    TORCH_CHECK(logical_64 > 0, "logical_to_physical_map must contain experts");
    TORCH_CHECK(max_replicas_64 > 0,
                "logical_to_physical_map must contain replica slots");
    TORCH_CHECK(logical_64 <= std::numeric_limits<int>::max() &&
                    max_replicas_64 <= std::numeric_limits<int>::max(),
                "the placement shape exceeds the CUDA kernel limit");
    TORCH_CHECK(logical_64 <=
                    std::numeric_limits<int>::max() / max_replicas_64,
                "the flattened placement exceeds the CUDA kernel limit");
    TORCH_CHECK(num_tokens_64 <= std::numeric_limits<int>::max() &&
                    top_k_64 <= std::numeric_limits<int>::max() &&
                    (top_k_64 == 0 ||
                     num_tokens_64 <= std::numeric_limits<int>::max() / top_k_64),
                "the TopK shape exceeds the CUDA kernel limit");
    const int logical = static_cast<int>(logical_64);
    const int max_replicas = static_cast<int>(max_replicas_64);
    const int num_tokens = static_cast<int>(num_tokens_64);
    const int top_k = static_cast<int>(top_k_64);
    TORCH_CHECK(logical_replica_counts.dim() == 1 &&
                    logical_replica_counts.numel() == logical_64,
                "logical_replica_counts has the wrong shape");
    TORCH_CHECK(rank_quota_prefix.sizes() ==
                    torch::IntArrayRef({logical_64, max_replicas_64}),
                "rank_quota_prefix has the wrong shape");
    TORCH_CHECK(counters.dim() == 1 && counters.numel() == logical_64,
                "counters has the wrong shape");

    const c10::cuda::CUDAGuard device_guard(logical_topk_ids.device());
    const auto stream =
        at::cuda::getCurrentCUDAStream(logical_topk_ids.get_device());
    if (topk_dtype == c10::kInt) {
        ultra_ep::kernels::run_sparse_reroute_quota(
            logical_topk_ids.data_ptr<int32_t>(),
            routed_topk_ids.data_ptr<int64_t>(),
            logical_to_physical_map.data_ptr<int32_t>(),
            logical_replica_counts.data_ptr<int32_t>(),
            rank_quota_prefix.data_ptr<int32_t>(),
            counters.data_ptr<int32_t>(),
            num_tokens,
            top_k,
            logical,
            max_replicas,
            stream.stream());
    } else {
        ultra_ep::kernels::run_sparse_reroute_quota(
            logical_topk_ids.data_ptr<int64_t>(),
            routed_topk_ids.data_ptr<int64_t>(),
            logical_to_physical_map.data_ptr<int32_t>(),
            logical_replica_counts.data_ptr<int32_t>(),
            rank_quota_prefix.data_ptr<int32_t>(),
            counters.data_ptr<int32_t>(),
            num_tokens,
            top_k,
            logical,
            max_replicas,
            stream.stream());
    }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "MLB placement and rerouting kernels derived from UltraEP v1.0.0";
    module.def("solve_placement", &solve_placement);
    module.def("reroute_sparse", &reroute_sparse);
}
