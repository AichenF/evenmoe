#include "api.cuh"
#include "launch.cuh"

namespace ultra_ep::kernels {

namespace {

__device__ __forceinline__ int quota_replica_index(
    int ordinal,
    int logical_id,
    int replica_count,
    int max_replicas,
    const int32_t* __restrict__ rank_quota_prefix) {
    constexpr int prefetch_replicas = 8;
    const int row_offset = logical_id * max_replicas;
    const int prefetch_count = min(replica_count, prefetch_replicas);
    int replica_index = 0;
    int local_prefix[prefetch_replicas];

#pragma unroll
    for (int index = 0; index < prefetch_replicas; ++index) {
        local_prefix[index] =
            index < prefetch_count ? rank_quota_prefix[row_offset + index] : 0;
    }
    const int quota_total = replica_count <= prefetch_replicas
        ? local_prefix[replica_count - 1]
        : rank_quota_prefix[row_offset + replica_count - 1];
    if (quota_total <= 0) {
        return ordinal % replica_count;
    }
    if (ordinal >= quota_total) {
        ordinal = ordinal < quota_total * 2 ? ordinal - quota_total
                                           : ordinal % quota_total;
    }
#pragma unroll
    for (int index = 0; index < prefetch_replicas; ++index) {
        if (index < prefetch_count) {
            replica_index += ordinal >= local_prefix[index] ? 1 : 0;
        }
    }
    for (int index = prefetch_replicas; index < replica_count; ++index) {
        replica_index += ordinal >= rank_quota_prefix[row_offset + index] ? 1 : 0;
    }
    return min(replica_index, max(replica_count - 1, 0));
}

template <typename IdType>
__global__ __launch_bounds__(256) void sparse_quota_reroute_kernel(
    const IdType* __restrict__ logical_topk_ids,
    int64_t* __restrict__ routed_topk_ids,
    const int32_t* __restrict__ logical_to_physical_map,
    const int32_t* __restrict__ logical_replica_counts,
    const int32_t* __restrict__ rank_quota_prefix,
    int32_t* __restrict__ counters,
    int num_entries,
    int max_replicas,
    int num_experts) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_entries;
         idx += gridDim.x * blockDim.x) {
        const int64_t logical_id = static_cast<int64_t>(logical_topk_ids[idx]);
        if (logical_id < 0 || logical_id >= num_experts) {
            routed_topk_ids[idx] = logical_id;
            continue;
        }
        const int replica_count =
            min(max(logical_replica_counts[logical_id], 0), max_replicas);
        if (replica_count <= 0) {
            routed_topk_ids[idx] = logical_id;
            continue;
        }
        const int ordinal = atomicAdd(&counters[logical_id], 1);
        const int replica_index = quota_replica_index(
            ordinal,
            static_cast<int>(logical_id),
            replica_count,
            max_replicas,
            rank_quota_prefix);
        routed_topk_ids[idx] =
            logical_to_physical_map[logical_id * max_replicas + replica_index];
    }
}

template <typename IdType>
void run_sparse_reroute_quota_impl(const IdType* logical_topk_ids,
                                   int64_t* routed_topk_ids,
                                   const int32_t* logical_to_physical_map,
                                   const int32_t* logical_replica_counts,
                                   const int32_t* rank_quota_prefix,
                                   int32_t* counters,
                                   int num_tokens,
                                   int top_k,
                                   int num_global_logical_experts,
                                   int max_replicas,
                                   cudaStream_t stream) {
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(
        counters, 0, num_global_logical_experts * sizeof(int32_t), stream));
    const int num_entries = num_tokens * top_k;
    if (num_entries == 0) {
        return;
    }
    constexpr int block_size = 256;
    const int num_blocks = min(256, (num_entries - 1) / block_size + 1);
    const auto config = make_launch_config(dim3(num_blocks), dim3(block_size), stream);
    launch_kernel(sparse_quota_reroute_kernel<IdType>,
                  config,
                  logical_topk_ids,
                  routed_topk_ids,
                  logical_to_physical_map,
                  logical_replica_counts,
                  rank_quota_prefix,
                  counters,
                  num_entries,
                  max_replicas,
                  num_global_logical_experts);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
}

}  // namespace

void run_sparse_reroute_quota(const int32_t* logical_topk_ids,
                              int64_t* routed_topk_ids,
                              const int32_t* logical_to_physical_map,
                              const int32_t* logical_replica_counts,
                              const int32_t* rank_quota_prefix,
                              int32_t* counters,
                              int num_tokens,
                              int top_k,
                              int num_global_logical_experts,
                              int max_replicas,
                              cudaStream_t stream) {
    run_sparse_reroute_quota_impl(logical_topk_ids,
                                  routed_topk_ids,
                                  logical_to_physical_map,
                                  logical_replica_counts,
                                  rank_quota_prefix,
                                  counters,
                                  num_tokens,
                                  top_k,
                                  num_global_logical_experts,
                                  max_replicas,
                                  stream);
}

void run_sparse_reroute_quota(const int64_t* logical_topk_ids,
                              int64_t* routed_topk_ids,
                              const int32_t* logical_to_physical_map,
                              const int32_t* logical_replica_counts,
                              const int32_t* rank_quota_prefix,
                              int32_t* counters,
                              int num_tokens,
                              int top_k,
                              int num_global_logical_experts,
                              int max_replicas,
                              cudaStream_t stream) {
    run_sparse_reroute_quota_impl(logical_topk_ids,
                                  routed_topk_ids,
                                  logical_to_physical_map,
                                  logical_replica_counts,
                                  rank_quota_prefix,
                                  counters,
                                  num_tokens,
                                  top_k,
                                  num_global_logical_experts,
                                  max_replicas,
                                  stream);
}

}  // namespace ultra_ep::kernels
