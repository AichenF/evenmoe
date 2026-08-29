#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ultra_ep::kernels {

void solve_placement(const int32_t* expert_loads,
                     const int32_t* expert_loads_per_rank,
                     int32_t* physical_to_logical_map,
                     int32_t* logical_to_physical_map,
                     int32_t* logical_replica_counts,
                     int32_t* logical_instance_quota,
                     int32_t* logical_instance_quota_prefix,
                     int32_t* rank_quota_prefix,
                     cudaStream_t stream,
                     int num_global_logical_experts,
                     int num_ranks,
                     int num_local_master_experts,
                     int num_local_redundant_experts,
                     int num_nvl_ranks,
                     int max_replicas_dim,
                     float balance_threshold,
                     int32_t min_tokens_per_replica,
                     bool allow_zero_master_quota,
                     bool locality_aware,
                     float oracle_eps,
                     int kernel_stage,
                     int rank_quota_source_rank);

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
                              cudaStream_t stream);

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
                              cudaStream_t stream);

}  // namespace ultra_ep::kernels
