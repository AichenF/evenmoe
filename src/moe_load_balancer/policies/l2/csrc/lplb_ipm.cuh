// Copyright 2023-2026 SGLang Team
// Licensed under the Apache License, Version 2.0.
// Adapted from DeepSeek-AI/LPLB's minilp.cu.
//
// Single-SM Interior Point Method for:
//   min c^T x, subject to A x = b and x >= 0.

#include <cublasdx.hpp>
#include <cstdint>

namespace moe_load_balancer::lplb {

template <int NC, int NV>
struct ipm_smem {
  float b[NC];
  float a[NC][NV];
  float c[NV];
  float ax2[NC][NV];
  float ax2a[NC][NC];
  float x[NV];
  float ax2c[NC];
  float r[NV];
  float d[NV];
  float alpha;
  bool avail_flag;
};

template <int N, int BLOCK_DIM>
__device__ __forceinline__ void cholesky_factor(float a[N][N]) {
  const int tid = threadIdx.x;
  for (int k = 0; k < N; k++) {
    if (tid == 0) {
      a[k][k] = sqrtf(fmaxf(a[k][k], 1e-12f));
    }
    __syncthreads();
    const float pivot = a[k][k];
    for (int i = k + 1 + tid; i < N; i += BLOCK_DIM) {
      a[i][k] /= pivot;
    }
    __syncthreads();
    for (int idx = tid; idx < N * N; idx += BLOCK_DIM) {
      const int i = idx / N;
      const int j = idx % N;
      if (j > k && i >= j && i < N) {
        a[i][j] -= a[i][k] * a[j][k];
      }
    }
    __syncthreads();
  }
}

template <int N, int BLOCK_DIM>
__device__ __forceinline__ void cholesky_apply(const float a[N][N], float b[N]) {
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int i = 0; i < N; i++) {
      float sum = b[i];
      for (int j = 0; j < i; j++) {
        sum -= a[i][j] * b[j];
      }
      b[i] = sum / a[i][i];
    }
    for (int i = N - 1; i >= 0; i--) {
      float sum = b[i];
      for (int j = i + 1; j < N; j++) {
        sum -= a[j][i] * b[j];
      }
      b[i] = sum / a[i][i];
    }
  }
  __syncthreads();
}

template <int N, int SM_VER, int BLOCK_DIM>
__device__ __forceinline__ void cholesky_solve(float a[N][N], float b[N]) {
  cholesky_factor<N, BLOCK_DIM>(a);
  cholesky_apply<N, BLOCK_DIM>(a, b);
}

template <int M, int N, int K, int SM_VER, int BLOCK_DIM>
__device__ __forceinline__ void matmul_nt(float* a, float* b, float* c) {
  decltype(
      cublasdx::Size<M, N, K>() +
      cublasdx::Function<cublasdx::function::MM>() +
      cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>() +
      cublasdx::SM<SM_VER>() + cublasdx::Block() +
      cublasdx::BlockDim<BLOCK_DIM>())()
      .execute(1.f, a, b, 0.f, c);
}

template <int M, int N, int K, int SM_VER, int BLOCK_DIM>
__device__ __forceinline__ void matmul_nn(float* a, float* b, float* c) {
  decltype(
      cublasdx::Size<M, N, K>() +
      cublasdx::Function<cublasdx::function::MM>() +
      cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>() +
      cublasdx::SM<SM_VER>() + cublasdx::Block() +
      cublasdx::BlockDim<BLOCK_DIM>())()
      .execute(1.f, a, b, 0.f, c);
}

template <int NC, int NV, int BLOCK_DIM, int SM_VER, int NUM_ITERS>
__global__ void ipm_solve_kernel(
    float* __restrict__ result,
    const float* __restrict__ input_a,
    const float* __restrict__ input_b,
    const float* __restrict__ input_c) {
  using SMem = ipm_smem<NC, NV>;
  extern __shared__ unsigned char raw_smem[];
  SMem* smem = reinterpret_cast<SMem*>(raw_smem);
  const int tid = threadIdx.x;
  const int dim = blockDim.x;
  auto& a = smem->a;
  auto& b = smem->b;
  auto& c = smem->c;

  for (int i = tid; i < NC * NV; i += dim) {
    const int row = i / NV;
    const int column = i % NV;
    a[row][column] = input_a[i];
  }
  for (int i = tid; i < NC; i += dim) {
    b[i] = input_b[i];
  }
  for (int i = tid; i < NV; i += dim) {
    c[i] = input_c[i];
  }
  __syncthreads();

  auto& ax2 = smem->ax2;
  auto& ax2a = smem->ax2a;
  auto& x = smem->x;
  auto& ax2c = smem->ax2c;
  auto& r = smem->r;
  auto& d = smem->d;
  auto& alpha = smem->alpha;
  float d_max = 0.f;
  float max_residual = 0.f;

  for (int j = tid; j < NV; j += dim) {
    x[j] = 1.f;
  }
  __syncthreads();

  for (int step = 0; step < NUM_ITERS; step++) {
    for (int offset = tid; offset < NC * NV; offset += dim) {
      const int row = offset / NV;
      const int column = offset % NV;
      ax2[row][column] = a[row][column] * x[column] * x[column];
    }
    __syncthreads();

    matmul_nt<NC, NC, NV, SM_VER, BLOCK_DIM>(ax2[0], a[0], ax2a[0]);
    matmul_nt<NC, 1, NV, SM_VER, BLOCK_DIM>(ax2[0], c, ax2c);
    cholesky_solve<NC, SM_VER, BLOCK_DIM>(ax2a, ax2c);
    matmul_nn<1, NV, NC, SM_VER, BLOCK_DIM>(ax2c, a[0], r);

    if (tid < 32) {
      d_max = 0.f;
      for (int j = tid; j < NV; j += 32) {
        const float value = x[j] * (c[j] - r[j]);
        d[j] = value;
        d_max = fmaxf(d_max, value);
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        d_max = fmaxf(d_max, __shfl_xor_sync(0xffffffff, d_max, offset));
      }
      if (tid == 0) {
        alpha = d_max > 1e-9f ? 0.999f / d_max : 1.0f;
      }
    }
    __syncthreads();
    for (int j = tid; j < NV; j += dim) {
      x[j] *= 1.f - alpha * d[j];
    }
    __syncthreads();
  }

  matmul_nt<NC, 1, NV, SM_VER, BLOCK_DIM>(a[0], x, ax2c);
  if (tid < 32) {
    max_residual = 0.f;
    for (int i = tid; i < NC; i += 32) {
      max_residual = fmaxf(max_residual, fabsf(ax2c[i] - b[i]));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      max_residual = fmaxf(
          max_residual,
          __shfl_down_sync(0xffffffff, max_residual, offset));
    }
  }

  auto& available = smem->avail_flag;
  if (tid == 0) {
    available = d_max < 0.1f && x[NV - 1] >= 0.f &&
                x[NV - 1] < 1e-4f && max_residual < 0.05f;
  }
  __syncthreads();
  for (int i = tid; i < NV; i += dim) {
    result[i] = available ? x[i] : 0.5f;
  }
}

}  // namespace moe_load_balancer::lplb
