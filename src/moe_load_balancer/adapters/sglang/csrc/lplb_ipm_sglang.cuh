// Copyright 2023-2026 SGLang Team
// Licensed under the Apache License, Version 2.0.
//
// Thin SGLang JIT ABI wrapper around MLB's framework-neutral CUDA solver.

#include <dlpack/dlpack.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>
#include <tvm/ffi/container/tensor.h>

#include "../../../policies/l2/csrc/lplb_ipm.cuh"

namespace {

template <int NC, int NV, int BLOCK_DIM, int SM_VER, int NUM_ITERS>
void ipm_solve(
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView c,
    tvm::ffi::TensorView result) {
  using namespace host;
  SymbolicDevice symbolic_device;
  TensorMatcher({NC, NV}).with_dtype<float>().with_device<kDLCUDA>(symbolic_device).verify(a);
  TensorMatcher({NC}).with_dtype<float>().with_device<kDLCUDA>(symbolic_device).verify(b);
  TensorMatcher({NV}).with_dtype<float>().with_device<kDLCUDA>(symbolic_device).verify(c);
  TensorMatcher({NV}).with_dtype<float>().with_device<kDLCUDA>(symbolic_device).verify(result);
  const DLDevice device = symbolic_device.unwrap();
  const size_t shared_memory_bytes =
      sizeof(moe_load_balancer::lplb::ipm_smem<NC, NV>);
  using Kernel = void (*)(float*, const float*, const float*, const float*);
  Kernel kernel = moe_load_balancer::lplb::ipm_solve_kernel<
      NC, NV, BLOCK_DIM, SM_VER, NUM_ITERS>;
  if (shared_memory_bytes > 48 * 1024) {
    cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_memory_bytes));
  }
  LaunchKernel(1, BLOCK_DIM, device, shared_memory_bytes)(
      kernel,
      static_cast<float*>(result.data_ptr()),
      static_cast<const float*>(a.data_ptr()),
      static_cast<const float*>(b.data_ptr()),
      static_cast<const float*>(c.data_ptr()));
}

}  // namespace
