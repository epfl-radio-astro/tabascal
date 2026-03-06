#pragma once

// Unified CUDA/HIP compatibility header.
// Include this instead of raw cuda_runtime.h / hip_runtime.h in GPU sources.

#ifdef __HIPCC__

#include <hip/hip_complex.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipcub/block/block_reduce.hpp>
#include <hipcub/warp/warp_reduce.hpp>
namespace cub = hipcub;

// Map CUDA complex types and functions to their HIP equivalents.
// hipDoubleComplex has the same .x/.y layout as cuDoubleComplex.
#define cuDoubleComplex hipDoubleComplex
#define cuCmul hipCmul
#define cuCadd hipCadd
#define cuConj hipConj

// Runtime type/function aliases not provided by HIP's headers.
using cudaStream_t = hipStream_t;
using cudaDeviceProp = hipDeviceProp_t;
#define cudaGetLastError hipGetLastError
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetDeviceProperties hipGetDeviceProperties

// HIP cooperative groups does not support cg::reduce with custom operators.
// Implement an equivalent via explicit warp-shuffle reduction.
template <typename TileT>
__device__ inline cuDoubleComplex tab_cg_reduce_add(TileT &tile,
                                                    cuDoubleComplex val) {
  for (int offset = tile.size() / 2; offset > 0; offset >>= 1) {
    val.x += tile.shfl_down(val.x, offset);
    val.y += tile.shfl_down(val.y, offset);
  }
  return val;
}

#else // CUDA

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>

template <typename TileT>
__device__ inline cuDoubleComplex tab_cg_reduce_add(TileT &tile,
                                                    cuDoubleComplex val) {
  return cg::reduce(tile, val, cuCadd);
}

#endif
