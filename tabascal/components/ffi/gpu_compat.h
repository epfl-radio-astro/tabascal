#pragma once

// Unified CUDA/HIP compatibility header.
// Include this instead of raw cuda_runtime.h / hip_runtime.h in GPU sources.

#ifdef __HIPCC__

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipcub/block/block_reduce.hpp>
#include <hipcub/warp/warp_reduce.hpp>
namespace cub = hipcub;

// Map CUDA complex types and functions to their HIP equivalents.
// hipDoubleComplex / hipFloatComplex have the same .x/.y layout as their
// cu* counterparts.
#define cuDoubleComplex hipDoubleComplex
#define cuFloatComplex hipFloatComplex
#define cuCmul hipCmul
#define cuCmulf hipCmulf
#define cuCadd hipCadd
#define cuCaddf hipCaddf
#define cuConj hipConj
#define cuConjf hipConjf

// Runtime type/function aliases not provided by HIP's headers.
using cudaStream_t = hipStream_t;
using cudaDeviceProp = hipDeviceProp_t;
#define cudaGetLastError hipGetLastError
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetDeviceProperties hipGetDeviceProperties

#else // CUDA

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>

#endif

namespace tabascal {
namespace gpu {

template <typename T> struct gpu_complex_traits;

template <> struct gpu_complex_traits<double> {
  using complex_t = cuDoubleComplex;
  __device__ static inline complex_t mul(complex_t a, complex_t b) { return cuCmul(a, b); }
  __device__ static inline complex_t add(complex_t a, complex_t b) { return cuCadd(a, b); }
  __device__ static inline complex_t conj(complex_t a) { return cuConj(a); }
  __device__ static inline void sincos_(double x, double *s, double *c) {
    sincos(x, s, c);
  }
};

template <> struct gpu_complex_traits<float> {
  using complex_t = cuFloatComplex;
  __device__ static inline complex_t mul(complex_t a, complex_t b) { return cuCmulf(a, b); }
  __device__ static inline complex_t add(complex_t a, complex_t b) { return cuCaddf(a, b); }
  __device__ static inline complex_t conj(complex_t a) { return cuConjf(a); }
  __device__ static inline void sincos_(float x, float *s, float *c) {
    sincosf(x, s, c);
  }
};

} // namespace gpu
} // namespace tabascal
