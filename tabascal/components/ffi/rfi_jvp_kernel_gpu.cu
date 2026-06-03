#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unistd.h>

#include "gpu_compat.h"
#include "tensor.hpp"
#include "util_gpu.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace tabascal {
namespace gpu {

template <typename T, int BLOCK_SIZE, typename INT_T>
__global__ void __launch_bounds__(BLOCK_SIZE)
    rfi_jvp_kernel(T scale, Tensor1D<const int *, INT_T> a1,
                   Tensor1D<const int *, INT_T> a2,
                   Tensor4D<const typename gpu_complex_traits<T>::complex_t *, INT_T> rfi_amp_fine,
                   Tensor4D<const typename gpu_complex_traits<T>::complex_t *, INT_T> rfi_amp_fine_grad,
                   Tensor4D<const T *, INT_T> rfi_phase,
                   Tensor4D<const T *, INT_T> rfi_phase_grad,
                   Tensor3D<typename gpu_complex_traits<T>::complex_t *, INT_T> rfi_grad) {

  using traits = gpu_complex_traits<T>;
  using complex_t = typename traits::complex_t;

  using BlockReduce_t = cub::BlockReduce<T, BLOCK_SIZE>;

  __shared__ typename BlockReduce_t::TempStorage temp_storage_1;
  __shared__ typename BlockReduce_t::TempStorage temp_storage_2;

  // rfi_amp_fine layout: (n_ant, n_freq, n_time, n_rfi * n_int_f * n_int_t)
  const auto n_freq = rfi_amp_fine.shape[1];
  const auto n_time = rfi_amp_fine.shape[2];
  const auto n_reduce = rfi_amp_fine.shape[3];
  const auto n_bl = a1.shape[0];

  assert(a1.shape[0] == a2.shape[0]);
  assert(a1.shape[0] == rfi_grad.shape[0]);
  assert(rfi_phase.shape[0] == rfi_amp_fine.shape[0]);
  assert(rfi_phase.shape[1] == rfi_amp_fine.shape[1]);
  assert(rfi_phase.shape[2] == rfi_amp_fine.shape[2]);
  assert(rfi_phase.shape[3] == rfi_amp_fine.shape[3]);
  assert(rfi_grad.shape[1] == n_freq);
  assert(rfi_grad.shape[2] == n_time);


  for (INT_T i_bl = blockIdx.y; i_bl < n_bl; i_bl += gridDim.y) {
    INT_T i_a1 = a1(i_bl);
    INT_T i_a2 = a2(i_bl);

    for (INT_T i_f = blockIdx.z; i_f < n_freq; i_f += gridDim.z) {

      for (INT_T i_t = blockIdx.x; i_t < n_time; i_t += gridDim.x) {
        complex_t sum{0, 0};

        const auto ptr_rfi_amp_1 = &rfi_amp_fine(i_a1, i_f, i_t, 0);
        const auto ptr_rfi_amp_2 = &rfi_amp_fine(i_a2, i_f, i_t, 0);

        const auto ptr_rfi_amp_grad_1 = &rfi_amp_fine_grad(i_a1, i_f, i_t, 0);
        const auto ptr_rfi_amp_grad_2 = &rfi_amp_fine_grad(i_a2, i_f, i_t, 0);

        const auto ptr_rfi_phase_1 = &rfi_phase(i_a1, i_f, i_t, 0);
        const auto ptr_rfi_phase_2 = &rfi_phase(i_a2, i_f, i_t, 0);

        const auto ptr_rfi_phase_grad_1 = &rfi_phase_grad(i_a1, i_f, i_t, 0);
        const auto ptr_rfi_phase_grad_2 = &rfi_phase_grad(i_a2, i_f, i_t, 0);

        // block reduction
        for (INT_T i_red = threadIdx.x; i_red < n_reduce; i_red += BLOCK_SIZE) {

          const auto val_rfi_amp_1 = ptr_rfi_amp_1[i_red];
          const auto val_rfi_amp_2 = ptr_rfi_amp_2[i_red];

          const auto val_rfi_amp_grad_1 = ptr_rfi_amp_grad_1[i_red];
          const auto val_rfi_amp_grad_2 = ptr_rfi_amp_grad_2[i_red];

          const auto val_rfi_phase_1 = ptr_rfi_phase_1[i_red];
          const auto val_rfi_phase_2 = ptr_rfi_phase_2[i_red];

          const auto val_rfi_phase_grad_1 = ptr_rfi_phase_grad_1[i_red];
          const auto val_rfi_phase_grad_2 = ptr_rfi_phase_grad_2[i_red];

          complex_t e_val;
          traits::sincos_(val_rfi_phase_1 - val_rfi_phase_2, &e_val.y, &e_val.x);

          // g1 = val_e * (val_rfi_amp_grad_1 * conj(val_rfi_amp_2) +
          //               val_rfi_amp_1 * conj(val_rfi_amp_grad_2))
          auto g1 = traits::mul(val_rfi_amp_grad_1, traits::conj(val_rfi_amp_2));
          g1 = traits::add(g1, traits::mul(val_rfi_amp_1, traits::conj(val_rfi_amp_grad_2)));
          g1 = traits::mul(g1, e_val);

          // g2 = val_e * i * (phase_grad_1 - phase_grad_2) * val_rfi_amp_1 * conj(val_rfi_amp_2)
          complex_t g2{val_rfi_phase_grad_1 - val_rfi_phase_grad_2, T(0)};
          g2 = traits::mul(g2, traits::mul(val_rfi_amp_1, traits::conj(val_rfi_amp_2)));
          g2 = traits::mul(g2, complex_t{-e_val.y, e_val.x});

          sum = traits::add(sum, traits::add(g1, g2));
        }

        sum.x = BlockReduce_t(temp_storage_1).Sum(sum.x);
        sum.y = BlockReduce_t(temp_storage_2).Sum(sum.y);
        __syncthreads(); // required for reuse of temp_storage

        if (threadIdx.x == 0) {
          sum.x *= scale;
          sum.y *= scale;

          rfi_grad(i_bl, i_f, i_t) = sum;
        }
      }
    }
  }
}

template <typename T, typename INT_T, ffi::DataType AMP_DT, ffi::DataType PHASE_DT>
ffi::Error calc_rfi_jvp_gpu_dispatch(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> a2,
    ffi::Buffer<AMP_DT, 6> rfi_amp_fine, ffi::Buffer<AMP_DT, 6> rfi_amp_fine_grad,
    ffi::Buffer<PHASE_DT, 6> rfi_phase, ffi::Buffer<PHASE_DT, 6> rfi_phase_grad,
    ffi::Result<ffi::BufferR3<AMP_DT>> rfi_grad) {
  using complex_t = typename gpu_complex_traits<T>::complex_t;

  if (a1.dimensions()[0] != a2.dimensions()[0]) {
    return ffi::Error::InvalidArgument(
        "Expected a1 and a2 to have the same size");
  }

  for (int i = 0; i < 6; ++i) {
    if (rfi_amp_fine.dimensions()[i] != rfi_phase.dimensions()[i]) {
      return ffi::Error::InvalidArgument(
          "Expected rfi_amp_fine and rfi_phase to have the same shape");
    }
  }

  if (rfi_grad->dimensions()[0] != a1.dimensions()[0]) {
    return ffi::Error::InvalidArgument(
        "Expected rfi_grad and a1 to have the same number of baselines");
  }

  if (rfi_grad->dimensions()[1] != rfi_amp_fine.dimensions()[1]) {
    return ffi::Error::InvalidArgument(
        "Expected rfi_grad and rfi_amp_fine to have the same number of "
        "frequencies");
  }

  if (rfi_grad->dimensions()[2] != rfi_amp_fine.dimensions()[2]) {
    return ffi::Error::InvalidArgument(
        "Expected rfi_grad and rfi_amp_fine to have the same number of times");
  }

  for (int i = 0; i < 6; ++i) {
    if (rfi_amp_fine.dimensions()[i] != rfi_amp_fine_grad.dimensions()[i]) {
      return ffi::Error::InvalidArgument(
          "Expected rfi_amp_fine and rfi_amp_fine_grad to have the same shape");
    }
    if (rfi_phase.dimensions()[i] != rfi_phase_grad.dimensions()[i]) {
      return ffi::Error::InvalidArgument(
          "Expected rfi_phase and rfi_phase_grad to have the same shape");
    }
  }

  Tensor1D<const int *, INT_T> a1_tensor(a1.typed_data(), a1.dimensions()[0]);
  Tensor1D<const int *, INT_T> a2_tensor(a2.typed_data(), a2.dimensions()[0]);
  Tensor4D<const complex_t *, INT_T> rfi_amp_fine_tensor(
      (const complex_t *)rfi_amp_fine.typed_data(),
      rfi_amp_fine.dimensions()[0], rfi_amp_fine.dimensions()[1],
      rfi_amp_fine.dimensions()[2],
      rfi_amp_fine.dimensions()[3] * rfi_amp_fine.dimensions()[4] *
          rfi_amp_fine.dimensions()[5]);

  Tensor4D<const complex_t *, INT_T> rfi_amp_fine_grad_tensor(
      (const complex_t *)rfi_amp_fine_grad.typed_data(),
      rfi_amp_fine_grad.dimensions()[0], rfi_amp_fine_grad.dimensions()[1],
      rfi_amp_fine_grad.dimensions()[2],
      rfi_amp_fine_grad.dimensions()[3] * rfi_amp_fine_grad.dimensions()[4] *
          rfi_amp_fine_grad.dimensions()[5]);

  Tensor4D<const T *, INT_T> rfi_phase_tensor(
      rfi_phase.typed_data(), rfi_phase.dimensions()[0],
      rfi_phase.dimensions()[1], rfi_phase.dimensions()[2],
      rfi_phase.dimensions()[3] * rfi_phase.dimensions()[4] *
          rfi_phase.dimensions()[5]);

  Tensor4D<const T *, INT_T> rfi_phase_grad_tensor(
      rfi_phase_grad.typed_data(), rfi_phase_grad.dimensions()[0],
      rfi_phase_grad.dimensions()[1], rfi_phase_grad.dimensions()[2],
      rfi_phase_grad.dimensions()[3] * rfi_phase_grad.dimensions()[4] *
          rfi_phase_grad.dimensions()[5]);

  Tensor3D<complex_t *, INT_T> rfi_grad_tensor(
      (complex_t *)rfi_grad->typed_data(), rfi_grad->dimensions()[0],
      rfi_grad->dimensions()[1], rfi_grad->dimensions()[2]);

  const INT_T n_int_f = (INT_T)rfi_amp_fine.dimensions()[4];
  const INT_T n_int_t = (INT_T)rfi_amp_fine.dimensions()[5];

  const T scale = T(1) / T(n_int_t * n_int_f);

  const auto n_time = rfi_grad_tensor.shape[2];
  const auto n_bl = a1.dimensions()[0];
  const auto n_freq = rfi_grad_tensor.shape[1];

  auto grid = create_clamped_grid(n_time, n_bl, n_freq);

  if (rfi_phase_tensor.shape[3] / 2 < 32) {
    constexpr int block_size = 32;
    dim3 block(block_size);
    rfi_jvp_kernel<T, block_size, INT_T><<<grid, block, 0, stream>>>(
        scale, a1_tensor, a2_tensor, rfi_amp_fine_tensor,
        rfi_amp_fine_grad_tensor, rfi_phase_tensor, rfi_phase_grad_tensor,
        rfi_grad_tensor);
  } else if (rfi_phase_tensor.shape[3] / 2 < 64) {
    constexpr int block_size = 64;
    dim3 block(block_size);
    rfi_jvp_kernel<T, block_size, INT_T><<<grid, block, 0, stream>>>(
        scale, a1_tensor, a2_tensor, rfi_amp_fine_tensor,
        rfi_amp_fine_grad_tensor, rfi_phase_tensor, rfi_phase_grad_tensor,
        rfi_grad_tensor);
  } else if (rfi_phase_tensor.shape[3] / 2 < 128) {
    constexpr int block_size = 128;
    dim3 block(block_size);
    rfi_jvp_kernel<T, block_size, INT_T><<<grid, block, 0, stream>>>(
        scale, a1_tensor, a2_tensor, rfi_amp_fine_tensor,
        rfi_amp_fine_grad_tensor, rfi_phase_tensor, rfi_phase_grad_tensor,
        rfi_grad_tensor);
  } else {
    constexpr int block_size = 256;
    dim3 block(block_size);
    rfi_jvp_kernel<T, block_size, INT_T><<<grid, block, 0, stream>>>(
        scale, a1_tensor, a2_tensor, rfi_amp_fine_tensor,
        rfi_amp_fine_grad_tensor, rfi_phase_tensor, rfi_phase_grad_tensor,
        rfi_grad_tensor);
  }

  const auto status = cudaGetLastError();
  if (status != cudaSuccess) {
    return ffi::Error::Internal(std::string("GPU kernel launch error: ") +
                                cudaGetErrorString(status));
  }

  return ffi::Error::Success();
}

template <typename T, ffi::DataType AMP_DT, ffi::DataType PHASE_DT>
ffi::Error calc_rfi_jvp_gpu_impl_tmpl(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a1_sorter, ffi::BufferR1<ffi::S32> a1_start,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32> a2_sorter,
    ffi::BufferR1<ffi::S32> a2_start, ffi::Buffer<AMP_DT, 6> rfi_amp_fine,
    ffi::Buffer<AMP_DT, 6> rfi_amp_fine_grad,
    ffi::Buffer<PHASE_DT, 6> rfi_phase, ffi::Buffer<PHASE_DT, 6> rfi_phase_grad,
    ffi::Result<ffi::BufferR3<AMP_DT>> rfi_grad) {
  constexpr std::int64_t max32 = std::numeric_limits<std::int32_t>::max();
  // use 32 bit indexing if possible
  if (a1.element_count() < max32 && a2.element_count() < max32 &&
      rfi_amp_fine.element_count() < max32 &&
      rfi_phase.element_count() < max32 && rfi_grad->element_count() < max32) {
    return calc_rfi_jvp_gpu_dispatch<T, std::int32_t, AMP_DT, PHASE_DT>(
        stream, a1, a2, rfi_amp_fine, rfi_amp_fine_grad, rfi_phase,
        rfi_phase_grad, rfi_grad);
  } else {
    return calc_rfi_jvp_gpu_dispatch<T, std::int64_t, AMP_DT, PHASE_DT>(
        stream, a1, a2, rfi_amp_fine, rfi_amp_fine_grad, rfi_phase,
        rfi_phase_grad, rfi_grad);
  }
}

ffi::Error calc_rfi_jvp_gpu_f32_impl(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a1_sorter, ffi::BufferR1<ffi::S32> a1_start,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32> a2_sorter,
    ffi::BufferR1<ffi::S32> a2_start, ffi::Buffer<ffi::C64, 6> rfi_amp_fine,
    ffi::Buffer<ffi::C64, 6> rfi_amp_fine_grad,
    ffi::Buffer<ffi::F32, 6> rfi_phase, ffi::Buffer<ffi::F32, 6> rfi_phase_grad,
    ffi::Result<ffi::BufferR3<ffi::C64>> rfi_grad) {
  return calc_rfi_jvp_gpu_impl_tmpl<float, ffi::C64, ffi::F32>(
      stream, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine,
      rfi_amp_fine_grad, rfi_phase, rfi_phase_grad, rfi_grad);
}

ffi::Error calc_rfi_jvp_gpu_f64_impl(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a1_sorter, ffi::BufferR1<ffi::S32> a1_start,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32> a2_sorter,
    ffi::BufferR1<ffi::S32> a2_start, ffi::Buffer<ffi::C128, 6> rfi_amp_fine,
    ffi::Buffer<ffi::C128, 6> rfi_amp_fine_grad,
    ffi::Buffer<ffi::F64, 6> rfi_phase, ffi::Buffer<ffi::F64, 6> rfi_phase_grad,
    ffi::Result<ffi::BufferR3<ffi::C128>> rfi_grad) {
  return calc_rfi_jvp_gpu_impl_tmpl<double, ffi::C128, ffi::F64>(
      stream, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine,
      rfi_amp_fine_grad, rfi_phase, rfi_phase_grad, rfi_grad);
}

// Type aliases to avoid commas inside XLA_FFI_DEFINE_HANDLER_SYMBOL macro args.
using rfi_amp_f32_t = ffi::Buffer<ffi::C64, 6>;
using rfi_phase_f32_t = ffi::Buffer<ffi::F32, 6>;
using rfi_amp_f64_t = ffi::Buffer<ffi::C128, 6>;
using rfi_phase_f64_t = ffi::Buffer<ffi::F64, 6>;

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_jvp_gpu_f32, calc_rfi_jvp_gpu_f32_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<rfi_amp_f32_t>()
                                  .Arg<rfi_amp_f32_t>()
                                  .Arg<rfi_phase_f32_t>()
                                  .Arg<rfi_phase_f32_t>()
                                  .Ret<ffi::BufferR3<ffi::C64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_jvp_gpu_f64, calc_rfi_jvp_gpu_f64_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<rfi_amp_f64_t>()
                                  .Arg<rfi_amp_f64_t>()
                                  .Arg<rfi_phase_f64_t>()
                                  .Arg<rfi_phase_f64_t>()
                                  .Ret<ffi::BufferR3<ffi::C128>>());

} // namespace gpu
} // namespace tabascal
