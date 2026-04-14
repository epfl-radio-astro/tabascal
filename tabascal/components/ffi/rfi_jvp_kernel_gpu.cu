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

template <int BLOCK_SIZE, int WARP_SIZE, typename INT_T>
__global__ void __launch_bounds__(BLOCK_SIZE)
    rfi_jvp_kernel(Tensor1D<const int *, INT_T> a1,
                   Tensor1D<const int *, INT_T> a2,
                   Tensor4D<const cuDoubleComplex *, INT_T> rfi_amp_fine,
                   Tensor4D<const cuDoubleComplex *, INT_T> rfi_amp_fine_grad,
                   Tensor4D<const double *, INT_T> rfi_phase,
                   Tensor4D<const double *, INT_T> rfi_phase_grad,
                   Tensor3D<cuDoubleComplex *, INT_T> rfi_grad) {

  static_assert(BLOCK_SIZE % WARP_SIZE == 0);
  constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

  using WarpReduce_t = cub::WarpReduce<double>;

  __shared__ typename WarpReduce_t::TempStorage temp_storage_1[WARPS_PER_BLOCK];
  __shared__ typename WarpReduce_t::TempStorage temp_storage_2[WARPS_PER_BLOCK];

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;

  const int n_warps_global = gridDim.x * WARPS_PER_BLOCK;

  const auto n_rfi = rfi_amp_fine.shape[0];
  const auto n_freq_fine = rfi_amp_fine.shape[2];
  const auto n_time_fine = rfi_amp_fine.shape[3];
  const auto n_bl = a1.shape[0];
  const auto n_time = rfi_grad.shape[2];
  const auto n_freq = rfi_grad.shape[1];

  assert(a1.shape[0] == a2.shape[0]);
  assert(a1.shape[0] == rfi_grad.shape[0]);
  assert(rfi_phase.shape[0] == rfi_amp_fine.shape[0]);
  assert(rfi_phase.shape[1] == rfi_amp_fine.shape[1]);
  assert(rfi_phase.shape[2] == rfi_amp_fine.shape[2]);
  assert(rfi_phase.shape[3] == rfi_amp_fine.shape[3]);

  const auto n_int_t = n_time_fine / n_time;
  const auto n_int_f = n_freq_fine / n_freq;
  const double n_int_inv = 1.f / double(n_int_t * n_int_f);

  for (INT_T i_bl = blockIdx.y; i_bl < n_bl; i_bl += gridDim.y) {
    INT_T i_a1 = a1(i_bl);
    INT_T i_a2 = a2(i_bl);

    for (INT_T i_f = blockIdx.z; i_f < n_freq; i_f += gridDim.z) {
      const auto i_f_fine_begin = i_f * n_int_f;

      // warp reduction
      for (INT_T i_t = blockIdx.x * WARPS_PER_BLOCK + warp_id; i_t < n_time;
           i_t += n_warps_global) {
        cuDoubleComplex sum{0, 0};

        const auto i_t_fine_begin = i_t * n_int_t;

        for (INT_T i_rfi = 0; i_rfi < n_rfi; ++i_rfi) {

          for (INT_T i_f_fine = i_f_fine_begin;
               i_f_fine < i_f_fine_begin + n_int_f; ++i_f_fine) {

            for (INT_T i_t_fine = i_t_fine_begin + lane_id;
                 i_t_fine < i_t_fine_begin + n_int_t; i_t_fine += WARP_SIZE) {

              const auto val_rfi_amp_1 =
                  rfi_amp_fine(i_rfi, i_a1, i_f_fine, i_t_fine);
              const auto val_rfi_amp_2 =
                  rfi_amp_fine(i_rfi, i_a2, i_f_fine, i_t_fine);

              const auto val_rfi_amp_grad_1 =
                  rfi_amp_fine_grad(i_rfi, i_a1, i_f_fine, i_t_fine);
              const auto val_rfi_amp_grad_2 =
                  rfi_amp_fine_grad(i_rfi, i_a2, i_f_fine, i_t_fine);

              const auto val_rfi_phase_1 =
                  rfi_phase(i_rfi, i_a1, i_f_fine, i_t_fine);
              const auto val_rfi_phase_2 =
                  rfi_phase(i_rfi, i_a2, i_f_fine, i_t_fine);

              const auto val_rfi_phase_grad_1 =
                  rfi_phase_grad(i_rfi, i_a1, i_f_fine, i_t_fine);
              const auto val_rfi_phase_grad_2 =
                  rfi_phase_grad(i_rfi, i_a2, i_f_fine, i_t_fine);

              cuDoubleComplex e_val;
              sincos(val_rfi_phase_1 - val_rfi_phase_2, &e_val.y, &e_val.x);

              // const auto g1 =
              //     val_e * (val_rfi_amp_grad_1 * std::conj(val_rfi_amp_2) +
              //              val_rfi_amp_1 * std::conj(val_rfi_amp_grad_2));

              auto g1 = cuCmul(val_rfi_amp_grad_1, cuConj(val_rfi_amp_2));

              g1 =
                  cuCadd(g1, cuCmul(val_rfi_amp_1, cuConj(val_rfi_amp_grad_2)));

              g1 = cuCmul(g1, e_val);

              // const auto g2 = val_e * std::complex<double>(0, 1) *
              //                 (val_rfi_phase_grad_1 - val_rfi_phase_grad_2) *
              //                 val_rfi_amp_1 * std::conj(val_rfi_amp_2);

              auto g2 = cuDoubleComplex{
                  val_rfi_phase_grad_1 - val_rfi_phase_grad_2, 0};

              g2 = cuCmul(g2, cuCmul(val_rfi_amp_1, cuConj(val_rfi_amp_2)));

              g2 = cuCmul(g2, cuDoubleComplex{-e_val.y, e_val.x});

              sum = cuCadd(sum, cuCadd(g1, g2));
            }
          }
        }
        sum.x = WarpReduce_t(temp_storage_1[warp_id]).Sum(sum.x);
        sum.y = WarpReduce_t(temp_storage_2[warp_id]).Sum(sum.y);
        __syncwarp();

        if (lane_id == 0) {
          sum.x *= n_int_inv;
          sum.y *= n_int_inv;

          rfi_grad(i_bl, i_f, i_t) = sum;
        }
      }
    }
  }
}

using rfi_amp_fine_t = ffi::Buffer<ffi::C128, 6>;
using rfi_phase_t = ffi::Buffer<ffi::F64, 6>;

// A wrapper function providing the interface between the XLA FFI call and our
// library function `ComputeRFI` above. This function handles the batch
// dimensions by calling `ComputeRFI` within a loop.
template <typename INT_T>
ffi::Error calc_rfi_jvp_gpu_dispatch(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> a2,
    rfi_amp_fine_t rfi_amp_fine, rfi_amp_fine_t rfi_amp_fine_grad,
    rfi_phase_t rfi_phase, rfi_phase_t rfi_phase_grad,
    ffi::ResultBufferR3<ffi::C128> rfi_grad) {
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

  if (rfi_grad->dimensions()[1] != rfi_amp_fine.dimensions()[2]) {
    return ffi::Error::InvalidArgument(
        "Expected rfi_grad and rfi_amp_fine to have the same number of "
        "frequencies");
  }

  if (rfi_grad->dimensions()[2] != rfi_amp_fine.dimensions()[4]) {
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
  Tensor4D<const cuDoubleComplex *, INT_T> rfi_amp_fine_tensor(
      (const cuDoubleComplex *)rfi_amp_fine.typed_data(),
      rfi_amp_fine.dimensions()[0], rfi_amp_fine.dimensions()[1],
      rfi_amp_fine.dimensions()[2] * rfi_amp_fine.dimensions()[3],
      rfi_amp_fine.dimensions()[4] * rfi_amp_fine.dimensions()[5]);

  Tensor4D<const cuDoubleComplex *, INT_T> rfi_amp_fine_grad_tensor(
      (const cuDoubleComplex *)rfi_amp_fine_grad.typed_data(),
      rfi_amp_fine_grad.dimensions()[0], rfi_amp_fine_grad.dimensions()[1],
      rfi_amp_fine_grad.dimensions()[2] * rfi_amp_fine_grad.dimensions()[3],
      rfi_amp_fine_grad.dimensions()[4] * rfi_amp_fine_grad.dimensions()[5]);

  Tensor4D<const double *, INT_T> rfi_phase_tensor(
      rfi_phase.typed_data(), rfi_phase.dimensions()[0],
      rfi_phase.dimensions()[1],
      rfi_phase.dimensions()[2] * rfi_phase.dimensions()[3],
      rfi_phase.dimensions()[4] * rfi_phase.dimensions()[5]);

  Tensor4D<const double *, INT_T> rfi_phase_grad_tensor(
      rfi_phase_grad.typed_data(), rfi_phase_grad.dimensions()[0],
      rfi_phase_grad.dimensions()[1],
      rfi_phase_grad.dimensions()[2] * rfi_phase_grad.dimensions()[3],
      rfi_phase_grad.dimensions()[4] * rfi_phase_grad.dimensions()[5]);

  Tensor3D<cuDoubleComplex *, INT_T> rfi_grad_tensor(
      (cuDoubleComplex *)rfi_grad->typed_data(), rfi_grad->dimensions()[0],
      rfi_grad->dimensions()[1], rfi_grad->dimensions()[2]);

  // Cooperative group size. Must be power of 2. Used to iterate over n_int and
  // reduce result. If 32, equal to warp size on Nvidia for fast reduce
  // operation.

  constexpr int block_size = 256;
  const auto n_time = rfi_grad_tensor.shape[2];
  const auto n_bl = a1.dimensions()[0];
  const auto n_freq = rfi_grad_tensor.shape[1];

  const int warp_size = get_device_prop().warpSize;

  dim3 block(block_size);
  auto n_warps = block.x / warp_size;

  auto grid =
      create_clamped_grid((n_time + n_warps - 1) / n_warps, n_bl, n_freq);

  if (warp_size == 32) {
    rfi_jvp_kernel<block_size, 32, INT_T><<<grid, block, 0, stream>>>(
        a1_tensor, a2_tensor, rfi_amp_fine_tensor, rfi_amp_fine_grad_tensor,
        rfi_phase_tensor, rfi_phase_grad_tensor, rfi_grad_tensor);
  } else if (warp_size == 64) {
    rfi_jvp_kernel<block_size, 64, INT_T><<<grid, block, 0, stream>>>(
        a1_tensor, a2_tensor, rfi_amp_fine_tensor, rfi_amp_fine_grad_tensor,
        rfi_phase_tensor, rfi_phase_grad_tensor, rfi_grad_tensor);
  } else {
    return ffi::Error::Internal("Unsupported GPU warp size.");
  }

  const auto status = cudaGetLastError();
  if (status != cudaSuccess) {
    return ffi::Error::Internal(std::string("GPU kernel launch error: ") +
                                cudaGetErrorString(status));
  }

  return ffi::Error::Success();
}

ffi::Error calc_rfi_jvp_gpu_impl(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a1_sorter, ffi::BufferR1<ffi::S32> a1_start,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32> a2_sorter,
    ffi::BufferR1<ffi::S32> a2_start, rfi_amp_fine_t rfi_amp_fine,
    rfi_amp_fine_t rfi_amp_fine_grad, rfi_phase_t rfi_phase,
    rfi_phase_t rfi_phase_grad, ffi::ResultBufferR3<ffi::C128> rfi_grad) {
  constexpr std::int64_t max32 = std::numeric_limits<std::int32_t>::max();
  // use 32 bit indexing if possible
  if (a1.element_count() < max32 && a2.element_count() < max32 &&
      rfi_amp_fine.element_count() < max32 &&
      rfi_phase.element_count() < max32 && rfi_grad->element_count() < max32) {
    return calc_rfi_jvp_gpu_dispatch<std::int32_t>(stream, a1, a2, rfi_amp_fine,
                                                   rfi_amp_fine_grad, rfi_phase,
                                                   rfi_phase_grad, rfi_grad);
  } else {
    return calc_rfi_jvp_gpu_dispatch<std::int64_t>(stream, a1, a2, rfi_amp_fine,
                                                   rfi_amp_fine_grad, rfi_phase,
                                                   rfi_phase_grad, rfi_grad);
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_jvp_gpu, calc_rfi_jvp_gpu_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<rfi_amp_fine_t>()
                                  .Arg<rfi_amp_fine_t>()
                                  .Arg<rfi_phase_t>()
                                  .Arg<rfi_phase_t>()
                                  .Ret<ffi::BufferR3<ffi::C128>>());

} // namespace gpu
} // namespace tabascal
