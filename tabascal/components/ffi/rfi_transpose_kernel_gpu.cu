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

template <int BLOCK_SIZE, typename INT_T>
__global__ void __launch_bounds__(BLOCK_SIZE)
    rfi_transpose_kernel(INT_T n_int_f, INT_T n_int_t,
                         Tensor1D<const int *, INT_T> a1,
                         Tensor1D<const int *, INT_T> a1_sorter,
                         Tensor1D<const int *, INT_T> a1_start,
                         Tensor1D<const int *, INT_T> a2,
                         Tensor1D<const int *, INT_T> a2_sorter,
                         Tensor1D<const int *, INT_T> a2_start,
                         Tensor3D<const cuDoubleComplex *, INT_T> rfi_amp_fine,
                         Tensor3D<const double *, INT_T> rfi_phase,
                         Tensor3D<const cuDoubleComplex *, INT_T> rfi_vis_grad,
                         Tensor3D<cuDoubleComplex *, INT_T> rfi_amp_fine_grad,
                         Tensor3D<double *, INT_T> rfi_phase_grad) {

  // Specialize BlockReduce type for our thread block
  using BlockReduce_t =
      cub::BlockReduce<double, BLOCK_SIZE, cub::BLOCK_REDUCE_WARP_REDUCTIONS>;

  // Shared memory
  __shared__ typename BlockReduce_t::TempStorage temp_storage[3];

  const auto n_rfi = rfi_amp_fine.shape[0];
  const auto n_ant = rfi_amp_fine.shape[1];
  const auto n_bl = a1.shape[0];
  const auto n_time = rfi_vis_grad.shape[2];
  const auto n_freq = rfi_vis_grad.shape[1];

  assert(a1.shape[0] == a2.shape[0]);
  assert(a1.shape[0] == rfi_vis_grad.shape[0]);
  assert(rfi_phase.shape[0] == rfi_amp_fine.shape[0]);
  assert(rfi_phase.shape[1] == rfi_amp_fine.shape[1]);
  assert(rfi_phase.shape[2] == rfi_amp_fine.shape[2]);

  const double n_int_inv = 1.f / double(n_int_t * n_int_f);

  for (INT_T i_tf_fine = blockIdx.x; i_tf_fine < rfi_amp_fine.shape[2];
       i_tf_fine += gridDim.x) {

    const auto i_t = (i_tf_fine % (n_time * n_int_t)) / n_int_t;
    const auto i_f = (i_tf_fine / (n_time * n_int_t)) / n_int_f;

    assert(i_t < rfi_vis_grad.shape[2]);
    assert(i_f < rfi_vis_grad.shape[1]);

    for (INT_T i_rfi = blockIdx.z; i_rfi < n_rfi; i_rfi += gridDim.z) {
      for (INT_T i_ant = blockIdx.y; i_ant < n_ant; i_ant += gridDim.y) {

        cuDoubleComplex rfi_amp_sum{0, 0};
        double rfi_phase_sum = 0;

        const auto my_val_rfi_amp = rfi_amp_fine(i_rfi, i_ant, i_tf_fine);
        const auto my_val_rfi_phase = rfi_phase(i_rfi, i_ant, i_tf_fine);

        const INT_T a1_begin = a1_start(i_ant);
        const INT_T a1_end = (i_ant == n_ant - 1) ? n_bl : a1_start(i_ant + 1);

        for (INT_T i_bl_a1 = a1_begin + threadIdx.x; i_bl_a1 < a1_end;
             i_bl_a1 += BLOCK_SIZE) {
          const INT_T i_bl = a1_sorter(i_bl_a1);
          const INT_T i_a2 = a2(i_bl);

          const auto &val_rfi_amp_1 = my_val_rfi_amp;
          const auto val_rfi_amp_2 = rfi_amp_fine(i_rfi, i_a2, i_tf_fine);

          const auto &val_rfi_phase_1 = my_val_rfi_phase;
          const auto val_rfi_phase_2 = rfi_phase(i_rfi, i_a2, i_tf_fine);

          auto val_rfi_vis_grad = rfi_vis_grad(i_bl, i_f, i_t);
          val_rfi_vis_grad.x *= n_int_inv;
          val_rfi_vis_grad.y *= n_int_inv;

          cuDoubleComplex e_val;
          sincos(val_rfi_phase_1 - val_rfi_phase_2, &e_val.y, &e_val.x);

          const auto t1 =
              cuCmul(cuCmul(val_rfi_vis_grad, cuConj(val_rfi_amp_2)), e_val);

          rfi_amp_sum = cuCadd(t1, rfi_amp_sum);

          const auto f1 =
              (cuCmul(cuCmul(cuCmul(cuDoubleComplex{-e_val.y, e_val.x},
                                    val_rfi_vis_grad),
                             val_rfi_amp_1),
                      cuConj(val_rfi_amp_2)))
                  .x;

          rfi_phase_sum += f1;
        }

        const INT_T a2_begin = a2_start(i_ant);
        const INT_T a2_end = (i_ant == n_ant - 1) ? n_bl : a2_start(i_ant + 1);

        for (INT_T i_bl_a2 = a2_begin + threadIdx.x; i_bl_a2 < a2_end;
             i_bl_a2 += BLOCK_SIZE) {
          const INT_T i_bl = a2_sorter(i_bl_a2);
          const INT_T i_a1 = a1(i_bl);

          const auto val_rfi_amp_1 = rfi_amp_fine(i_rfi, i_a1, i_tf_fine);
          const auto &val_rfi_amp_2 = my_val_rfi_amp;

          const auto val_rfi_phase_1 = rfi_phase(i_rfi, i_a1, i_tf_fine);
          const auto &val_rfi_phase_2 = my_val_rfi_phase;

          auto val_rfi_vis_grad = rfi_vis_grad(i_bl, i_f, i_t);
          val_rfi_vis_grad.x *= n_int_inv;
          val_rfi_vis_grad.y *= n_int_inv;

          cuDoubleComplex e_val;
          sincos(val_rfi_phase_1 - val_rfi_phase_2, &e_val.y, &e_val.x);

          const auto t2 =
              cuConj(cuCmul(cuCmul(val_rfi_vis_grad, val_rfi_amp_1), e_val));

          rfi_amp_sum = cuCadd(t2, rfi_amp_sum);

          const auto f1 =
              (cuCmul(cuCmul(cuCmul(cuDoubleComplex{-e_val.y, e_val.x},
                                    val_rfi_vis_grad),
                             val_rfi_amp_1),
                      cuConj(val_rfi_amp_2)))
                  .x;

          rfi_phase_sum -= f1;
        }

        // for (INT_T i_bl_s = threadIdx.x; i_bl_s < n_bl; i_bl_s += blockDim.x)
        // {
        //   const auto i_bl = a1_sorter(i_bl_s);

        //   INT_T i_a1 = a1(i_bl);
        //   INT_T i_a2 = a2(i_bl);

        //   // only process if current antenna index is matched by a1 or a2
        //   if (i_a1 != i_ant && i_a2 != i_ant)
        //     continue;

        //   auto val_rfi_vis_grad = rfi_vis_grad(i_bl, i_f, i_t);
        //   val_rfi_vis_grad.x *= n_int_inv;
        //   val_rfi_vis_grad.y *= n_int_inv;

        //   const auto val_rfi_amp_1 = rfi_amp_fine(i_rfi, i_a1, i_tf_fine);
        //   const auto val_rfi_amp_2 = rfi_amp_fine(i_rfi, i_a2, i_tf_fine);

        //   const auto val_rfi_phase_1 = rfi_phase(i_rfi, i_a1, i_tf_fine);
        //   const auto val_rfi_phase_2 = rfi_phase(i_rfi, i_a2, i_tf_fine);

        //   cuDoubleComplex e_val;
        //   sincos(val_rfi_phase_1 - val_rfi_phase_2, &e_val.y, &e_val.x);

        //   if (i_a1 == i_ant) {
        //     const auto t1 =
        //         cuCmul(cuCmul(val_rfi_vis_grad, cuConj(val_rfi_amp_2)),
        //         e_val);

        //     rfi_amp_sum = cuCadd(t1, rfi_amp_sum);
        //   }

        //   if (i_a2 == i_ant) {
        //     const auto t2 =
        //         cuConj(cuCmul(cuCmul(val_rfi_vis_grad, val_rfi_amp_1),
        //         e_val));

        //     rfi_amp_sum = cuCadd(t2, rfi_amp_sum);
        //   }

        //   const auto f1 =
        //       (cuCmul(cuCmul(cuCmul(cuDoubleComplex{-e_val.y, e_val.x},
        //                             val_rfi_vis_grad),
        //                      val_rfi_amp_1),
        //               cuConj(val_rfi_amp_2)))
        //           .x;

        //   if (i_a1 == i_ant) {
        //     rfi_phase_sum += f1;
        //   }

        //   if (i_a2 == i_ant) {
        //     rfi_phase_sum -= f1;
        //   }
        // }

        rfi_amp_sum.x = BlockReduce_t(temp_storage[0]).Sum(rfi_amp_sum.x);
        rfi_amp_sum.y = BlockReduce_t(temp_storage[1]).Sum(rfi_amp_sum.y);
        rfi_phase_sum = BlockReduce_t(temp_storage[2]).Sum(rfi_phase_sum);
        __syncthreads();

        if (threadIdx.x == 0) {
          rfi_amp_fine_grad(i_rfi, i_ant, i_tf_fine) = rfi_amp_sum;
          rfi_phase_grad(i_rfi, i_ant, i_tf_fine) = rfi_phase_sum;
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
ffi::Error calc_rfi_transpose_gpu_dispatch(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a1_sorter, ffi::BufferR1<ffi::S32> a1_start,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32> a2_sorter,
    ffi::BufferR1<ffi::S32> a2_start, rfi_amp_fine_t rfi_amp_fine,
    rfi_phase_t rfi_phase, ffi::BufferR3<ffi::C128> rfi_vis_grad,
    ffi::Result<rfi_amp_fine_t> rfi_amp_fine_grad,
    ffi::Result<rfi_phase_t> rfi_phase_grad) {

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

  if (rfi_vis_grad.dimensions()[0] != a1.dimensions()[0]) {
    return ffi::Error::InvalidArgument(
        "Expected rfi_vis_grad and a1 to have the same number of baselines");
  }

  if (rfi_vis_grad.dimensions()[1] != rfi_amp_fine.dimensions()[2]) {
    return ffi::Error::InvalidArgument(
        "Expected rfi_vis_grad and rfi_amp_fine to have the same number of "
        "frequencies");
  }

  if (rfi_vis_grad.dimensions()[2] != rfi_amp_fine.dimensions()[4]) {
    return ffi::Error::InvalidArgument("Expected rfi_vis_grad and rfi_amp_fine "
                                       "to have the same number of times");
  }

  for (int i = 0; i < 6; ++i) {
    if (rfi_amp_fine.dimensions()[i] != rfi_amp_fine_grad->dimensions()[i]) {
      return ffi::Error::InvalidArgument(
          "Expected rfi_amp_fine and rfi_amp_fine_grad to have the same shape");
    }
    if (rfi_phase.dimensions()[i] != rfi_phase_grad->dimensions()[i]) {
      return ffi::Error::InvalidArgument(
          "Expected rfi_phase and rfi_phase_grad to have the same shape");
    }
  }

  Tensor1D<const int *, INT_T> a1_tensor(a1.typed_data(), a1.dimensions()[0]);
  Tensor1D<const int *, INT_T> a1_sorter_tensor(a1_sorter.typed_data(),
                                                a1_sorter.dimensions()[0]);
  Tensor1D<const int *, INT_T> a1_start_tensor(a1_start.typed_data(),
                                               a1_start.dimensions()[0]);
  Tensor1D<const int *, INT_T> a2_tensor(a2.typed_data(), a2.dimensions()[0]);
  Tensor1D<const int *, INT_T> a2_sorter_tensor(a2_sorter.typed_data(),
                                                a2_sorter.dimensions()[0]);
  Tensor1D<const int *, INT_T> a2_start_tensor(a2_start.typed_data(),
                                               a2_start.dimensions()[0]);

  Tensor3D<const cuDoubleComplex *, INT_T> rfi_amp_fine_tensor(
      (const cuDoubleComplex *)rfi_amp_fine.typed_data(),
      rfi_amp_fine.dimensions()[0], rfi_amp_fine.dimensions()[1],
      rfi_amp_fine.dimensions()[2] * rfi_amp_fine.dimensions()[3] *
          rfi_amp_fine.dimensions()[4] * rfi_amp_fine.dimensions()[5]);
  Tensor3D<cuDoubleComplex *, INT_T> rfi_amp_fine_grad_tensor(
      (cuDoubleComplex *)rfi_amp_fine_grad->typed_data(),
      rfi_amp_fine_grad->dimensions()[0], rfi_amp_fine_grad->dimensions()[1],
      rfi_amp_fine_grad->dimensions()[2] * rfi_amp_fine_grad->dimensions()[3] *
          rfi_amp_fine_grad->dimensions()[4] *
          rfi_amp_fine_grad->dimensions()[5]);
  Tensor3D<const double *, INT_T> rfi_phase_tensor(
      rfi_phase.typed_data(), rfi_phase.dimensions()[0],
      rfi_phase.dimensions()[1],
      rfi_phase.dimensions()[2] * rfi_phase.dimensions()[3] *
          rfi_phase.dimensions()[4] * rfi_phase.dimensions()[5]);
  Tensor3D<double *, INT_T> rfi_phase_grad_tensor(
      rfi_phase_grad->typed_data(), rfi_phase_grad->dimensions()[0],
      rfi_phase_grad->dimensions()[1],
      rfi_phase_grad->dimensions()[2] * rfi_phase_grad->dimensions()[3] *
          rfi_phase_grad->dimensions()[4] * rfi_phase_grad->dimensions()[5]);

  Tensor3D<const cuDoubleComplex *, INT_T> rfi_grad_tensor(
      (const cuDoubleComplex *)rfi_vis_grad.typed_data(),
      rfi_vis_grad.dimensions()[0], rfi_vis_grad.dimensions()[1],
      rfi_vis_grad.dimensions()[2]);

  const auto n_rfi = rfi_amp_fine_tensor.shape[0];
  const auto n_ant = rfi_amp_fine_tensor.shape[1];
  const auto n_tf_fine = rfi_amp_fine_tensor.shape[2];
  const auto n_int_t = rfi_amp_fine.dimensions()[5];
  const auto n_int_f = rfi_amp_fine.dimensions()[3];

  // Cooperative group size. Must be power of 2. Used to iterate over n_int and
  // reduce result. If 32, equal to warp size on Nvidia for fast reduce
  // operation.

  // For 64 antenna, 32 yields best results
  // The kernel will read n_bl indices, but only compute for n_ant values
  // Therefore a possible good choice would be n_ant / 2
  constexpr int block_size = 32;

  dim3 block(block_size);

  auto grid = create_clamped_grid(n_tf_fine, n_ant, n_rfi);

  rfi_transpose_kernel<block_size, INT_T><<<grid, block, 0, stream>>>(
      n_int_f, n_int_t, a1_tensor, a1_sorter_tensor, a1_start_tensor, a2_tensor,
      a2_sorter_tensor, a2_start_tensor, rfi_amp_fine_tensor, rfi_phase_tensor,
      rfi_grad_tensor, rfi_amp_fine_grad_tensor, rfi_phase_grad_tensor);

  const auto status = cudaGetLastError();
  if (status != cudaSuccess) {
    return ffi::Error::Internal(std::string("GPU kernel launch error: ") +
                                cudaGetErrorString(status));
  }

  return ffi::Error::Success();
}

ffi::Error calc_rfi_transpose_gpu_impl(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a1_sorter, ffi::BufferR1<ffi::S32> a1_start,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32> a2_sorter,
    ffi::BufferR1<ffi::S32> a2_start, rfi_amp_fine_t rfi_amp_fine,
    rfi_phase_t rfi_phase, ffi::BufferR3<ffi::C128> rfi_vis_grad,
    ffi::Result<rfi_amp_fine_t> rfi_amp_fine_grad,
    ffi::Result<rfi_phase_t> rfi_phase_grad) {
  constexpr std::int64_t max32 = std::numeric_limits<std::int32_t>::max();
  // use 32 bit indexing if possible
  if (a1.element_count() < max32 && a2.element_count() < max32 &&
      rfi_amp_fine.element_count() < max32 &&
      rfi_phase.element_count() < max32 &&
      rfi_vis_grad.element_count() < max32) {
    return calc_rfi_transpose_gpu_dispatch<std::int32_t>(
        stream, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine,
        rfi_phase, rfi_vis_grad, rfi_amp_fine_grad, rfi_phase_grad);
  } else {
    return calc_rfi_transpose_gpu_dispatch<std::int64_t>(
        stream, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine,
        rfi_phase, rfi_vis_grad, rfi_amp_fine_grad, rfi_phase_grad);
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_transpose_gpu,
                              calc_rfi_transpose_gpu_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<rfi_amp_fine_t>()
                                  .Arg<rfi_phase_t>()
                                  .Arg<ffi::BufferR3<ffi::C128>>()
                                  .Ret<rfi_amp_fine_t>()
                                  .Ret<rfi_phase_t>());
} // namespace gpu
} // namespace tabascal
