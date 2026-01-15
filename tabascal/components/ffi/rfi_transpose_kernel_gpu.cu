#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <complex>
#include <cassert>
#include <stdexcept>
#include <limits>
#include <cstring>
#include <unistd.h>
#include <cuda_runtime_api.h>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "tensor.hpp"

namespace ffi = xla::ffi;
namespace cg = cooperative_groups;

namespace tabascal {
namespace gpu {

template <int BLOCK_SIZE, typename INT_T>
__global__ void __launch_bounds__(BLOCK_SIZE) rfi_transpose_kernel(
    INT_T n_int_f, INT_T n_int_t, Tensor1D<const int *, INT_T> a1,
    Tensor1D<const int *, INT_T> a2,
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

        for (INT_T i_bl = threadIdx.x; i_bl < n_bl; i_bl += blockDim.x) {

          INT_T i_a1 = a1(i_bl);
          INT_T i_a2 = a2(i_bl);

          // only process if current antenna index is matched by a1 or a2
          if (i_a1 != i_ant && i_a2 != i_ant)
            continue;

          auto val_rfi_vis_grad = rfi_vis_grad(i_bl, i_f, i_t);
          val_rfi_vis_grad.x *= n_int_inv;
          val_rfi_vis_grad.y *= n_int_inv;

          const auto val_rfi_amp_1 = rfi_amp_fine(i_rfi, i_a1, i_tf_fine);
          const auto val_rfi_amp_2 = rfi_amp_fine(i_rfi, i_a2, i_tf_fine);

          const auto val_rfi_phase_1 = rfi_phase(i_rfi, i_a1, i_tf_fine);
          const auto val_rfi_phase_2 = rfi_phase(i_rfi, i_a2, i_tf_fine);

          cuDoubleComplex e_val;
          sincos(val_rfi_phase_1 - val_rfi_phase_2, &e_val.y, &e_val.x);

          if (i_a1 == i_ant) {
            const auto t1 =
                cuCmul(cuCmul(val_rfi_vis_grad, cuConj(val_rfi_amp_2)), e_val);

            rfi_amp_sum = cuCadd(t1, rfi_amp_sum);
          }

          if (i_a2 == i_ant) {
            const auto t2 =
                cuConj(cuCmul(cuCmul(val_rfi_vis_grad, val_rfi_amp_1), e_val));

            rfi_amp_sum = cuCadd(t2, rfi_amp_sum);
          }

          const auto f1 =
              (cuCmul(cuCmul(cuCmul(cuDoubleComplex{-e_val.y, e_val.x},
                                    val_rfi_vis_grad),
                             val_rfi_amp_1),
                      cuConj(val_rfi_amp_2)))
                  .x;

          if (i_a1 == i_ant) {
            rfi_phase_sum += f1;
          }

          if (i_a2 == i_ant) {
            rfi_phase_sum -= f1;
          }
        }

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
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> a2,
    rfi_amp_fine_t rfi_amp_fine, rfi_phase_t rfi_phase,
    ffi::BufferR3<ffi::C128> rfi_vis_grad,
    ffi::Result<rfi_amp_fine_t> rfi_amp_fine_grad,
    ffi::Result<rfi_phase_t> rfi_phase_grad) {
  // rfi_transpose_amp_fine and rfi_transpose_phase shape is
  // (n_rfi, n_ant, n_freq, n_int_freq, n_time, n_int_time)

  // if (a1.dimensions().size() != 1) {
  //   return ffi::Error::InvalidArgument("Expected 1d a1");
  // }

  Tensor1D<const int *, INT_T> a1_tensor(a1.typed_data(), a1.dimensions()[0]);
  Tensor1D<const int *, INT_T> a2_tensor(a2.typed_data(), a2.dimensions()[0]);
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

  // For 64 antenna, 32 yields best results
  // The kernel will read n_bl indices, but only compute for n_ant values
  // Therefore a possible good choice would be n_ant / 2
  constexpr int block_size = 32;

  dim3 block(block_size);
  dim3 grid(n_tf_fine, n_ant, n_rfi);

  rfi_transpose_kernel<block_size, INT_T><<<grid, block, 0, stream>>>(
      n_int_f, n_int_t, a1_tensor, a2_tensor, rfi_amp_fine_tensor,
      rfi_phase_tensor, rfi_grad_tensor, rfi_amp_fine_grad_tensor,
      rfi_phase_grad_tensor);

  const auto status = cudaGetLastError();
  if (status != cudaSuccess) {
    return ffi::Error::Internal(std::string("GPU kernel launch error: ") +
                                cudaGetErrorString(status));
  }

  return ffi::Error::Success();
}

ffi::Error
calc_rfi_transpose_gpu_impl(cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
                            ffi::BufferR1<ffi::S32> a2,
                            rfi_amp_fine_t rfi_amp_fine, rfi_phase_t rfi_phase,
                            ffi::BufferR3<ffi::C128> rfi_vis_grad,
                            ffi::Result<rfi_amp_fine_t> rfi_amp_fine_grad,
                            ffi::Result<rfi_phase_t> rfi_phase_grad) {
  constexpr std::int64_t max32 = std::numeric_limits<std::int32_t>::max();
  // use 32 bit indexing if possible
  if (a1.element_count() < max32 && a2.element_count() < max32 &&
      rfi_amp_fine.element_count() < max32 &&
      rfi_phase.element_count() < max32 &&
      rfi_vis_grad.element_count() < max32) {
    return calc_rfi_transpose_gpu_dispatch<std::int32_t>(
        stream, a1, a2, rfi_amp_fine, rfi_phase, rfi_vis_grad,
        rfi_amp_fine_grad, rfi_phase_grad);
  } else {
    return calc_rfi_transpose_gpu_dispatch<std::int64_t>(
        stream, a1, a2, rfi_amp_fine, rfi_phase, rfi_vis_grad,
        rfi_amp_fine_grad, rfi_phase_grad);
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_transpose_gpu,
                              calc_rfi_transpose_gpu_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<rfi_amp_fine_t>()
                                  .Arg<rfi_phase_t>()
                                  .Arg<ffi::BufferR3<ffi::C128>>()
                                  .Ret<rfi_amp_fine_t>()
                                  .Ret<rfi_phase_t>());
} // namespace gpu
} // namespace tabascal
