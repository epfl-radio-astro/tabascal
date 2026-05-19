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
    rfi_transpose_kernel(double n_int_inv,
                         Tensor1D<const int *, INT_T> a1,
                         Tensor1D<const int *, INT_T> a1_sorter,
                         Tensor1D<const int *, INT_T> a1_start,
                         Tensor1D<const int *, INT_T> a2,
                         Tensor1D<const int *, INT_T> a2_sorter,
                         Tensor1D<const int *, INT_T> a2_start,
                         Tensor4D<const cuDoubleComplex *, INT_T> rfi_amp_fine,
                         Tensor4D<const double *, INT_T> rfi_phase,
                         Tensor3D<const cuDoubleComplex *, INT_T> rfi_vis_grad,
                         Tensor4D<cuDoubleComplex *, INT_T> rfi_amp_fine_grad,
                         Tensor4D<double *, INT_T> rfi_phase_grad) {

  // rfi_amp_fine layout: (n_ant, n_freq, n_time, n_rfi * n_int_f * n_int_t)
  const auto n_ant = rfi_amp_fine.shape[0];
  const auto n_freq = rfi_amp_fine.shape[1];
  const auto n_time = rfi_amp_fine.shape[2];
  const auto n_red = rfi_amp_fine.shape[3];
  const auto n_bl = a1.shape[0];

  assert(a1.shape[0] == a2.shape[0]);
  assert(a1.shape[0] == rfi_vis_grad.shape[0]);
  assert(rfi_phase.shape[0] == rfi_amp_fine.shape[0]);
  assert(rfi_phase.shape[1] == rfi_amp_fine.shape[1]);
  assert(rfi_phase.shape[2] == rfi_amp_fine.shape[2]);
  assert(rfi_phase.shape[3] == rfi_amp_fine.shape[3]);
  assert(rfi_vis_grad.shape[1] == n_freq);
  assert(rfi_vis_grad.shape[2] == n_time);

  const INT_T n_ft = n_freq * n_time;

  // Grid: (n_freq * n_time, n_ant). Threads in a block split i_red, which
  // covers the collapsed (n_rfi, n_int_f, n_int_t) inner dimension. All warp
  // lanes access consecutive i_red values of the same antenna -> coalesced
  // loads on the hot path. No inter-thread reduction is needed because each
  // thread writes a distinct output element.
  for (INT_T i_ft = blockIdx.x; i_ft < n_ft; i_ft += gridDim.x) {
    const INT_T i_t = i_ft % n_time;
    const INT_T i_f = i_ft / n_time;

    for (INT_T i_ant = blockIdx.y; i_ant < n_ant; i_ant += gridDim.y) {

      const INT_T a1_begin = a1_start(i_ant);
      const INT_T a1_end = (i_ant == n_ant - 1) ? n_bl : a1_start(i_ant + 1);
      const INT_T a2_begin = a2_start(i_ant);
      const INT_T a2_end = (i_ant == n_ant - 1) ? n_bl : a2_start(i_ant + 1);

      for (INT_T i_red = threadIdx.x; i_red < n_red; i_red += BLOCK_SIZE) {

        const auto my_amp = rfi_amp_fine(i_ant, i_f, i_t, i_red);
        const auto my_phase = rfi_phase(i_ant, i_f, i_t, i_red);

        cuDoubleComplex amp_sum{0, 0};
        double phase_sum = 0;

        // a1 loop: i_ant is the "first" antenna of each baseline.
        for (INT_T i_bl_a1 = a1_begin; i_bl_a1 < a1_end; ++i_bl_a1) {
          const INT_T i_bl = a1_sorter(i_bl_a1);
          const INT_T i_a2 = a2(i_bl);

          const auto other_amp = rfi_amp_fine(i_a2, i_f, i_t, i_red);
          const auto other_phase = rfi_phase(i_a2, i_f, i_t, i_red);
          // Same address for every lane in the warp -> broadcast load.
          const auto vis_grad = rfi_vis_grad(i_bl, i_f, i_t);

          cuDoubleComplex e;
          sincos(my_phase - other_phase, &e.y, &e.x);

          const auto t1 = cuCmul(cuCmul(vis_grad, cuConj(other_amp)), e);
          amp_sum = cuCadd(amp_sum, t1);
          // f1 = Re(i * t1 * my_amp) = -t1.y*my_amp.x - t1.x*my_amp.y
          phase_sum += -t1.y * my_amp.x - t1.x * my_amp.y;
        }

        // a2 loop: i_ant is the "second" antenna of each baseline.
        for (INT_T i_bl_a2 = a2_begin; i_bl_a2 < a2_end; ++i_bl_a2) {
          const INT_T i_bl = a2_sorter(i_bl_a2);
          const INT_T i_a1 = a1(i_bl);

          const auto other_amp = rfi_amp_fine(i_a1, i_f, i_t, i_red);
          const auto other_phase = rfi_phase(i_a1, i_f, i_t, i_red);
          const auto vis_grad = rfi_vis_grad(i_bl, i_f, i_t);

          cuDoubleComplex e;
          sincos(other_phase - my_phase, &e.y, &e.x);

          const auto t2 = cuConj(cuCmul(cuCmul(vis_grad, other_amp), e));
          amp_sum = cuCadd(amp_sum, t2);
          // f1 = Im(t2 * my_amp) = t2.x*my_amp.y + t2.y*my_amp.x
          phase_sum -= t2.x * my_amp.y + t2.y * my_amp.x;
        }

        amp_sum.x *= n_int_inv;
        amp_sum.y *= n_int_inv;
        phase_sum *= n_int_inv;

        rfi_amp_fine_grad(i_ant, i_f, i_t, i_red) = amp_sum;
        rfi_phase_grad(i_ant, i_f, i_t, i_red) = phase_sum;
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

  if (rfi_vis_grad.dimensions()[1] != rfi_amp_fine.dimensions()[1]) {
    return ffi::Error::InvalidArgument(
        "Expected rfi_vis_grad and rfi_amp_fine to have the same number of "
        "frequencies");
  }

  if (rfi_vis_grad.dimensions()[2] != rfi_amp_fine.dimensions()[2]) {
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

  // rfi_amp_fine / rfi_phase (and their grads) layout:
  //   (n_ant, n_freq, n_time, n_rfi, n_int_freq, n_int_time)
  // Collapse the three innermost dims (n_rfi, n_int_freq, n_int_time) into
  // one for kernel indexing.
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

  Tensor4D<const cuDoubleComplex *, INT_T> rfi_amp_fine_tensor(
      (const cuDoubleComplex *)rfi_amp_fine.typed_data(),
      rfi_amp_fine.dimensions()[0], rfi_amp_fine.dimensions()[1],
      rfi_amp_fine.dimensions()[2],
      rfi_amp_fine.dimensions()[3] * rfi_amp_fine.dimensions()[4] *
          rfi_amp_fine.dimensions()[5]);
  Tensor4D<cuDoubleComplex *, INT_T> rfi_amp_fine_grad_tensor(
      (cuDoubleComplex *)rfi_amp_fine_grad->typed_data(),
      rfi_amp_fine_grad->dimensions()[0], rfi_amp_fine_grad->dimensions()[1],
      rfi_amp_fine_grad->dimensions()[2],
      rfi_amp_fine_grad->dimensions()[3] * rfi_amp_fine_grad->dimensions()[4] *
          rfi_amp_fine_grad->dimensions()[5]);
  Tensor4D<const double *, INT_T> rfi_phase_tensor(
      rfi_phase.typed_data(), rfi_phase.dimensions()[0],
      rfi_phase.dimensions()[1], rfi_phase.dimensions()[2],
      rfi_phase.dimensions()[3] * rfi_phase.dimensions()[4] *
          rfi_phase.dimensions()[5]);
  Tensor4D<double *, INT_T> rfi_phase_grad_tensor(
      rfi_phase_grad->typed_data(), rfi_phase_grad->dimensions()[0],
      rfi_phase_grad->dimensions()[1], rfi_phase_grad->dimensions()[2],
      rfi_phase_grad->dimensions()[3] * rfi_phase_grad->dimensions()[4] *
          rfi_phase_grad->dimensions()[5]);

  Tensor3D<const cuDoubleComplex *, INT_T> rfi_grad_tensor(
      (const cuDoubleComplex *)rfi_vis_grad.typed_data(),
      rfi_vis_grad.dimensions()[0], rfi_vis_grad.dimensions()[1],
      rfi_vis_grad.dimensions()[2]);

  const INT_T n_freq = (INT_T)rfi_amp_fine.dimensions()[1];
  const INT_T n_time = (INT_T)rfi_amp_fine.dimensions()[2];
  const INT_T n_int_f = (INT_T)rfi_amp_fine.dimensions()[4];
  const INT_T n_int_t = (INT_T)rfi_amp_fine.dimensions()[5];
  const INT_T n_ant = (INT_T)rfi_amp_fine.dimensions()[0];
  const INT_T n_red = rfi_amp_fine_tensor.shape[3];

  const double n_int_inv = 1.0 / double(n_int_t * n_int_f);

  // Grid: (n_freq * n_time, n_ant). Each block covers one (i_ant, i_f, i_t);
  // threads in the block split the collapsed (n_rfi, n_int_f, n_int_t) dim.
  auto grid = create_clamped_grid(n_freq * n_time, n_ant, 1);

  // BLOCK_SIZE chosen to match n_red (the per-block parallel dim). The ladder
  // avoids leaving most of a large block idle when n_red is small.
  auto launch = [&](auto block_size_c) {
    constexpr int block_size = decltype(block_size_c)::value;
    dim3 block(block_size);
    rfi_transpose_kernel<block_size, INT_T><<<grid, block, 0, stream>>>(
        n_int_inv, a1_tensor, a1_sorter_tensor, a1_start_tensor, a2_tensor,
        a2_sorter_tensor, a2_start_tensor, rfi_amp_fine_tensor,
        rfi_phase_tensor, rfi_grad_tensor, rfi_amp_fine_grad_tensor,
        rfi_phase_grad_tensor);
  };

  if (n_red <= 32) {
    launch(std::integral_constant<int, 32>{});
  } else if (n_red <= 64) {
    launch(std::integral_constant<int, 64>{});
  } else if (n_red <= 128) {
    launch(std::integral_constant<int, 128>{});
  } else {
    launch(std::integral_constant<int, 256>{});
  }

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
