#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <latch>
#include <unistd.h>

#include "tensor.hpp"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

// Generates code for every target that this compiler can support.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rfi_kernel.cpp" // this file

#include "hwy_dispatch.hpp"

namespace tabascal {

namespace ffi = xla::ffi;

namespace HWY_NAMESPACE { // required: unique per target

namespace hn = ::hwy::HWY_NAMESPACE;

#include "complex_vector_inl.hpp"

HWY_ATTR void
rfi_kernel_opt(double n_int_inv,
               Tensor1D<const int *> a1, Tensor1D<const int *> a2,
               Tensor4D<const std::complex<double> *> rfi_amp_fine,
               Tensor4D<const double *> rfi_phase,
               Tensor3D<std::complex<double> *> rfi_vis) {

  using D = TagType<double>;

  const TagType<double> d;
  constexpr std::int64_t n_lanes = hn::Lanes(d);

  // rfi_amp_fine layout: (n_ant, n_freq, n_time, n_rfi * n_int_f * n_int_t)
  const auto n_freq = rfi_amp_fine.shape[1];
  const auto n_time = rfi_amp_fine.shape[2];
  const auto n_red = rfi_amp_fine.shape[3];
  const auto n_bl = a1.shape[0];

  assert(a1.shape[0] == a2.shape[0]);
  assert(a1.shape[0] == rfi_vis.shape[0]);
  assert(rfi_phase.shape[0] == rfi_amp_fine.shape[0]);
  assert(rfi_phase.shape[1] == rfi_amp_fine.shape[1]);
  assert(rfi_phase.shape[2] == rfi_amp_fine.shape[2]);
  assert(rfi_phase.shape[3] == rfi_amp_fine.shape[3]);
  assert(rfi_vis.shape[1] == n_freq);
  assert(rfi_vis.shape[2] == n_time);

  for (std::int64_t i_bl = 0; i_bl < n_bl; ++i_bl) {
    std::int64_t i_a1 = a1(i_bl);
    std::int64_t i_a2 = a2(i_bl);

    for (std::int64_t i_f = 0; i_f < n_freq; ++i_f) {
      for (std::int64_t i_t = 0; i_t < n_time; ++i_t) {
        std::complex<double> sum{0, 0};

        const auto ptr_val_rfi_amp_1 = &rfi_amp_fine(i_a1, i_f, i_t, 0);
        const auto ptr_val_rfi_amp_2 = &rfi_amp_fine(i_a2, i_f, i_t, 0);

        const auto ptr_val_rfi_phase_1 = &rfi_phase(i_a1, i_f, i_t, 0);
        const auto ptr_val_rfi_phase_2 = &rfi_phase(i_a2, i_f, i_t, 0);

        std::int64_t i_red = 0;
        for (; i_red + n_lanes <= n_red; i_red += n_lanes) {
          const auto val_rfi_phase_1 =
              hn::LoadU(d, ptr_val_rfi_phase_1 + i_red);
          const auto val_rfi_phase_2 =
              hn::LoadU(d, ptr_val_rfi_phase_2 + i_red);

          const auto val_rfi_amp_1 = LoadU(d, ptr_val_rfi_amp_1 + i_red);
          const auto val_rfi_amp_2 = LoadU(d, ptr_val_rfi_amp_2 + i_red);

          const auto phase_diff = hn::Sub(val_rfi_phase_1, val_rfi_phase_2);

          const auto c = hn::Cos(d, phase_diff);
          const auto s = hn::Sin(d, phase_diff);

          const auto e = ComplexV<D>{c, s};

          const auto res = Mul(MulConj(val_rfi_amp_1, val_rfi_amp_2), e);

          sum += std::complex<double>(hn::ReduceSum(d, res.re),
                                      hn::ReduceSum(d, res.im));
        }

        for (; i_red < n_red; ++i_red) {
          const auto val_rfi_amp_1 = ptr_val_rfi_amp_1[i_red];
          const auto val_rfi_amp_2 = ptr_val_rfi_amp_2[i_red];

          const auto val_rfi_phase_1 = ptr_val_rfi_phase_1[i_red];
          const auto val_rfi_phase_2 = ptr_val_rfi_phase_2[i_red];

          std::complex<double> e(
              std::cos(val_rfi_phase_1 - val_rfi_phase_2),
              std::sin(val_rfi_phase_1 - val_rfi_phase_2));

          sum += val_rfi_amp_1 * std::conj(val_rfi_amp_2) * e;
        }

        sum *= n_int_inv;

        rfi_vis(i_bl, i_f, i_t) = sum;
      }
    }
  }
}
} // namespace HWY_NAMESPACE

#if HWY_ONCE

using rfi_amp_fine_t = ffi::Buffer<ffi::C128, 6>;
using rfi_phase_t = ffi::Buffer<ffi::F64, 6>;

ffi::Error calc_rfi_vis_cpu_impl(
    ffi::ThreadPool thread_pool,
    ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> a1_sorter,
    ffi::BufferR1<ffi::S32> a1_start, ffi::BufferR1<ffi::S32> a2,
    ffi::BufferR1<ffi::S32> a2_sorter, ffi::BufferR1<ffi::S32> a2_start,
    rfi_amp_fine_t rfi_amp_fine, rfi_phase_t rfi_phase,
    ffi::ResultBufferR3<ffi::C128> rfi_vis) {
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

  if (rfi_vis->dimensions()[0] != a1.dimensions()[0]) {
    return ffi::Error::InvalidArgument(
        "Expected rfi_vis and a1 to have the same number of baselines");
  }

  if (rfi_vis->dimensions()[1] != rfi_amp_fine.dimensions()[1]) {
    return ffi::Error::InvalidArgument(
        "Expected rfi_vis and rfi_amp_fine to have the same number of "
        "frequencies");
  }

  if (rfi_vis->dimensions()[2] != rfi_amp_fine.dimensions()[2]) {
    return ffi::Error::InvalidArgument(
        "Expected rfi_vis and rfi_amp_fine to have the same number of times");
  }

  // rfi_amp_fine / rfi_phase layout:
  //   (n_ant, n_freq, n_time, n_rfi, n_int_freq, n_int_time)
  // Collapse the three innermost contiguous dims (n_rfi, n_int_freq,
  // n_int_time) into one for kernel indexing.
  Tensor1D<const int *> a1_tensor(a1.typed_data(), a1.dimensions()[0]);
  Tensor1D<const int *> a2_tensor(a2.typed_data(), a2.dimensions()[0]);
  Tensor4D<const std::complex<double> *> rfi_amp_fine_tensor(
      rfi_amp_fine.typed_data(), rfi_amp_fine.dimensions()[0],
      rfi_amp_fine.dimensions()[1], rfi_amp_fine.dimensions()[2],
      rfi_amp_fine.dimensions()[3] * rfi_amp_fine.dimensions()[4] *
          rfi_amp_fine.dimensions()[5]);
  Tensor4D<const double *> rfi_phase_tensor(
      rfi_phase.typed_data(), rfi_phase.dimensions()[0],
      rfi_phase.dimensions()[1], rfi_phase.dimensions()[2],
      rfi_phase.dimensions()[3] * rfi_phase.dimensions()[4] *
          rfi_phase.dimensions()[5]);

  Tensor3D<std::complex<double> *> rfi_vis_tensor(
      rfi_vis->typed_data(), rfi_vis->dimensions()[0], rfi_vis->dimensions()[1],
      rfi_vis->dimensions()[2]);

  const auto n_int_f = rfi_amp_fine.dimensions()[4];
  const auto n_int_t = rfi_amp_fine.dimensions()[5];
  const double n_int_inv = 1.0 / double(n_int_f * n_int_t);

  const int64_t n_threads = std::max<int64_t>(thread_pool.num_threads(), 1);

  const int64_t n_bl = a1.dimensions()[0];
  const int64_t n_bl_per_thread = (n_bl + n_threads -1) / n_threads;

  std::latch done(n_threads);

  for (int64_t thread_id = 0; thread_id < n_threads; ++thread_id) {
    const int64_t i_bl_start = thread_id * n_bl_per_thread;
    if (i_bl_start >= n_bl) {
      done.count_down();
      continue;
    }
    thread_pool.Schedule([&, thread_id, i_bl_start]() {
      const int64_t n_bl_this_thread =
          std::min(i_bl_start + n_bl_per_thread, n_bl) - i_bl_start;

      Tensor1D<const int *> a1_tensor_th(a1.typed_data() + i_bl_start,
                                         n_bl_this_thread);
      Tensor1D<const int *> a2_tensor_th(a2.typed_data() + i_bl_start,
                                         n_bl_this_thread);

      Tensor3D<std::complex<double> *> rfi_vis_tensor_th(
          &rfi_vis_tensor(i_bl_start, 0, 0), n_bl_this_thread,
          rfi_vis->dimensions()[1], rfi_vis->dimensions()[2]);

      TABASCAL_EXPORT_AND_DISPATCH_T(rfi_kernel_opt)
      (n_int_inv, a1_tensor_th, a2_tensor_th, rfi_amp_fine_tensor,
       rfi_phase_tensor, rfi_vis_tensor_th);

      done.count_down();
    });
  }

  done.wait(); // blocks the caller thread via futex — no spin

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_vis_cpu, calc_rfi_vis_cpu_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::ThreadPool>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<rfi_amp_fine_t>()
                                  .Arg<rfi_phase_t>()
                                  .Ret<ffi::BufferR3<ffi::C128>>());

#endif

} // namespace tabascal
