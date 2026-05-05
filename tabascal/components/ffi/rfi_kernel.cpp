#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <iostream>
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

template <class D> struct ComplexV {
  hn::Vec<D> re, im;
};

template <class D>
HWY_INLINE HWY_ATTR ComplexV<D> Mul(ComplexV<D> a, ComplexV<D> b) {

  const auto re = hn::Sub(hn::Mul(a.re, b.re), hn::Mul(a.im, b.im));
  const auto im = hn::Add(hn::Mul(a.im, b.re), hn::Mul(a.re, b.im));

  return ComplexV<D>{re, im};
};

template <class D>
HWY_INLINE HWY_ATTR ComplexV<D> MulConj(ComplexV<D> a, ComplexV<D> b) {

  const auto re = hn::Add(hn::Mul(a.re, b.re), hn::Mul(a.im, b.im));
  const auto im = hn::Sub(hn::Mul(a.im, b.re), hn::Mul(a.re, b.im));

  return ComplexV<D>{re, im};
};

template <class D>
HWY_INLINE HWY_ATTR ComplexV<D> LoadU(D d,
                                      const std::complex<hn::TFromD<D>> *ptr) {

  using T = hn::TFromD<D>;

  constexpr std::int64_t n_lanes = hn::Lanes(d);

  auto scalar_ptr = reinterpret_cast<const T *>(ptr);

  const auto val_1 = hn::LoadU(d, scalar_ptr);
  const auto val_2 = hn::LoadU(d, scalar_ptr + n_lanes);

  const auto re = hn::ConcatEven(d, val_2, val_1);
  const auto im = hn::ConcatOdd(d, val_2, val_1);

  return ComplexV<D>{re, im};
};

HWY_ATTR void
rfi_kernel_opt(Tensor1D<const int *> a1, Tensor1D<const int *> a2,
               Tensor4D<const std::complex<double> *> rfi_amp_fine,
               Tensor4D<const double *> rfi_phase,
               Tensor3D<std::complex<double> *> rfi_vis) {

  using D = TagType<double>;

  const TagType<double> d;
  constexpr std::int64_t n_lanes = hn::Lanes(d);

  const auto n_rfi = rfi_amp_fine.shape[0];
  const auto n_ant = rfi_amp_fine.shape[1];
  const auto n_freq_fine = rfi_amp_fine.shape[2];
  const auto n_time_fine = rfi_amp_fine.shape[3];
  const auto n_bl = a1.shape[0];
  const auto n_time = rfi_vis.shape[2];
  const auto n_freq = rfi_vis.shape[1];

  assert(a1.shape[0] == a2.shape[0]);
  assert(a1.shape[0] == rfi_vis.shape[0]);
  assert(rfi_phase.shape[0] == rfi_amp_fine.shape[0]);
  assert(rfi_phase.shape[1] == rfi_amp_fine.shape[1]);
  assert(rfi_phase.shape[2] == rfi_amp_fine.shape[2]);
  assert(rfi_phase.shape[3] == rfi_amp_fine.shape[3]);

  const auto n_int_t = n_time_fine / n_time;
  const auto n_int_f = n_freq_fine / n_freq;
  const double n_int_inv = 1.f / double(n_int_t * n_int_f);

  for (std::int64_t i_bl = 0; i_bl < n_bl; ++i_bl) {
    std::int64_t i_a1 = a1(i_bl);
    std::int64_t i_a2 = a2(i_bl);

    for (std::int64_t i_f = 0; i_f < n_freq; ++i_f) {
      const auto i_f_fine_begin = i_f * n_int_f;
      for (std::int64_t i_t = 0; i_t < n_time; ++i_t) {
        std::complex<double> sum{0, 0};

        const auto i_t_fine_begin = i_t * n_int_t;

        for (std::int64_t i_rfi = 0; i_rfi < n_rfi; ++i_rfi) {

          for (std::int64_t i_f_fine = i_f_fine_begin;
               i_f_fine < i_f_fine_begin + n_int_f; ++i_f_fine) {

            auto ptr_val_rfi_amp_1 = &rfi_amp_fine(i_rfi, i_a1, i_f_fine, 0);
            auto ptr_val_rfi_amp_2 = &rfi_amp_fine(i_rfi, i_a2, i_f_fine, 0);

            auto ptr_val_rfi_phase_1 = &rfi_phase(i_rfi, i_a1, i_f_fine, 0);
            auto ptr_val_rfi_phase_2 = &rfi_phase(i_rfi, i_a2, i_f_fine, 0);

            std::int64_t i_t_fine = i_t_fine_begin;
            for (; i_t_fine + n_lanes <= i_t_fine_begin + n_int_t;
                 i_t_fine += n_lanes) {
              const auto val_rfi_phase_1 =
                  hn::LoadU(d, ptr_val_rfi_phase_1 + i_t_fine);
              const auto val_rfi_phase_2 =
                  hn::LoadU(d, ptr_val_rfi_phase_2 + i_t_fine);

              const auto val_rfi_amp_1 = LoadU(d, ptr_val_rfi_amp_1 + i_t_fine);

              const auto val_rfi_amp_2 = LoadU(d, ptr_val_rfi_amp_2 + i_t_fine);

              const auto phase_diff = hn::Sub(val_rfi_phase_1, val_rfi_phase_2);

              const auto c = hn::Cos(d, phase_diff);
              const auto s = hn::Sin(d, phase_diff);

              const auto e = ComplexV<D>{c, s};

              const auto res = Mul(MulConj(val_rfi_amp_1, val_rfi_amp_2), e);

              sum += std::complex<double>(hn::ReduceSum(d, res.re),
                                          hn::ReduceSum(d, res.im));
            }

            for (; i_t_fine < i_t_fine_begin + n_int_t; ++i_t_fine) {
              const auto val_rfi_amp_1 =
                  rfi_amp_fine(i_rfi, i_a1, i_f_fine, i_t_fine);
              const auto val_rfi_amp_2 =
                  rfi_amp_fine(i_rfi, i_a2, i_f_fine, i_t_fine);

              const auto val_rfi_phase_1 =
                  rfi_phase(i_rfi, i_a1, i_f_fine, i_t_fine);
              const auto val_rfi_phase_2 =
                  rfi_phase(i_rfi, i_a2, i_f_fine, i_t_fine);

              // ideal shape for memory access: (n_ant, n_f, n_t, n_f_int,
              // n_t_int, n_rfi) currently: (n_rfi, n_ant, n_f * n_f_int, n_t *
              // n_t_int)
              std::complex<double> e(
                  std::cos(val_rfi_phase_1 - val_rfi_phase_2),
                  std::sin(val_rfi_phase_1 - val_rfi_phase_2));

              auto res = val_rfi_amp_1 * std::conj(val_rfi_amp_2) * e;

              sum += res;
            }
          }
        }

        sum *= n_int_inv;

        rfi_vis(i_bl, i_f, i_t) = sum;
      }
    }
  }
}
} // namespace HWY_NAMESPACE

#if HWY_ONCE

void rfi_kernel(Tensor1D<const int *> a1, Tensor1D<const int *> a2,
                Tensor4D<const std::complex<double> *> rfi_amp_fine,
                Tensor4D<const double *> rfi_phase,
                Tensor3D<std::complex<double> *> rfi_vis) {

  const auto n_rfi = rfi_amp_fine.shape[0];
  const auto n_ant = rfi_amp_fine.shape[1];
  const auto n_freq_fine = rfi_amp_fine.shape[2];
  const auto n_time_fine = rfi_amp_fine.shape[3];
  const auto n_bl = a1.shape[0];
  const auto n_time = rfi_vis.shape[2];
  const auto n_freq = rfi_vis.shape[1];

  assert(a1.shape[0] == a2.shape[0]);
  assert(a1.shape[0] == rfi_vis.shape[0]);
  assert(rfi_phase.shape[0] == rfi_amp_fine.shape[0]);
  assert(rfi_phase.shape[1] == rfi_amp_fine.shape[1]);
  assert(rfi_phase.shape[2] == rfi_amp_fine.shape[2]);
  assert(rfi_phase.shape[3] == rfi_amp_fine.shape[3]);

  const auto n_int_t = n_time_fine / n_time;
  const auto n_int_f = n_freq_fine / n_freq;
  const double n_int_inv = 1.f / double(n_int_t * n_int_f);

  for (std::int64_t i_bl = 0; i_bl < n_bl; ++i_bl) {
    std::int64_t i_a1 = a1(i_bl);
    std::int64_t i_a2 = a2(i_bl);

    for (std::int64_t i_f = 0; i_f < n_freq; ++i_f) {
      const auto i_f_fine_begin = i_f * n_int_f;
      for (std::int64_t i_t = 0; i_t < n_time; ++i_t) {
        std::complex<double> sum{0, 0};

        const auto i_t_fine_begin = i_t * n_int_t;

        for (std::int64_t i_rfi = 0; i_rfi < n_rfi; ++i_rfi) {

          for (std::int64_t i_f_fine = i_f_fine_begin;
               i_f_fine < i_f_fine_begin + n_int_f; ++i_f_fine) {

            for (std::int64_t i_t_fine = i_t_fine_begin;
                 i_t_fine < i_t_fine_begin + n_int_t; ++i_t_fine) {

              const auto val_rfi_amp_1 =
                  rfi_amp_fine(i_rfi, i_a1, i_f_fine, i_t_fine);
              const auto val_rfi_amp_2 =
                  rfi_amp_fine(i_rfi, i_a2, i_f_fine, i_t_fine);

              const auto val_rfi_phase_1 =
                  rfi_phase(i_rfi, i_a1, i_f_fine, i_t_fine);
              const auto val_rfi_phase_2 =
                  rfi_phase(i_rfi, i_a2, i_f_fine, i_t_fine);

              // ideal shape for memory access: (n_ant, n_f, n_t, n_f_int,
              // n_t_int, n_rfi) currently: (n_rfi, n_ant, n_f * n_f_int, n_t *
              // n_t_int)
              std::complex<double> e(
                  std::cos(val_rfi_phase_1 - val_rfi_phase_2),
                  std::sin(val_rfi_phase_1 - val_rfi_phase_2));

              auto res = val_rfi_amp_1 * std::conj(val_rfi_amp_2) * e;

              sum += res;
            }
          }
        }

        sum *= n_int_inv;

        rfi_vis(i_bl, i_f, i_t) = sum;
      }
    }
  }
}

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

  if (rfi_vis->dimensions()[1] != rfi_amp_fine.dimensions()[2]) {
    return ffi::Error::InvalidArgument(
        "Expected rfi_vis and rfi_amp_fine to have the same number of "
        "frequencies");
  }

  if (rfi_vis->dimensions()[2] != rfi_amp_fine.dimensions()[4]) {
    return ffi::Error::InvalidArgument(
        "Expected rfi_vis and rfi_amp_fine to have the same number of times");
  }

  Tensor1D<const int *> a1_tensor(a1.typed_data(), a1.dimensions()[0]);
  Tensor1D<const int *> a2_tensor(a2.typed_data(), a2.dimensions()[0]);
  Tensor4D<const std::complex<double> *> rfi_amp_fine_tensor(
      rfi_amp_fine.typed_data(), rfi_amp_fine.dimensions()[0],
      rfi_amp_fine.dimensions()[1],
      rfi_amp_fine.dimensions()[2] * rfi_amp_fine.dimensions()[3],
      rfi_amp_fine.dimensions()[4] * rfi_amp_fine.dimensions()[5]);
  Tensor4D<const double *> rfi_phase_tensor(
      rfi_phase.typed_data(), rfi_phase.dimensions()[0],
      rfi_phase.dimensions()[1],
      rfi_phase.dimensions()[2] * rfi_phase.dimensions()[3],
      rfi_phase.dimensions()[4] * rfi_phase.dimensions()[5]);

  Tensor3D<std::complex<double> *> rfi_vis_tensor(
      rfi_vis->typed_data(), rfi_vis->dimensions()[0], rfi_vis->dimensions()[1],
      rfi_vis->dimensions()[2]);

  // ffi::Ffi::Bind().Ctx<ffi::ThreadPool>().To(
  //     [](ffi::ThreadPool thread_pool) -> ffi::Error {
  //       return ffi::Error::Success();
  //     });

  // {
  //   auto start = std::chrono::high_resolution_clock::now();
  //   TABASCAL_EXPORT_AND_DISPATCH_T(rfi_kernel_opt)
  //   (a1_tensor, a2_tensor, rfi_amp_fine_tensor, rfi_phase_tensor,
  //    rfi_vis_tensor);
  //   auto end = std::chrono::high_resolution_clock::now();
  //   auto duration =
  //       std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  //   std::cerr << "opt time: " << duration.count() << " ms" << std::endl;
  // }

  // {
  //   auto start = std::chrono::high_resolution_clock::now();
  //   rfi_kernel(a1_tensor, a2_tensor, rfi_amp_fine_tensor, rfi_phase_tensor,
  //              rfi_vis_tensor);
  //   auto end = std::chrono::high_resolution_clock::now();
  //   auto duration =
  //       std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  //   std::cerr << "ref time: " << duration.count() << " ms" << std::endl;
  // }

  const int64_t n_threads = std::max<int64_t>(thread_pool.num_threads(), 1);

  const int64_t n_bl = a1.dimensions()[0];
  const int64_t n_bl_per_thread = (n_bl + n_threads -1) / n_threads;

  std::latch done(n_threads);

  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t thread_id = 0; thread_id < n_threads; ++thread_id) {
    thread_pool.Schedule([&, thread_id]() {

      const int64_t i_bl_start = thread_id * n_bl_per_thread;
      const int64_t n_bl_this_thread =
          std::max(i_bl_start + n_bl_per_thread, n_bl) - i_bl_start;

      Tensor1D<const int *> a1_tensor_th(a1.typed_data() + i_bl_start,
                                         n_bl_this_thread);
      Tensor1D<const int *> a2_tensor_th(a2.typed_data() + i_bl_start,
                                         n_bl_this_thread);

      Tensor3D<std::complex<double> *> rfi_vis_tensor_th(
          &rfi_vis_tensor(i_bl_start, 0, 0), n_bl_this_thread,
          rfi_vis->dimensions()[1], rfi_vis->dimensions()[2]);

      TABASCAL_EXPORT_AND_DISPATCH_T(rfi_kernel_opt)
      (a1_tensor, a2_tensor, rfi_amp_fine_tensor, rfi_phase_tensor,
       rfi_vis_tensor);

      done.count_down();
    });
  }

  done.wait(); // blocks the caller thread via futex — no spin
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cerr << "opt threaded (" << n_threads << ") time: " << duration.count()
            << " ms" << std::endl;

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
