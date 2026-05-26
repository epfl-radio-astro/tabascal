#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <latch>
#include <stdexcept>
#include <array>
#include <vector>
#include <unistd.h>

#include "tensor.hpp"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

// Generates code for every target that this compiler can support.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rfi_transpose_kernel.cpp" // this file

#include "hwy_dispatch.hpp"


namespace tabascal {

namespace ffi = xla::ffi;

namespace HWY_NAMESPACE { // required: unique per target

namespace hn = ::hwy::HWY_NAMESPACE;

#include "complex_vector_inl.hpp"

HWY_ATTR void
rfi_transpose_kernel_opt(
    double n_int_inv, std::int64_t i_ant_start, std::int64_t i_ant_end,
    Tensor1D<const int *> a1, Tensor1D<const int *> a1_sorter,
    Tensor1D<const int *> a1_start, Tensor1D<const int *> a2,
    Tensor1D<const int *> a2_sorter, Tensor1D<const int *> a2_start,
    Tensor4D<const std::complex<double> *> rfi_amp_fine,
    Tensor4D<const double *> rfi_phase,
    Tensor3D<const std::complex<double> *> rfi_vis_grad,
    Tensor4D<std::complex<double> *> rfi_amp_fine_grad,
    Tensor4D<double *> rfi_phase_grad) {

  using D = TagType<double>;

  const TagType<double> d;
  constexpr std::int64_t n_lanes = hn::Lanes(d);

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

  const auto inv_v = hn::Set(d, n_int_inv);

  for (std::int64_t i_ant = i_ant_start; i_ant < i_ant_end; ++i_ant) {

    const std::int64_t a1_begin = a1_start(i_ant);
    const std::int64_t a1_end =
        (i_ant == n_ant - 1) ? n_bl : a1_start(i_ant + 1);
    const std::int64_t a2_begin = a2_start(i_ant);
    const std::int64_t a2_end =
        (i_ant == n_ant - 1) ? n_bl : a2_start(i_ant + 1);

    for (std::int64_t i_f = 0; i_f < n_freq; ++i_f) {
      for (std::int64_t i_t = 0; i_t < n_time; ++i_t) {

        auto *p_my_amp_out =
            &rfi_amp_fine_grad(i_ant, i_f, i_t, 0);
        auto *p_my_phase_out = &rfi_phase_grad(i_ant, i_f, i_t, 0);
        const auto *p_my_amp_in = &rfi_amp_fine(i_ant, i_f, i_t, 0);
        const auto *p_my_phase_in = &rfi_phase(i_ant, i_f, i_t, 0);

        std::int64_t i_red = 0;
        for (; i_red + n_lanes <= n_red; i_red += n_lanes) {
          const auto my_amp = LoadU(d, p_my_amp_in + i_red);
          const auto my_phase = hn::LoadU(d, p_my_phase_in + i_red);

          ComplexV<D> amp_sum{hn::Zero(d), hn::Zero(d)};
          auto phase_sum = hn::Zero(d);

          // a1 loop: i_ant is the "first" antenna of the baseline.
          for (std::int64_t i_bl_a1 = a1_begin; i_bl_a1 < a1_end; ++i_bl_a1) {
            const std::int64_t i_bl = a1_sorter(i_bl_a1);
            const std::int64_t i_a2 = a2(i_bl);

            const auto other_amp =
                LoadU(d, &rfi_amp_fine(i_a2, i_f, i_t, i_red));
            const auto other_phase =
                hn::LoadU(d, &rfi_phase(i_a2, i_f, i_t, i_red));
            const auto vis_grad_scalar = rfi_vis_grad(i_bl, i_f, i_t);
            const auto vis_grad =
                ComplexV<D>{hn::Set(d, vis_grad_scalar.real()),
                            hn::Set(d, vis_grad_scalar.imag())};

            const auto phase_diff = hn::Sub(my_phase, other_phase);
            const auto e =
                ComplexV<D>{hn::Cos(d, phase_diff), hn::Sin(d, phase_diff)};

            // t1 = vis_grad * conj(other_amp) * e
            const auto t1 = Mul(MulConj(vis_grad, other_amp), e);
            amp_sum = Add(amp_sum, t1);
            // phase_sum += -t1.im * my_amp.re - t1.re * my_amp.im
            phase_sum = hn::NegMulAdd(t1.im, my_amp.re, phase_sum);
            phase_sum = hn::NegMulAdd(t1.re, my_amp.im, phase_sum);
          }

          // a2 loop: i_ant is the "second" antenna of the baseline.
          for (std::int64_t i_bl_a2 = a2_begin; i_bl_a2 < a2_end; ++i_bl_a2) {
            const std::int64_t i_bl = a2_sorter(i_bl_a2);
            const std::int64_t i_a1 = a1(i_bl);

            const auto other_amp =
                LoadU(d, &rfi_amp_fine(i_a1, i_f, i_t, i_red));
            const auto other_phase =
                hn::LoadU(d, &rfi_phase(i_a1, i_f, i_t, i_red));
            const auto vis_grad_scalar = rfi_vis_grad(i_bl, i_f, i_t);
            const auto vis_grad =
                ComplexV<D>{hn::Set(d, vis_grad_scalar.real()),
                            hn::Set(d, vis_grad_scalar.imag())};

            const auto phase_diff = hn::Sub(other_phase, my_phase);
            const auto e =
                ComplexV<D>{hn::Cos(d, phase_diff), hn::Sin(d, phase_diff)};

            // t2 = conj(vis_grad * other_amp * e)
            const auto t2_pre = Mul(Mul(vis_grad, other_amp), e);
            const auto t2 = ComplexV<D>{t2_pre.re, hn::Neg(t2_pre.im)};
            amp_sum = Add(amp_sum, t2);
            // phase_sum -= t2.re * my_amp.im + t2.im * my_amp.re
            phase_sum = hn::NegMulAdd(t2.re, my_amp.im, phase_sum);
            phase_sum = hn::NegMulAdd(t2.im, my_amp.re, phase_sum);
          }

          amp_sum.re = hn::Mul(amp_sum.re, inv_v);
          amp_sum.im = hn::Mul(amp_sum.im, inv_v);
          phase_sum = hn::Mul(phase_sum, inv_v);

          StoreU(d, amp_sum, p_my_amp_out + i_red);
          hn::StoreU(phase_sum, d, p_my_phase_out + i_red);
        }

        for (; i_red < n_red; ++i_red) {
          const auto my_amp = p_my_amp_in[i_red];
          const auto my_phase = p_my_phase_in[i_red];

          std::complex<double> amp_sum{0, 0};
          double phase_sum = 0;

          for (std::int64_t i_bl_a1 = a1_begin; i_bl_a1 < a1_end; ++i_bl_a1) {
            const std::int64_t i_bl = a1_sorter(i_bl_a1);
            const std::int64_t i_a2 = a2(i_bl);

            const auto other_amp = rfi_amp_fine(i_a2, i_f, i_t, i_red);
            const auto other_phase = rfi_phase(i_a2, i_f, i_t, i_red);
            const auto vis_grad = rfi_vis_grad(i_bl, i_f, i_t);

            const std::complex<double> e(std::cos(my_phase - other_phase),
                                         std::sin(my_phase - other_phase));

            const auto t1 = vis_grad * std::conj(other_amp) * e;
            amp_sum += t1;
            phase_sum +=
                -t1.imag() * my_amp.real() - t1.real() * my_amp.imag();
          }

          for (std::int64_t i_bl_a2 = a2_begin; i_bl_a2 < a2_end; ++i_bl_a2) {
            const std::int64_t i_bl = a2_sorter(i_bl_a2);
            const std::int64_t i_a1 = a1(i_bl);

            const auto other_amp = rfi_amp_fine(i_a1, i_f, i_t, i_red);
            const auto other_phase = rfi_phase(i_a1, i_f, i_t, i_red);
            const auto vis_grad = rfi_vis_grad(i_bl, i_f, i_t);

            const std::complex<double> e(std::cos(other_phase - my_phase),
                                         std::sin(other_phase - my_phase));

            const auto t2 = std::conj(vis_grad * other_amp * e);
            amp_sum += t2;
            phase_sum -=
                t2.real() * my_amp.imag() + t2.imag() * my_amp.real();
          }

          p_my_amp_out[i_red] = amp_sum * n_int_inv;
          p_my_phase_out[i_red] = phase_sum * n_int_inv;
        }
      }
    }
  }
}

} // namespace HWY_NAMESPACE

#if HWY_ONCE

using rfi_amp_fine_t = ffi::Buffer<ffi::C128, 6>;
using rfi_phase_t = ffi::Buffer<ffi::F64, 6>;

ffi::Error calc_rfi_transpose_cpu_impl(
    ffi::ThreadPool thread_pool,
    ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> a1_sorter,
    ffi::BufferR1<ffi::S32> a1_start, ffi::BufferR1<ffi::S32> a2,
    ffi::BufferR1<ffi::S32> a2_sorter, ffi::BufferR1<ffi::S32> a2_start,
    rfi_amp_fine_t rfi_amp_fine, rfi_phase_t rfi_phase,
    ffi::BufferR3<ffi::C128> rfi_vis_grad,
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
  // Collapse the three innermost contiguous dims (n_rfi, n_int_freq,
  // n_int_time) into one for kernel indexing.
  Tensor1D<const int *> a1_tensor(a1.typed_data(), a1.dimensions()[0]);
  Tensor1D<const int *> a1_sorter_tensor(a1_sorter.typed_data(),
                                         a1_sorter.dimensions()[0]);
  Tensor1D<const int *> a1_start_tensor(a1_start.typed_data(),
                                        a1_start.dimensions()[0]);
  Tensor1D<const int *> a2_tensor(a2.typed_data(), a2.dimensions()[0]);
  Tensor1D<const int *> a2_sorter_tensor(a2_sorter.typed_data(),
                                         a2_sorter.dimensions()[0]);
  Tensor1D<const int *> a2_start_tensor(a2_start.typed_data(),
                                        a2_start.dimensions()[0]);

  Tensor4D<const std::complex<double> *> rfi_amp_fine_tensor(
      rfi_amp_fine.typed_data(), rfi_amp_fine.dimensions()[0],
      rfi_amp_fine.dimensions()[1], rfi_amp_fine.dimensions()[2],
      rfi_amp_fine.dimensions()[3] * rfi_amp_fine.dimensions()[4] *
          rfi_amp_fine.dimensions()[5]);
  Tensor4D<std::complex<double> *> rfi_amp_fine_grad_tensor(
      rfi_amp_fine_grad->typed_data(), rfi_amp_fine_grad->dimensions()[0],
      rfi_amp_fine_grad->dimensions()[1], rfi_amp_fine_grad->dimensions()[2],
      rfi_amp_fine_grad->dimensions()[3] * rfi_amp_fine_grad->dimensions()[4] *
          rfi_amp_fine_grad->dimensions()[5]);
  Tensor4D<const double *> rfi_phase_tensor(
      rfi_phase.typed_data(), rfi_phase.dimensions()[0],
      rfi_phase.dimensions()[1], rfi_phase.dimensions()[2],
      rfi_phase.dimensions()[3] * rfi_phase.dimensions()[4] *
          rfi_phase.dimensions()[5]);
  Tensor4D<double *> rfi_phase_grad_tensor(
      rfi_phase_grad->typed_data(), rfi_phase_grad->dimensions()[0],
      rfi_phase_grad->dimensions()[1], rfi_phase_grad->dimensions()[2],
      rfi_phase_grad->dimensions()[3] * rfi_phase_grad->dimensions()[4] *
          rfi_phase_grad->dimensions()[5]);

  Tensor3D<const std::complex<double> *> rfi_grad_tensor(
      rfi_vis_grad.typed_data(), rfi_vis_grad.dimensions()[0],
      rfi_vis_grad.dimensions()[1], rfi_vis_grad.dimensions()[2]);

  const auto n_int_f = rfi_amp_fine.dimensions()[4];
  const auto n_int_t = rfi_amp_fine.dimensions()[5];
  const double n_int_inv = 1.0 / double(n_int_f * n_int_t);

  const int64_t n_threads = std::max<int64_t>(thread_pool.num_threads(), 1);

  const int64_t n_ant = rfi_amp_fine_tensor.shape[0];
  const int64_t n_ant_per_thread = (n_ant + n_threads - 1) / n_threads;

  std::latch done(n_threads);

  for (int64_t thread_id = 0; thread_id < n_threads; ++thread_id) {
    const int64_t i_ant_start = thread_id * n_ant_per_thread;
    if (i_ant_start >= n_ant) {
      done.count_down();
      continue;
    }
    thread_pool.Schedule([&, i_ant_start]() {
      const int64_t i_ant_end =
          std::min(i_ant_start + n_ant_per_thread, n_ant);

      TABASCAL_EXPORT_AND_DISPATCH_T(rfi_transpose_kernel_opt)
      (n_int_inv, i_ant_start, i_ant_end, a1_tensor, a1_sorter_tensor,
       a1_start_tensor, a2_tensor, a2_sorter_tensor, a2_start_tensor,
       rfi_amp_fine_tensor, rfi_phase_tensor, rfi_grad_tensor,
       rfi_amp_fine_grad_tensor, rfi_phase_grad_tensor);

      done.count_down();
    });
  }

  done.wait();

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_transpose_cpu,
                              calc_rfi_transpose_cpu_impl,
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
                                  .Arg<ffi::BufferR3<ffi::C128>>()
                                  .Ret<rfi_amp_fine_t>()
                                  .Ret<rfi_phase_t>());

#endif

} // namespace tabascal
