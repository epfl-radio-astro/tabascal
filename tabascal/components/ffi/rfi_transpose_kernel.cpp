#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <array>
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
rfi_transpose_kernel_opt(std::int64_t n_int_f, std::int64_t n_int_t,
                         Tensor1D<const int *> a1, Tensor1D<const int *> a2,
                         Tensor3D<const std::complex<double> *> rfi_amp_fine,
                         Tensor3D<const double *> rfi_phase,
                         Tensor3D<const std::complex<double> *> rfi_vis_grad,
                         Tensor3D<std::complex<double> *> rfi_amp_fine_grad,
                         Tensor3D<double *> rfi_phase_grad) {

  using D = TagType<double>;

  const TagType<double> d;
  constexpr std::int64_t n_lanes = hn::Lanes(d);

  // set to 0 before accumulation
  const auto out_size = rfi_amp_fine_grad.shape[0] *
                        rfi_amp_fine_grad.shape[1] * rfi_amp_fine_grad.shape[2];

  std::memset(rfi_amp_fine_grad.ptr, 0,
              sizeof(std::complex<double>) * out_size);
  std::memset(rfi_phase_grad.ptr, 0, sizeof(double) * out_size);

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

  const ComplexV<D> i_unit{hn::Zero(d), hn::Set(d, 1)};

  for (std::int64_t i_bl = 0; i_bl < n_bl; ++i_bl) {
    std::int64_t i_a1 = a1(i_bl);
    std::int64_t i_a2 = a2(i_bl);

    for (std::int64_t i_rfi = 0; i_rfi < n_rfi; ++i_rfi) {
      for (std::int64_t i_f = 0; i_f < n_freq; ++i_f) {
        const auto i_f_fine_begin = i_f * n_int_f;
        for (std::int64_t i_t = 0; i_t < n_time; ++i_t) {
          const auto i_t_fine_begin = i_t * n_int_t;

          const auto vis_grad_scalar =
              rfi_vis_grad(i_bl, i_f, i_t) * n_int_inv;
          const auto val_rfi_vis_grad =
              ComplexV<D>{hn::Set(d, vis_grad_scalar.real()),
                          hn::Set(d, vis_grad_scalar.imag())};

          for (std::int64_t i_f_fine = i_f_fine_begin;
               i_f_fine < i_f_fine_begin + n_int_f; ++i_f_fine) {

            const std::int64_t i_tf_fine_base =
                i_f_fine * n_time * n_int_t + i_t_fine_begin;

            std::int64_t i_int_t = 0;
            for (; i_int_t + n_lanes <= n_int_t; i_int_t += n_lanes) {
              const auto i_tf_fine = i_tf_fine_base + i_int_t;

              const auto val_rfi_phase_1 =
                  hn::LoadU(d, &rfi_phase(i_rfi, i_a1, i_tf_fine));
              const auto val_rfi_phase_2 =
                  hn::LoadU(d, &rfi_phase(i_rfi, i_a2, i_tf_fine));
              const auto val_rfi_amp_1 =
                  LoadU(d, &rfi_amp_fine(i_rfi, i_a1, i_tf_fine));
              const auto val_rfi_amp_2 =
                  LoadU(d, &rfi_amp_fine(i_rfi, i_a2, i_tf_fine));

              const auto phase_diff =
                  hn::Sub(val_rfi_phase_1, val_rfi_phase_2);
              const auto val_c = hn::Cos(d, phase_diff);
              const auto val_s = hn::Sin(d, phase_diff);
              const auto val_e = ComplexV<D>{val_c, val_s};

              const auto t1 =
                  Mul(MulConj(val_rfi_vis_grad, val_rfi_amp_2), val_e);
              auto t2 = Mul(Mul(val_rfi_vis_grad, val_rfi_amp_1), val_e);
              t2 = ComplexV<D>{t2.re, hn::Neg(t2.im)};

              StoreAddU(d, t1, &rfi_amp_fine_grad(i_rfi, i_a1, i_tf_fine));
              StoreAddU(d, t2, &rfi_amp_fine_grad(i_rfi, i_a2, i_tf_fine));

              const auto f1 = Mul(
                  val_rfi_vis_grad,
                  Mul(i_unit,
                      Mul(val_e, MulConj(val_rfi_amp_1, val_rfi_amp_2))));

              auto *pg1 = &rfi_phase_grad(i_rfi, i_a1, i_tf_fine);
              auto *pg2 = &rfi_phase_grad(i_rfi, i_a2, i_tf_fine);
              hn::StoreU(hn::Add(hn::LoadU(d, pg1), f1.re), d, pg1);
              hn::StoreU(hn::Sub(hn::LoadU(d, pg2), f1.re), d, pg2);
            }

            for (; i_int_t < n_int_t; ++i_int_t) {
              const auto i_tf_fine = i_tf_fine_base + i_int_t;

              const auto val_rfi_amp_1 =
                  rfi_amp_fine(i_rfi, i_a1, i_tf_fine);
              const auto val_rfi_amp_2 =
                  rfi_amp_fine(i_rfi, i_a2, i_tf_fine);
              const auto val_rfi_phase_1 = rfi_phase(i_rfi, i_a1, i_tf_fine);
              const auto val_rfi_phase_2 = rfi_phase(i_rfi, i_a2, i_tf_fine);

              std::complex<double> val_e(
                  std::cos(val_rfi_phase_1 - val_rfi_phase_2),
                  std::sin(val_rfi_phase_1 - val_rfi_phase_2));

              const auto t1 =
                  vis_grad_scalar * std::conj(val_rfi_amp_2) * val_e;
              const auto t2 =
                  std::conj(vis_grad_scalar * val_rfi_amp_1 * val_e);

              rfi_amp_fine_grad(i_rfi, i_a1, i_tf_fine) += t1;
              rfi_amp_fine_grad(i_rfi, i_a2, i_tf_fine) += t2;

              const auto f1 = (vis_grad_scalar * std::complex<double>(0, 1) *
                               val_e * val_rfi_amp_1 *
                               std::conj(val_rfi_amp_2))
                                  .real();

              rfi_phase_grad(i_rfi, i_a1, i_tf_fine) += f1;
              rfi_phase_grad(i_rfi, i_a2, i_tf_fine) -= f1;
            }
          }
        }
      }
    }
  }
}

} // namespace HWY_NAMESPACE

#if HWY_ONCE

void rfi_transpose_kernel(std::int64_t n_int_f, std::int64_t n_int_t,
                          Tensor1D<const int *> a1, Tensor1D<const int *> a2,
                          Tensor3D<const std::complex<double> *> rfi_amp_fine,
                          Tensor3D<const double *> rfi_phase,
                          Tensor3D<const std::complex<double> *> rfi_vis_grad,
                          Tensor3D<std::complex<double> *> rfi_amp_fine_grad,
                          Tensor3D<double *> rfi_phase_grad) {

  // set to 0 before accumulation
  const auto out_size = rfi_amp_fine_grad.shape[0] *
                        rfi_amp_fine_grad.shape[1] * rfi_amp_fine_grad.shape[2];

  std::memset(rfi_amp_fine_grad.ptr, 0,
              sizeof(std::complex<double>) * out_size);
  std::memset(rfi_phase_grad.ptr, 0, sizeof(double) * out_size);

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

  for (std::int64_t i_bl = 0; i_bl < n_bl; ++i_bl) {
    std::int64_t i_a1 = a1(i_bl);
    std::int64_t i_a2 = a2(i_bl);

    for (std::int64_t i_rfi = 0; i_rfi < n_rfi; ++i_rfi) {
      for (std::int64_t i_tf_fine = 0; i_tf_fine < rfi_amp_fine.shape[2];
           ++i_tf_fine) {

        const auto i_t = (i_tf_fine % (n_time * n_int_t)) / n_int_t;
        const auto i_f = (i_tf_fine / (n_time * n_int_t)) / n_int_f;

        assert(i_t < rfi_vis_grad.shape[2]);
        assert(i_f < rfi_vis_grad.shape[1]);

        const auto val_rfi_vis_grad = rfi_vis_grad(i_bl, i_f, i_t) * n_int_inv;

        const auto val_rfi_amp_1 = rfi_amp_fine(i_rfi, i_a1, i_tf_fine);
        const auto val_rfi_amp_2 = rfi_amp_fine(i_rfi, i_a2, i_tf_fine);

        const auto val_rfi_phase_1 = rfi_phase(i_rfi, i_a1, i_tf_fine);
        const auto val_rfi_phase_2 = rfi_phase(i_rfi, i_a2, i_tf_fine);

        std::complex<double> val_e(std::cos(val_rfi_phase_1 - val_rfi_phase_2),
                                   std::sin(val_rfi_phase_1 - val_rfi_phase_2));

        const auto t1 = val_rfi_vis_grad * std::conj(val_rfi_amp_2) * val_e;
        const auto t2 = std::conj(val_rfi_vis_grad * val_rfi_amp_1 * val_e);

        rfi_amp_fine_grad(i_rfi, i_a1, i_tf_fine) += t1;
        rfi_amp_fine_grad(i_rfi, i_a2, i_tf_fine) += t2;

        const auto f1 = (val_rfi_vis_grad * std::complex<double>(0, 1) * val_e *
                         val_rfi_amp_1 * std::conj(val_rfi_amp_2))
                            .real();

        rfi_phase_grad(i_rfi, i_a1, i_tf_fine) += f1;
        rfi_phase_grad(i_rfi, i_a2, i_tf_fine) -= f1;
      }
    }
  }
}

using rfi_amp_fine_t = ffi::Buffer<ffi::C128, 6>;
using rfi_phase_t = ffi::Buffer<ffi::F64, 6>;

ffi::Error calc_rfi_transpose_cpu_impl(
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

  Tensor1D<const int *> a1_tensor(a1.typed_data(), a1.dimensions()[0]);
  Tensor1D<const int *> a2_tensor(a2.typed_data(), a2.dimensions()[0]);
  Tensor3D<const std::complex<double> *> rfi_amp_fine_tensor(
      rfi_amp_fine.typed_data(), rfi_amp_fine.dimensions()[0],
      rfi_amp_fine.dimensions()[1],
      rfi_amp_fine.dimensions()[2] * rfi_amp_fine.dimensions()[3] *
          rfi_amp_fine.dimensions()[4] * rfi_amp_fine.dimensions()[5]);
  Tensor3D<std::complex<double> *> rfi_amp_fine_grad_tensor(
      rfi_amp_fine_grad->typed_data(), rfi_amp_fine_grad->dimensions()[0],
      rfi_amp_fine_grad->dimensions()[1],
      rfi_amp_fine_grad->dimensions()[2] * rfi_amp_fine_grad->dimensions()[3] *
          rfi_amp_fine_grad->dimensions()[4] *
          rfi_amp_fine_grad->dimensions()[5]);
  Tensor3D<const double *> rfi_phase_tensor(
      rfi_phase.typed_data(), rfi_phase.dimensions()[0],
      rfi_phase.dimensions()[1],
      rfi_phase.dimensions()[2] * rfi_phase.dimensions()[3] *
          rfi_phase.dimensions()[4] * rfi_phase.dimensions()[5]);
  Tensor3D<double *> rfi_phase_grad_tensor(
      rfi_phase_grad->typed_data(), rfi_phase_grad->dimensions()[0],
      rfi_phase_grad->dimensions()[1],
      rfi_phase_grad->dimensions()[2] * rfi_phase_grad->dimensions()[3] *
          rfi_phase_grad->dimensions()[4] * rfi_phase_grad->dimensions()[5]);

  Tensor3D<const std::complex<double> *> rfi_grad_tensor(
      rfi_vis_grad.typed_data(), rfi_vis_grad.dimensions()[0],
      rfi_vis_grad.dimensions()[1], rfi_vis_grad.dimensions()[2]);

  const auto n_time = rfi_phase_grad->dimensions()[2];
  const auto n_freq = rfi_phase_grad->dimensions()[1];

  const auto n_int_t = rfi_amp_fine.dimensions()[5];
  const auto n_int_f = rfi_amp_fine.dimensions()[3];

  {
    auto start = std::chrono::high_resolution_clock::now();
    TABASCAL_EXPORT_AND_DISPATCH_T(rfi_transpose_kernel_opt)
    (n_int_f, n_int_t, a1_tensor, a2_tensor, rfi_amp_fine_tensor,
     rfi_phase_tensor, rfi_grad_tensor, rfi_amp_fine_grad_tensor,
     rfi_phase_grad_tensor);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cerr << "opt time: " << duration.count() << " ms" << std::endl;
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_transpose_cpu,
                              calc_rfi_transpose_cpu_impl,
                              ffi::Ffi::Bind()
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
