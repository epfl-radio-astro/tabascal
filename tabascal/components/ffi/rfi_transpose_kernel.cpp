#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <complex>
#include <cassert>
#include <stdexcept>
#include <cstring>
#include <unistd.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "tensor.hpp"

namespace ffi = xla::ffi;

namespace tabascal {

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

    for (std::int64_t i_tf_fine = 0; i_tf_fine < rfi_amp_fine.shape[2];
         ++i_tf_fine) {

      const auto i_t = (i_tf_fine % (n_time * n_int_t)) / n_int_t;
      const auto i_f = (i_tf_fine / (n_time * n_int_t)) / n_int_f;

      assert(i_t < rfi_vis_grad.shape[2]);
      assert(i_f < rfi_vis_grad.shape[1]);

      const auto val_rfi_vis_grad = rfi_vis_grad(i_bl, i_f, i_t) * n_int_inv;

      for (std::int64_t i_rfi = 0; i_rfi < n_rfi; ++i_rfi) {

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

ffi::Error
calc_rfi_transpose_cpu_impl(ffi::BufferR1<ffi::S32> a1,
                            ffi::BufferR1<ffi::S32> a2,
                            rfi_amp_fine_t rfi_amp_fine, rfi_phase_t rfi_phase,
                            ffi::BufferR3<ffi::C128> rfi_vis_grad,
                            ffi::Result<rfi_amp_fine_t> rfi_amp_fine_grad,
                            ffi::Result<rfi_phase_t> rfi_phase_grad) {
  // rfi_amp_fine and rfi_phase shape is
  // (n_rfi, n_ant, n_freq, n_int_freq, n_time, n_int_time)

  // if (a1.dimensions().size() != 1) {
  //   return ffi::Error::InvalidArgument("Expected 1d a1");
  // }

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

  rfi_transpose_kernel(n_int_f, n_int_t, a1_tensor, a2_tensor,
                       rfi_amp_fine_tensor, rfi_phase_tensor, rfi_grad_tensor,
                       rfi_amp_fine_grad_tensor, rfi_phase_grad_tensor);

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_transpose_cpu,
                              calc_rfi_transpose_cpu_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<rfi_amp_fine_t>()
                                  .Arg<rfi_phase_t>()
                                  .Arg<ffi::BufferR3<ffi::C128>>()
                                  .Ret<rfi_amp_fine_t>()
                                  .Ret<rfi_phase_t>());
} // namespace tabascal
