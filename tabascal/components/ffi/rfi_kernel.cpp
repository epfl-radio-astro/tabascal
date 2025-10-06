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

ffi::Error calc_rfi_vis_cpu_impl(ffi::BufferR1<ffi::S32> a1,
                                 ffi::BufferR1<ffi::S32> a2,
                                 rfi_amp_fine_t rfi_amp_fine,
                                 rfi_phase_t rfi_phase,
                                 ffi::ResultBufferR3<ffi::C128> rfi_vis) {
  // rfi_amp_fine and rfi_phase shape is
  // (n_rfi, n_ant, n_freq, n_int_freq, n_time, n_int_time)

  // if (a1.dimensions().size() != 1) {
  //   return ffi::Error::InvalidArgument("Expected 1d a1");
  // }

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

  rfi_kernel(a1_tensor, a2_tensor, rfi_amp_fine_tensor, rfi_phase_tensor,
             rfi_vis_tensor);

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_vis_cpu, calc_rfi_vis_cpu_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<rfi_amp_fine_t>()
                                  .Arg<rfi_phase_t>()
                                  .Ret<ffi::BufferR3<ffi::C128>>());

} // namespace tabascal
