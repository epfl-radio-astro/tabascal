#pragma once

#include <cassert>
#include <cstdint>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#ifdef __CUDACC__
#define TAB_H_D __host__ __device__
#else
#define TAB_H_D
#endif

namespace ffi = xla::ffi;

namespace tabascal {
template <typename POINTER_T, typename INDEX_T = std::int64_t> struct Tensor1D {
  Tensor1D(POINTER_T p, INDEX_T s0) : ptr(p), shape{s0} {}

  TAB_H_D inline auto &operator()(INDEX_T i0) {
    assert(i0 < shape[0]);
    return ptr[i0];
  }

  POINTER_T ptr = nullptr;
  const INDEX_T shape[1] = {0};
};

template <typename POINTER_T, typename INDEX_T = std::int64_t> struct Tensor2D {
  Tensor2D(POINTER_T p, INDEX_T s0, INDEX_T s1) : ptr(p), shape{s0, s1} {}

  TAB_H_D inline auto &operator()(INDEX_T i0, INDEX_T i1) {
    assert(i0 < shape[0]);
    assert(i1 < shape[1]);
    assert(i1 + shape[1] * i0 < shape[0] * shape[1]);
    return ptr[i1 + shape[1] * i0];
  }

  TAB_H_D inline Tensor1D<POINTER_T, INDEX_T> slice(INDEX_T i0) {
    return Tensor1D<POINTER_T, INDEX_T>(ptr + i0 * shape[1], shape[0]);
  }

  POINTER_T ptr = nullptr;
  const INDEX_T shape[2] = {0};
};

template <typename POINTER_T, typename INDEX_T = std::int64_t> struct Tensor3D {
  Tensor3D(POINTER_T p, INDEX_T s0, INDEX_T s1, INDEX_T s2)
      : ptr(p), shape{s0, s1, s2} {}

  TAB_H_D inline auto &operator()(INDEX_T i0, INDEX_T i1, INDEX_T i2) {
    assert(i0 < shape[0]);
    assert(i1 < shape[1]);
    assert(i2 < shape[2]);
    assert(i2 + shape[2] * (i1 + i0 * shape[1]) <
           shape[0] * shape[1] * shape[2]);
    return ptr[i2 + shape[2] * (i1 + i0 * shape[1])];
  }

  TAB_H_D inline Tensor2D<POINTER_T, INDEX_T> slice(INDEX_T i0) {
    return Tensor2D<POINTER_T, INDEX_T>(ptr + i0 * shape[1] * shape[2],
                                        shape[0], shape[1]);
  }

  POINTER_T ptr = nullptr;
  const INDEX_T shape[3] = {0};
};

template <typename POINTER_T, typename INDEX_T = std::int64_t> struct Tensor4D {
  Tensor4D(POINTER_T p, INDEX_T s0, INDEX_T s1, INDEX_T s2, INDEX_T s3)
      : ptr(p), shape{s0, s1, s2, s3} {}

  TAB_H_D inline auto &operator()(INDEX_T i0, INDEX_T i1, INDEX_T i2,
                                  INDEX_T i3) {
    assert(i0 < shape[0]);
    assert(i1 < shape[1]);
    assert(i2 < shape[2]);
    assert(i3 < shape[3]);
    return ptr[i3 + shape[3] * (i2 + shape[2] * (i1 + i0 * shape[1]))];
  }

  TAB_H_D inline Tensor3D<POINTER_T, INDEX_T> slice(INDEX_T i0) {
    return Tensor3D<POINTER_T, INDEX_T>(ptr +
                                            i0 * shape[1] * shape[2] * shape[3],
                                        shape[0], shape[1], shape[2]);
  }

  POINTER_T ptr = nullptr;
  const INDEX_T shape[4] = {0};
};

template <typename POINTER_T, typename INDEX_T = std::int64_t> struct Tensor5D {
  Tensor5D(POINTER_T p, INDEX_T s0, INDEX_T s1, INDEX_T s2, INDEX_T s3,
           INDEX_T s4)
      : ptr(p), shape{s0, s1, s2, s3, s4} {}

  TAB_H_D inline auto &operator()(INDEX_T i0, INDEX_T i1, INDEX_T i2,
                                  INDEX_T i3, INDEX_T i4) {
    assert(i0 < shape[0]);
    assert(i1 < shape[1]);
    assert(i2 < shape[2]);
    assert(i3 < shape[3]);
    assert(i4 < shape[4]);
    return ptr[i4 +
               shape[4] *
                   (i3 + shape[3] * (i2 + shape[2] * (i1 + i0 * shape[1])))];
  }

  TAB_H_D inline Tensor4D<POINTER_T, INDEX_T> slice(INDEX_T i0) {
    return Tensor4D<POINTER_T, INDEX_T>(ptr + i0 * shape[1] * shape[2] *
                                                  shape[3] * shape[4],
                                        shape[0], shape[1], shape[2], shape[3]);
  }

  POINTER_T ptr = nullptr;
  const INDEX_T shape[5] = {0};
};

} // namespace tabascal
