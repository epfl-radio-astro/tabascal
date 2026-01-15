#pragma once

#include <cassert>
#include <cstdint>

#ifdef __CUDACC__
#define TAB_H_D __host__ __device__
#else
#define TAB_H_D
#endif

namespace tabascal {
/**
 * @brief A 1D tensor view wrapper for convenient indexing.
 *
 * @tparam POINTER_T The type of the pointer (e.g., int*, const float*).
 * @tparam INDEX_T The type of the index (default: std::int64_t).
 */
template <typename POINTER_T, typename INDEX_T = std::int64_t> struct Tensor1D {
  /**
   * @brief Construct a new Tensor1D object.
   *
   * @param p Pointer to the data.
   * @param s0 Size of the dimension.
   */
  Tensor1D(POINTER_T p, INDEX_T s0) : ptr(p), shape{s0} {}

  /**
   * @brief Access element at index i0.
   *
   * @param i0 Index.
   * @return Reference to the element.
   */
  TAB_H_D inline auto &operator()(INDEX_T i0) {
    assert(i0 < shape[0]);
    return ptr[i0];
  }

  POINTER_T ptr = nullptr;      ///< Pointer to the data buffer.
  const INDEX_T shape[1] = {0}; ///< Shape of the tensor.
};

/**
 * @brief A 2D tensor view wrapper for convenient indexing.
 *
 * @tparam POINTER_T The type of the pointer.
 * @tparam INDEX_T The type of the index (default: std::int64_t).
 */
template <typename POINTER_T, typename INDEX_T = std::int64_t> struct Tensor2D {
  /**
   * @brief Construct a new Tensor2D object.
   *
   * @param p Pointer to the data.
   * @param s0 Size of the first dimension.
   * @param s1 Size of the second dimension.
   */
  Tensor2D(POINTER_T p, INDEX_T s0, INDEX_T s1) : ptr(p), shape{s0, s1} {}

  /**
   * @brief Access element at indices (i0, i1).
   *
   * @param i0 First dimension index.
   * @param i1 Second dimension index.
   * @return Reference to the element.
   */
  TAB_H_D inline auto &operator()(INDEX_T i0, INDEX_T i1) {
    assert(i0 < shape[0]);
    assert(i1 < shape[1]);
    assert(i1 + shape[1] * i0 < shape[0] * shape[1]);
    return ptr[i1 + shape[1] * i0];
  }

  /**
   * @brief Create a slice along the first dimension.
   *
   * @param i0 Index of the first dimension slice.
   * @return A Tensor1D representing the slice.
   */
  TAB_H_D inline Tensor1D<POINTER_T, INDEX_T> slice(INDEX_T i0) {
    return Tensor1D<POINTER_T, INDEX_T>(ptr + i0 * shape[1], shape[0]);
  }

  POINTER_T ptr = nullptr;      ///< Pointer to the data buffer.
  const INDEX_T shape[2] = {0}; ///< Shape of the tensor.
};

/**
 * @brief A 3D tensor view wrapper for convenient indexing.
 *
 * @tparam POINTER_T The type of the pointer.
 * @tparam INDEX_T The type of the index (default: std::int64_t).
 */
template <typename POINTER_T, typename INDEX_T = std::int64_t> struct Tensor3D {
  /**
   * @brief Construct a new Tensor3D object.
   *
   * @param p Pointer to the data.
   * @param s0 Size of the first dimension.
   * @param s1 Size of the second dimension.
   * @param s2 Size of the third dimension.
   */
  Tensor3D(POINTER_T p, INDEX_T s0, INDEX_T s1, INDEX_T s2)
      : ptr(p), shape{s0, s1, s2} {}

  /**
   * @brief Access element at indices (i0, i1, i2).
   *
   * @param i0 First dimension index.
   * @param i1 Second dimension index.
   * @param i2 Third dimension index.
   * @return Reference to the element.
   */
  TAB_H_D inline auto &operator()(INDEX_T i0, INDEX_T i1, INDEX_T i2) {
    assert(i0 < shape[0]);
    assert(i1 < shape[1]);
    assert(i2 < shape[2]);
    assert(i2 + shape[2] * (i1 + i0 * shape[1]) <
           shape[0] * shape[1] * shape[2]);
    return ptr[i2 + shape[2] * (i1 + i0 * shape[1])];
  }

  /**
   * @brief Create a slice along the first dimension.
   *
   * @param i0 Index of the first dimension slice.
   * @return A Tensor2D representing the slice.
   */
  TAB_H_D inline Tensor2D<POINTER_T, INDEX_T> slice(INDEX_T i0) {
    return Tensor2D<POINTER_T, INDEX_T>(ptr + i0 * shape[1] * shape[2],
                                        shape[0], shape[1]);
  }

  POINTER_T ptr = nullptr;      ///< Pointer to the data buffer.
  const INDEX_T shape[3] = {0}; ///< Shape of the tensor.
};

/**
 * @brief A 4D tensor view wrapper for convenient indexing.
 *
 * @tparam POINTER_T The type of the pointer.
 * @tparam INDEX_T The type of the index (default: std::int64_t).
 */
template <typename POINTER_T, typename INDEX_T = std::int64_t> struct Tensor4D {
  /**
   * @brief Construct a new Tensor4D object.
   *
   * @param p Pointer to the data.
   * @param s0 Size of the first dimension.
   * @param s1 Size of the second dimension.
   * @param s2 Size of the third dimension.
   * @param s3 Size of the fourth dimension.
   */
  Tensor4D(POINTER_T p, INDEX_T s0, INDEX_T s1, INDEX_T s2, INDEX_T s3)
      : ptr(p), shape{s0, s1, s2, s3} {}

  /**
   * @brief Access element at indices (i0, i1, i2, i3).
   *
   * @param i0 First dimension index.
   * @param i1 Second dimension index.
   * @param i2 Third dimension index.
   * @param i3 Fourth dimension index.
   * @return Reference to the element.
   */
  TAB_H_D inline auto &operator()(INDEX_T i0, INDEX_T i1, INDEX_T i2,
                                  INDEX_T i3) {
    assert(i0 < shape[0]);
    assert(i1 < shape[1]);
    assert(i2 < shape[2]);
    assert(i3 < shape[3]);
    return ptr[i3 + shape[3] * (i2 + shape[2] * (i1 + i0 * shape[1]))];
  }

  /**
   * @brief Create a slice along the first dimension.
   *
   * @param i0 Index of the first dimension slice.
   * @return A Tensor3D representing the slice.
   */
  TAB_H_D inline Tensor3D<POINTER_T, INDEX_T> slice(INDEX_T i0) {
    return Tensor3D<POINTER_T, INDEX_T>(ptr +
                                            i0 * shape[1] * shape[2] * shape[3],
                                        shape[0], shape[1], shape[2]);
  }

  POINTER_T ptr = nullptr;      ///< Pointer to the data buffer.
  const INDEX_T shape[4] = {0}; ///< Shape of the tensor.
};

/**
 * @brief A 5D tensor view wrapper for convenient indexing.
 *
 * @tparam POINTER_T The type of the pointer.
 * @tparam INDEX_T The type of the index (default: std::int64_t).
 */
template <typename POINTER_T, typename INDEX_T = std::int64_t> struct Tensor5D {
  /**
   * @brief Construct a new Tensor5D object.
   *
   * @param p Pointer to the data.
   * @param s0 Size of the first dimension.
   * @param s1 Size of the second dimension.
   * @param s2 Size of the third dimension.
   * @param s3 Size of the fourth dimension.
   * @param s4 Size of the fifth dimension.
   */
  Tensor5D(POINTER_T p, INDEX_T s0, INDEX_T s1, INDEX_T s2, INDEX_T s3,
           INDEX_T s4)
      : ptr(p), shape{s0, s1, s2, s3, s4} {}

  /**
   * @brief Access element at indices (i0, i1, i2, i3, i4).
   *
   * @param i0 First dimension index.
   * @param i1 Second dimension index.
   * @param i2 Third dimension index.
   * @param i3 Fourth dimension index.
   * @param i4 Fifth dimension index.
   * @return Reference to the element.
   */
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

  /**
   * @brief Create a slice along the first dimension.
   *
   * @param i0 Index of the first dimension slice.
   * @return A Tensor4D representing the slice.
   */
  TAB_H_D inline Tensor4D<POINTER_T, INDEX_T> slice(INDEX_T i0) {
    return Tensor4D<POINTER_T, INDEX_T>(ptr + i0 * shape[1] * shape[2] *
                                                  shape[3] * shape[4],
                                        shape[0], shape[1], shape[2], shape[3]);
  }

  POINTER_T ptr = nullptr;      ///< Pointer to the data buffer.
  const INDEX_T shape[5] = {0}; ///< Shape of the tensor.
};

} // namespace tabascal
