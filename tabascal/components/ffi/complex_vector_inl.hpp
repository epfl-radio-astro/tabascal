// This file must included inside a HWY_NAMESPACE. It must not have an include guard.

template <class D> struct ComplexV {
  hn::Vec<D> re, im;
};

template <class D>
HWY_INLINE HWY_ATTR ComplexV<D> Mul(ComplexV<D> a, ComplexV<D> b) {

  const auto re = hn::NegMulAdd(a.im, b.im, hn::Mul(a.re, b.re));
  const auto im = hn::MulAdd(a.re, b.im, hn::Mul(a.im, b.re));

  return ComplexV<D>{re, im};
};

template <class D>
HWY_INLINE HWY_ATTR ComplexV<D> MulConj(ComplexV<D> a, ComplexV<D> b) {

  const auto re = hn::MulAdd(a.im, b.im, hn::Mul(a.re, b.re));
  const auto im = hn::NegMulAdd(a.re, b.im, hn::Mul(a.im, b.re));

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

template <class D>
HWY_INLINE HWY_ATTR void StoreU(D d, ComplexV<D> value,
                                 std::complex<hn::TFromD<D>> *ptr) {

  using T = hn::TFromD<D>;

  constexpr std::int64_t n_lanes = hn::Lanes(d);

  auto scalar_ptr = reinterpret_cast<T *>(ptr);

  const auto lo = hn::InterleaveWholeLower(d, value.re, value.im);
  const auto hi = hn::InterleaveWholeUpper(d, value.re, value.im);

  hn::StoreU(lo, d, scalar_ptr);
  hn::StoreU(hi, d, scalar_ptr + n_lanes);
};

template <class D>
HWY_INLINE HWY_ATTR void StoreAddU(D d, ComplexV<D> value,
                                 std::complex<hn::TFromD<D>> *ptr) {

  using T = hn::TFromD<D>;

  constexpr std::int64_t n_lanes = hn::Lanes(d);

  auto scalar_ptr = reinterpret_cast<T *>(ptr);

  const auto lo = hn::InterleaveWholeLower(d, value.re, value.im);
  const auto hi = hn::InterleaveWholeUpper(d, value.re, value.im);

  const auto lo_res = hn::Add(hn::LoadU(d, scalar_ptr), lo);
  const auto hi_res = hn::Add(hn::LoadU(d, scalar_ptr + n_lanes), hi);

  hn::StoreU(lo_res, d, scalar_ptr);
  hn::StoreU(hi_res, d, scalar_ptr + n_lanes);
};

template <class D>
HWY_INLINE HWY_ATTR void StoreSubU(D d, ComplexV<D> value,
                                 std::complex<hn::TFromD<D>> *ptr) {

  using T = hn::TFromD<D>;

  constexpr std::int64_t n_lanes = hn::Lanes(d);

  auto scalar_ptr = reinterpret_cast<T *>(ptr);

  const auto lo = hn::InterleaveWholeLower(d, value.re, value.im);
  const auto hi = hn::InterleaveWholeUpper(d, value.re, value.im);

  const auto lo_res = hn::Sub(hn::LoadU(d, scalar_ptr), lo);
  const auto hi_res = hn::Sub(hn::LoadU(d, scalar_ptr + n_lanes), hi);

  hn::StoreU(lo_res, d, scalar_ptr);
  hn::StoreU(hi_res, d, scalar_ptr + n_lanes);
};

