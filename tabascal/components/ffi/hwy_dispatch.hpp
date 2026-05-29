// no include guard for hwy/foreach_target.h to work

// Disable scalable vector architetures, because we rely on vector lengths at compile time for
// optizations. Fixed-length SVE targets (SVE_256, SVE2_128) are also disabled because SVE
// vector types are sizeless per ACLE and cannot be used as struct members (e.g. ComplexV).
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS HWY_ALL_SVE
#endif

#ifndef HWY_TARGET_INCLUDE
#error "HWY_TARGET_INCLUDE must be defined before including this header file"
#endif

#ifdef TABASCAL_MULTI_ARCH

// must come before highway.h
#include <hwy/foreach_target.h>
// -----
#include <hwy/highway.h>

#include <hwy/contrib/math/math-inl.h>

#undef TABASCAL_EXPORT_AND_DISPATCH_T
#undef TABASCAL_DISPATCH
#undef TABASCAL_EXPORT_FUNC

#define TABASCAL_EXPORT_AND_DISPATCH_T(arg) HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(arg)
#define TABASCAL_DISPATCH(arg) HWY_DYNAMIC_DISPATCH(arg)
#define TABASCAL_EXPORT_FUNC(arg) HWY_EXPORT(arg)

#else

#include <hwy/highway.h>

#include <hwy/contrib/math/math-inl.h>

#undef TABASCAL_EXPORT_AND_DISPATCH_T
#undef TABASCAL_DISPATCH
#undef TABASCAL_EXPORT_FUNC

#define TABASCAL_EXPORT_AND_DISPATCH_T(arg) HWY_STATIC_DISPATCH(arg)
#define TABASCAL_DISPATCH(arg) HWY_STATIC_DISPATCH(arg)
#define TABASCAL_EXPORT_FUNC(arg)

#endif

// define a fixed size vector type
namespace tabascal {
namespace HWY_NAMESPACE { // required: unique per target
#if HWY_HAVE_SCALABLE
// Use fixed vector length for SVE / SVE2
// Highway guarantees fixed tag size of 16 / sizeof(T)
// Most SVE2 hardware implementations use vector size of 16 bytes (128 bits)
template <typename T>
using TagType = ::hwy::HWY_NAMESPACE::FixedTag<T, 16 / sizeof(T)>;
#else
template <typename T> using TagType = ::hwy::HWY_NAMESPACE::ScalableTag<T>;
#endif
} // namespace HWY_NAMESPACE
} // namespace tabascal
