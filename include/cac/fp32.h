// fp32.h - IEEE 754 binary32 and binary64 decomposition on an absolute grid.
//
// Every finite value x of a binary format can be written exactly as
//     x = (-1)^neg * m * 2^q * u,
// where u is the smallest subnormal of the format (2^-149 or 2^-1074), m is
// the integer significand and q >= 0. The signed integer X = +-(m << q) is
// the "grid image" of x. All accumulators that claim order invariance work on
// grid images.
//
//                 significand   q range     |X| bound
//   binary32      m < 2^24      0 .. 253    2^277
//   binary64      m < 2^53      0 .. 2045   2^2098
#pragma once
#include "common.h"

namespace cac {

inline uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
inline uint64_t d2u(double d) { uint64_t u; std::memcpy(&u, &d, 8); return u; }
inline double u2d(uint64_t u) { double d; std::memcpy(&d, &u, 8); return d; }

const uint32_t kCanonNaN = 0x7fc00000u;
const uint32_t kPosInf = 0x7f800000u;
const uint32_t kNegInf = 0xff800000u;
const uint64_t kCanonNaN64 = 0x7ff8000000000000ULL;
const uint64_t kPosInf64 = 0x7ff0000000000000ULL;
const uint64_t kNegInf64 = 0xfff0000000000000ULL;

enum class FClass : uint8_t { Zero, Finite, Inf, NaN };

struct Dec {
  FClass cls;
  bool neg;
  uint64_t m;  // integer significand
  int q;       // exponent on the absolute grid
};

inline Dec decode(float f) {
  const uint32_t u = f2u(f);
  Dec d;
  d.neg = (u >> 31) != 0;
  const uint32_t be = (u >> 23) & 0xffu;
  const uint32_t fr = u & 0x7fffffu;
  if (be == 255u) { d.cls = fr ? FClass::NaN : FClass::Inf; d.m = 0; d.q = 0; }
  else if (be == 0u) { d.m = fr; d.q = 0; d.cls = fr ? FClass::Finite : FClass::Zero; }
  else { d.m = fr | 0x800000u; d.q = (int)be - 1; d.cls = FClass::Finite; }
  return d;
}

inline Dec decode(double f) {
  const uint64_t u = d2u(f);
  Dec d;
  d.neg = (u >> 63) != 0;
  const uint64_t be = (u >> 52) & 0x7ffu;
  const uint64_t fr = u & 0x000fffffffffffffULL;
  if (be == 2047u) { d.cls = fr ? FClass::NaN : FClass::Inf; d.m = 0; d.q = 0; }
  else if (be == 0u) { d.m = fr; d.q = 0; d.cls = fr ? FClass::Finite : FClass::Zero; }
  else { d.m = fr | 0x0010000000000000ULL; d.q = (int)be - 1; d.cls = FClass::Finite; }
  return d;
}

inline bool is_nan(float f) { return (f2u(f) & 0x7fffffffu) > 0x7f800000u; }
inline bool is_inf(float f) { return (f2u(f) & 0x7fffffffu) == 0x7f800000u; }
inline bool is_finite(float f) { return (f2u(f) & 0x7f800000u) != 0x7f800000u; }
inline bool is_nan(double f) { return (d2u(f) & 0x7fffffffffffffffULL) > kPosInf64; }
inline bool is_inf(double f) { return (d2u(f) & 0x7fffffffffffffffULL) == kPosInf64; }
inline bool is_finite(double f) { return (d2u(f) & kPosInf64) != kPosInf64; }
inline bool sign_bit(float f) { return (f2u(f) >> 31) != 0; }
inline bool sign_bit(double f) { return (d2u(f) >> 63) != 0; }

// Monotone integer index of a finite value: adjacent representable values
// differ by 1; +0 and -0 both map to 0. Used to count "values apart".
inline int64_t ordinal(float f) {
  const uint32_t u = f2u(f);
  const int64_t mag = (int64_t)(u & 0x7fffffffu);
  return (u >> 31) ? -mag : mag;
}
inline int64_t ordinal(double f) {
  const uint64_t u = d2u(f);
  const int64_t mag = (int64_t)(u & 0x7fffffffffffffffULL);
  return (u >> 63) ? -mag : mag;
}

// Bit pattern with all NaNs collapsed to one canonical quiet NaN, so that
// result comparison does not depend on NaN payload or sign.
inline uint64_t canon_bits(float f) { return is_nan(f) ? kCanonNaN : f2u(f); }
inline uint64_t canon_bits(double f) { return is_nan(f) ? kCanonNaN64 : d2u(f); }

}  // namespace cac
