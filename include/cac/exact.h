// exact.h - correctly rounded conversion, format traits, and the exact oracle.
#pragma once
#include "bigint.h"
#include "fp32.h"

namespace cac {

// Round a positive integer A to at most p significant bits, ties to even.
// On return the rounded value is m * 2^q. Values below 2^p are returned
// unchanged with q = 0 (this is the subnormal / smallest-binade range).
template <int N>
inline void round_core(const BigIntT<N>& A, int p, uint64_t* m, int* q) {
  const int h = A.bit_length();
  if (h <= p) { *m = A.w[0]; *q = 0; return; }
  int sh = h - p;
  uint64_t mm = A.extract_bits((unsigned)sh, (unsigned)p);
  const bool half = A.bit((unsigned)(sh - 1));
  const bool sticky = (sh - 1 > 0) && A.any_bits_below((unsigned)(sh - 1));
  if (half && (sticky || (mm & 1u))) {
    ++mm;
    if (mm == (1ULL << p)) { mm >>= 1; ++sh; }
  }
  *m = mm;
  *q = sh;
}

// Round an exact integer S (in units of 2^-149) to the nearest binary32,
// ties to even, with IEEE overflow to infinity. S == 0 returns +0 or -0
// according to zero_is_negative.
inline float round_to_f32(const BigInt& S, bool zero_is_negative = false) {
  if (S.is_zero()) return u2f(zero_is_negative ? 0x80000000u : 0u);
  const bool neg = S.is_neg();
  uint64_t m;
  int q;
  round_core(neg ? S.negated() : S, 24, &m, &q);
  uint32_t u;
  if (m < (1u << 23)) {
    u = (uint32_t)m;  // subnormal; q is 0 here by construction
  } else {
    const int be = q + 1;
    u = (be >= 255) ? kPosInf : (((uint32_t)be << 23) | ((uint32_t)m & 0x7fffffu));
  }
  if (neg) u |= 0x80000000u;
  return u2f(u);
}

// The same for binary64 (units of 2^-1074).
inline double round_to_f64(const BigInt64& S, bool zero_is_negative = false) {
  if (S.is_zero()) return u2d(zero_is_negative ? 0x8000000000000000ULL : 0ULL);
  const bool neg = S.is_neg();
  uint64_t m;
  int q;
  round_core(neg ? S.negated() : S, 53, &m, &q);
  uint64_t u;
  if (m < (1ULL << 52)) {
    u = m;
  } else {
    const int be = q + 1;
    u = (be >= 2047) ? kPosInf64 : (((uint64_t)be << 52) | (m & 0x000fffffffffffffULL));
  }
  if (neg) u |= 0x8000000000000000ULL;
  return u2d(u);
}

// Format traits: everything an accumulator needs to know about a binary
// floating-point format.
struct Fmt32 {
  typedef float T;
  typedef BigInt Big;
  static const int kMant = 24;     // significand bits
  static const int kGrid = 277;    // |grid image| < 2^kGrid
  static const int kUnitExp = 149; // the grid unit is 2^-kUnitExp
  static const int kBytes = 4;
  static const char* suffix() { return ""; }
  static T round(const Big& S, bool negzero = false) { return round_to_f32(S, negzero); }
  static T nan() { return u2f(kCanonNaN); }
  static T inf(bool neg) { return u2f(neg ? kNegInf : kPosInf); }
  static T zero(bool neg) { return u2f(neg ? 0x80000000u : 0u); }
  static uint64_t bits(T x) { return f2u(x); }
  static T from_bits(uint64_t b) { return u2f((uint32_t)b); }
};

struct Fmt64 {
  typedef double T;
  typedef BigInt64 Big;
  static const int kMant = 53;
  static const int kGrid = 2098;
  static const int kUnitExp = 1074;
  static const int kBytes = 8;
  static const char* suffix() { return "64"; }
  static T round(const Big& S, bool negzero = false) { return round_to_f64(S, negzero); }
  static T nan() { return u2d(kCanonNaN64); }
  static T inf(bool neg) { return u2d(neg ? kNegInf64 : kPosInf64); }
  static T zero(bool neg) { return u2d(neg ? 0x8000000000000000ULL : 0ULL); }
  static uint64_t bits(T x) { return d2u(x); }
  static T from_bits(uint64_t b) { return u2d(b); }
};

template <class T> struct FmtOf;
template <> struct FmtOf<float> { typedef Fmt32 type; };
template <> struct FmtOf<double> { typedef Fmt64 type; };

// Exact grid image of a finite value.
template <class T>
inline typename FmtOf<T>::type::Big exact_of(T f) {
  typedef typename FmtOf<T>::type::Big Big;
  const Dec d = decode(f);
  if (d.cls != FClass::Finite) return Big();
  return Big::from_i64_shifted(d.neg ? -(int64_t)d.m : (int64_t)d.m, (unsigned)d.q);
}

// Exact accumulator: the mathematical sum of everything deposited, with IEEE
// conventions for special values:
//   any NaN, or both +Inf and -Inf  -> NaN
//   only +Inf / only -Inf           -> that infinity
//   exact sum == 0                  -> -0 if every contribution was -0, else +0
//   otherwise                       -> the correctly rounded exact sum
// This is the value an IEEE summation would produce in any order if no
// intermediate rounding or overflow occurred. It is the oracle.
template <class Fm>
struct ExactAccT {
  typename Fm::Big sum;
  bool nan = false, pinf = false, ninf = false;
  bool all_negzero = true, any = false;

  void add(typename Fm::T x) {
    const Dec d = decode(x);
    any = true;
    switch (d.cls) {
      case FClass::NaN: nan = true; all_negzero = false; break;
      case FClass::Inf: (d.neg ? ninf : pinf) = true; all_negzero = false; break;
      case FClass::Zero: if (!d.neg) all_negzero = false; break;
      case FClass::Finite:
        all_negzero = false;
        sum.add_i64_shifted(d.neg ? -(int64_t)d.m : (int64_t)d.m, (unsigned)d.q);
        break;
    }
  }
  typename Fm::T result() const {
    if (nan || (pinf && ninf)) return Fm::nan();
    if (pinf) return Fm::inf(false);
    if (ninf) return Fm::inf(true);
    if (sum.is_zero()) return Fm::zero(any && all_negzero);
    return Fm::round(sum);
  }
};
typedef ExactAccT<Fmt32> ExactAcc;
typedef ExactAccT<Fmt64> ExactAcc64;

}  // namespace cac
