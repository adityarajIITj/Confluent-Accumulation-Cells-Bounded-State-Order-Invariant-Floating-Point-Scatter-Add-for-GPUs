// bigint.h - fixed-width two's-complement integers of N 64-bit limbs.
//
// Used by the exact oracle and by every accumulator's readout.
//   BigInt   (384 bits)  binary32: a sum of 2^31 values on the 2^-149 grid is
//                        below 2^308; shifted 64-bit cells are below 2^340.
//   BigInt64 (2304 bits) binary64: a sum of 2^31 values on the 2^-1074 grid
//                        is below 2^2129; shifted cells are below 2^2144.
// No compiler extensions (no __int128) so that MSVC can build it.
#pragma once
#include "common.h"

namespace cac {

template <int N>
struct BigIntT {
  uint64_t w[N];

  BigIntT() { for (int i = 0; i < N; ++i) w[i] = 0; }

  // v * 2^shift with sign extension. Requires shift < 64 * (N - 1).
  static BigIntT from_i64_shifted(int64_t v, unsigned shift) {
    if (shift >= 64u * (unsigned)(N - 1)) die("BigInt shift out of range");
    BigIntT r;
    const uint64_t lo = (uint64_t)v;
    const uint64_t ext = (v < 0) ? ~0ULL : 0ULL;
    const unsigned idx = shift / 64, off = shift % 64;
    r.w[idx] = lo << off;
    r.w[idx + 1] = off ? ((ext << off) | (lo >> (64 - off))) : ext;
    for (unsigned i = idx + 2; i < (unsigned)N; ++i) r.w[i] = ext;
    return r;
  }

  void add(const BigIntT& o) {
    uint64_t carry = 0;
    for (int i = 0; i < N; ++i) {
      const uint64_t s = w[i] + o.w[i];
      const uint64_t c1 = (s < w[i]) ? 1u : 0u;
      const uint64_t s2 = s + carry;
      const uint64_t c2 = (s2 < s) ? 1u : 0u;
      w[i] = s2;
      carry = c1 | c2;
    }
  }
  void add_i64_shifted(int64_t v, unsigned shift) { add(from_i64_shifted(v, shift)); }

  bool is_neg() const { return (w[N - 1] >> 63) != 0; }
  bool is_zero() const {
    for (int i = 0; i < N; ++i) if (w[i]) return false;
    return true;
  }
  BigIntT negated() const {
    BigIntT r;
    uint64_t carry = 1;
    for (int i = 0; i < N; ++i) {
      const uint64_t x = ~w[i];
      r.w[i] = x + carry;
      carry = (r.w[i] < x) ? 1u : 0u;
    }
    return r;
  }
  BigIntT abs() const { return is_neg() ? negated() : *this; }
  BigIntT sub(const BigIntT& o) const { BigIntT r = *this; r.add(o.negated()); return r; }

  // The remaining queries assume a non-negative value.
  int bit_length() const {
    for (int i = N - 1; i >= 0; --i) if (w[i]) return i * 64 + bit_width_u64(w[i]);
    return 0;
  }
  bool bit(unsigned pos) const { return ((w[pos / 64] >> (pos % 64)) & 1u) != 0; }
  // count bits starting at pos, count <= 64.
  uint64_t extract_bits(unsigned pos, unsigned count) const {
    const unsigned idx = pos / 64, off = pos % 64;
    uint64_t v = w[idx] >> off;
    if (off && idx + 1 < (unsigned)N) v |= w[idx + 1] << (64 - off);
    if (count < 64) v &= (1ULL << count) - 1;
    return v;
  }
  // true if any bit strictly below pos is set.
  bool any_bits_below(unsigned pos) const {
    const unsigned idx = pos / 64, off = pos % 64;
    for (unsigned i = 0; i < idx; ++i) if (w[i]) return true;
    if (off && (w[idx] & ((1ULL << off) - 1))) return true;
    return false;
  }
  BigIntT shl(unsigned s) const {
    BigIntT r;
    const unsigned idx = s / 64, off = s % 64;
    for (int i = N - 1; i >= 0; --i) {
      uint64_t v = 0;
      const int src = i - (int)idx;
      if (src >= 0) {
        v = w[src] << off;
        if (off && src - 1 >= 0) v |= w[src - 1] >> (64 - off);
      }
      r.w[i] = v;
    }
    return r;
  }
  // Signed three-way comparison.
  static int cmp(const BigIntT& a, const BigIntT& b) {
    const bool an = a.is_neg(), bn = b.is_neg();
    if (an != bn) return an ? -1 : 1;
    for (int i = N - 1; i >= 0; --i) {
      if (a.w[i] != b.w[i]) return (a.w[i] < b.w[i]) ? -1 : 1;
    }
    return 0;
  }
  // Approximate magnitude as a double (for reporting only, never for decisions).
  double to_double_approx() const {
    const BigIntT a = abs();
    double r = 0.0;
    for (int i = N - 1; i >= 0; --i) r = r * 18446744073709551616.0 + (double)a.w[i];
    return is_neg() ? -r : r;
  }
};

typedef BigIntT<6> BigInt;
typedef BigIntT<36> BigInt64;

}  // namespace cac
