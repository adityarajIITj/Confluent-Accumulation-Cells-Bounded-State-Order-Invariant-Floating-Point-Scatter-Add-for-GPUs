// Confluent Accumulation Cells (CAC) reference simulator.
// common.h - portable helpers: bit utilities, PRNG, hashing.
//
// Portability rule for the whole code base: every number reported by the
// simulator must be a function of the seed alone. We therefore avoid
// std::shuffle / std::*_distribution (implementation-defined sequences) and
// libm transcendental functions (not correctly rounded on every platform).
// Only +, -, *, /, sqrt and ldexp are used on floating-point data; these are
// exactly specified by IEEE 754.
#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace cac {

// Number of bits needed to represent x (0 for x == 0).
inline int bit_width_u64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return x ? 64 - __builtin_clzll(x) : 0;
#else
  int n = 0;
  while (x) { ++n; x >>= 1; }
  return n;
#endif
}

// xoshiro256** seeded through SplitMix64. Integer-only, portable.
struct Rng {
  uint64_t s[4];
  explicit Rng(uint64_t seed) {
    uint64_t z = seed;
    for (int i = 0; i < 4; ++i) {
      z += 0x9E3779B97F4A7C15ULL;
      uint64_t t = z;
      t = (t ^ (t >> 30)) * 0xBF58476D1CE4E5B9ULL;
      t = (t ^ (t >> 27)) * 0x94D049BB133111EBULL;
      s[i] = t ^ (t >> 31);
    }
  }
  static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  uint64_t next() {
    const uint64_t r = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return r;
  }
  // Uniform integer in [0, n). Multiply-shift on the top 32 bits; the bias is
  // below n / 2^32 and irrelevant here, and the result is platform independent.
  uint32_t below(uint32_t n) { return (uint32_t)(((next() >> 32) * (uint64_t)n) >> 32); }
  // Uniform double in [0, 1) with 53 random bits (exact arithmetic).
  double uniform01() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
  // Approximately standard normal: Irwin-Hall sum of 12 uniforms minus 6.
  // Uses additions only, so the value is bit-identical on every IEEE platform.
  double normal() {
    double a = 0.0;
    for (int i = 0; i < 12; ++i) a += uniform01();
    return a - 6.0;
  }
};

// Fisher-Yates shuffle driven by Rng (portable replacement for std::shuffle).
template <class T>
inline void shuffle_vec(std::vector<T>& v, Rng& rng) {
  for (size_t i = v.size(); i > 1; --i) {
    size_t j = rng.below((uint32_t)i);
    T tmp = v[i - 1]; v[i - 1] = v[j]; v[j] = tmp;
  }
}

// FNV-1a 64-bit running hash.
struct Fnv {
  uint64_t h = 0xcbf29ce484222325ULL;
  void byte(uint8_t b) { h ^= b; h *= 0x100000001b3ULL; }
  void u32(uint32_t x) { for (int i = 0; i < 4; ++i) byte((uint8_t)(x >> (8 * i))); }
  void u64(uint64_t x) { for (int i = 0; i < 8; ++i) byte((uint8_t)(x >> (8 * i))); }
};

[[noreturn]] inline void die(const char* msg) {
  std::fprintf(stderr, "fatal: %s\n", msg);
  std::exit(2);
}

}  // namespace cac
