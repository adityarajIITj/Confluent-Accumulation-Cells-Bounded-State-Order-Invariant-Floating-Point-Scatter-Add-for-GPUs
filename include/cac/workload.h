// workload.h - seeded scatter workloads.
//
// A workload is a list of (destination, value) contributions plus the initial
// destination values. Generators use only exactly specified floating-point
// operations so that the same seed produces the same bits on every platform.
#pragma once
#include "accum.h"

namespace cac {

template <class T>
struct WorkloadT {
  std::string name;
  uint32_t D = 0;
  std::vector<T> init;
  std::vector<ContributionT<T>> c;
};
typedef WorkloadT<float> Workload;
typedef WorkloadT<double> Workload64;

enum class DstPattern { Uniform, Zipf, Splat };
enum class ValDist { Normal, Wide, Cancel, Grad, Overflow, Subnormal };

inline const char* dst_name(DstPattern p) {
  switch (p) {
    case DstPattern::Uniform: return "uniform";
    case DstPattern::Zipf: return "zipf";
    case DstPattern::Splat: return "splat";
  }
  return "?";
}
inline const char* val_name(ValDist v) {
  switch (v) {
    case ValDist::Normal: return "normal";
    case ValDist::Wide: return "wide";
    case ValDist::Cancel: return "cancel";
    case ValDist::Grad: return "grad";
    case ValDist::Overflow: return "overflow";
    case ValDist::Subnormal: return "subnormal";
  }
  return "?";
}

// Geometric draw: number of leading zeros of a random 32-bit word.
inline int draw_geometric(Rng& rng) {
  const uint32_t r = (uint32_t)(rng.next() >> 32);
  int g = 0;
  while (g < 31 && !((r >> (31 - g)) & 1u)) ++g;
  return g;
}

// Draw one binary32 value. Cancel/Overflow return the first element of a pair
// and the partner through *pair (the caller sends both to one destination).
inline float draw_value(ValDist v, Rng& rng, float* pair, bool* has_pair) {
  *has_pair = false;
  switch (v) {
    case ValDist::Normal:
      return (float)rng.normal();
    case ValDist::Wide: {
      // sign * [1,2) * 2^e, e uniform in [-40, 40]: about 24 decades.
      const float mant = (float)(1.0 + rng.uniform01());
      const int e = (int)rng.below(81) - 40;
      const float x = std::ldexp(mant, e);
      return (rng.next() & 1u) ? -x : x;
    }
    case ValDist::Cancel: {
      // a and -(a + a*delta): the pair nearly cancels, leaving a residual
      // about 2^-20 times smaller than the operands.
      const float a = (float)(rng.normal() * 1048576.0);
      const float delta = (float)(rng.normal() * (1.0 / 1048576.0));
      *pair = -(a + a * delta);
      *has_pair = true;
      return a;
    }
    case ValDist::Grad: {
      // Gradient-like: normal * 2^-10 * 2^(g/2) with g geometric, giving a
      // heavy tail of occasional large contributions.
      const int g = draw_geometric(rng);
      const float root2 = std::sqrt(2.0f);
      const float scale = std::ldexp((g & 1) ? root2 : 1.0f, -10 + g / 2);
      return (float)rng.normal() * scale;
    }
    case ValDist::Overflow: {
      // Either a +-B pair with B in [2^126, 2^127) (three same-sign B in a row
      // overflow binary32, the exact sum of a pair is 0) or a small value.
      if (rng.below(4) == 0) return (float)rng.normal();
      const float b = std::ldexp((float)(1.0 + rng.uniform01()), 126);
      *pair = -b;
      *has_pair = true;
      return b;
    }
    case ValDist::Subnormal: {
      // [1,2) * 2^e with e in [-149, -120]: subnormals and the smallest normals.
      const float mant = (float)(1.0 + rng.uniform01());
      const int e = -149 + (int)rng.below(30);
      const float x = std::ldexp(mant, e);
      return (rng.next() & 1u) ? -x : x;
    }
  }
  return 0.0f;
}

// The binary64 counterparts, with ranges scaled to the wider format.
inline double draw_value(ValDist v, Rng& rng, double* pair, bool* has_pair) {
  *has_pair = false;
  switch (v) {
    case ValDist::Normal:
      return rng.normal();
    case ValDist::Wide: {
      // e uniform in [-100, 100]: about 60 decades.
      const double mant = 1.0 + rng.uniform01();
      const int e = (int)rng.below(201) - 100;
      const double x = std::ldexp(mant, e);
      return (rng.next() & 1u) ? -x : x;
    }
    case ValDist::Cancel: {
      // Residual about 2^-40 times smaller than the operands.
      const double a = rng.normal() * 1099511627776.0;
      const double delta = rng.normal() * (1.0 / 1099511627776.0);
      *pair = -(a + a * delta);
      *has_pair = true;
      return a;
    }
    case ValDist::Grad: {
      const int g = draw_geometric(rng);
      const double scale = std::ldexp((g & 1) ? std::sqrt(2.0) : 1.0, -10 + g / 2);
      return rng.normal() * scale;
    }
    case ValDist::Overflow: {
      if (rng.below(4) == 0) return rng.normal();
      const double b = std::ldexp(1.0 + rng.uniform01(), 1022);
      *pair = -b;
      *has_pair = true;
      return b;
    }
    case ValDist::Subnormal: {
      const double mant = 1.0 + rng.uniform01();
      const int e = -1074 + (int)rng.below(60);
      const double x = std::ldexp(mant, e);
      return (rng.next() & 1u) ? -x : x;
    }
  }
  return 0.0;
}

// Zipf-like destination sampler with exponent alpha = alpha_x2 / 2, built from
// division and sqrt only. Rank r is mapped through a fixed permutation so the
// hot destinations are scattered across the index space.
struct ZipfSampler {
  std::vector<double> cdf;
  std::vector<uint32_t> perm;
  ZipfSampler(uint32_t D, int alpha_x2, Rng& rng) : cdf(D), perm(D) {
    double acc = 0.0;
    for (uint32_t r = 1; r <= D; ++r) {
      const double x = (double)r;
      double p;
      switch (alpha_x2) {
        case 0: p = 1.0; break;
        case 1: p = std::sqrt(x); break;
        case 2: p = x; break;
        case 3: p = x * std::sqrt(x); break;
        default: p = x * x; break;
      }
      acc += 1.0 / p;
      cdf[r - 1] = acc;
    }
    for (uint32_t i = 0; i < D; ++i) perm[i] = i;
    shuffle_vec(perm, rng);
  }
  uint32_t draw(Rng& rng) const {
    const double u = rng.uniform01() * cdf.back();
    uint32_t lo = 0, hi = (uint32_t)cdf.size() - 1;
    while (lo < hi) {
      const uint32_t mid = (lo + hi) / 2;
      if (cdf[mid] >= u) hi = mid; else lo = mid + 1;
    }
    return perm[lo];
  }
};

// n is the target number of contributions (splat rounds it to whole
// footprints); D must be a perfect square for the splat pattern.
template <class T>
inline WorkloadT<T> make_workload_t(DstPattern dp, ValDist vd, uint32_t n, uint32_t D, uint64_t seed,
                                    int zipf_alpha_x2 = 2) {
  WorkloadT<T> w;
  w.name = std::string(dst_name(dp)) + "/" + val_name(vd);
  w.D = D;
  w.init.assign(D, (T)0);
  w.c.reserve(n + 64);
  Rng rng(seed);
  T pair = (T)0;
  bool has_pair = false;
  if (dp == DstPattern::Uniform) {
    while (w.c.size() < n) {
      const uint32_t d = rng.below(D);
      const T v = draw_value(vd, rng, &pair, &has_pair);
      w.c.push_back(ContributionT<T>{d, v});
      if (has_pair) w.c.push_back(ContributionT<T>{d, pair});
    }
  } else if (dp == DstPattern::Zipf) {
    ZipfSampler zs(D, zipf_alpha_x2, rng);
    while (w.c.size() < n) {
      const uint32_t d = zs.draw(rng);
      const T v = draw_value(vd, rng, &pair, &has_pair);
      w.c.push_back(ContributionT<T>{d, v});
      if (has_pair) w.c.push_back(ContributionT<T>{d, pair});
    }
  } else {
    // Each primitive deposits a 5x5 binomial footprint (weights k_i*k_j/256)
    // around a random centre of a G x G grid, as a forward splat or the
    // backward pass of an interpolation does.
    uint32_t G = 1;
    while (G * G < D) ++G;
    if (G * G != D) die("splat pattern needs a square destination count");
    static const int kern[5] = {1, 4, 6, 4, 1};
    while (w.c.size() < n) {
      const int cx = (int)rng.below(G), cy = (int)rng.below(G);
      const T v = draw_value(vd, rng, &pair, &has_pair);
      for (int rep = 0; rep < (has_pair ? 2 : 1); ++rep) {
        const T val = rep ? pair : v;
        for (int dy = -2; dy <= 2; ++dy) {
          for (int dx = -2; dx <= 2; ++dx) {
            const int x = cx + dx, y = cy + dy;
            if (x < 0 || y < 0 || x >= (int)G || y >= (int)G) continue;
            const T wgt = (T)(kern[dx + 2] * kern[dy + 2]) * ((T)1 / (T)256);
            w.c.push_back(ContributionT<T>{(uint32_t)(y * (int)G + x), val * wgt});
          }
        }
      }
    }
  }
  // Pairs are generated adjacent; shuffle so that lane order carries no
  // structure. The shuffle is part of the seeded workload definition.
  shuffle_vec(w.c, rng);
  return w;
}

inline Workload make_workload(DstPattern dp, ValDist vd, uint32_t n, uint32_t D, uint64_t seed,
                              int zipf_alpha_x2 = 2) {
  return make_workload_t<float>(dp, vd, n, D, seed, zipf_alpha_x2);
}
inline Workload64 make_workload64(DstPattern dp, ValDist vd, uint32_t n, uint32_t D, uint64_t seed,
                                  int zipf_alpha_x2 = 2) {
  return make_workload_t<double>(dp, vd, n, D, seed, zipf_alpha_x2);
}

struct CaseSpec {
  DstPattern dp;
  ValDist vd;
};

// The standard 14-case matrix.
inline std::vector<CaseSpec> standard_cases() {
  std::vector<CaseSpec> v;
  const DstPattern dps[3] = {DstPattern::Uniform, DstPattern::Zipf, DstPattern::Splat};
  const ValDist vds[4] = {ValDist::Normal, ValDist::Wide, ValDist::Cancel, ValDist::Grad};
  for (DstPattern dp : dps)
    for (ValDist vd : vds) v.push_back(CaseSpec{dp, vd});
  v.push_back(CaseSpec{DstPattern::Uniform, ValDist::Overflow});
  v.push_back(CaseSpec{DstPattern::Uniform, ValDist::Subnormal});
  return v;
}

template <class T>
inline uint64_t workload_digest(const WorkloadT<T>& w) {
  Fnv h;
  h.u32(w.D);
  for (T f : w.init) h.u64(FmtOf<T>::type::bits(f));
  for (const ContributionT<T>& x : w.c) { h.u32(x.dst); h.u64(FmtOf<T>::type::bits(x.val)); }
  return h.h;
}

}  // namespace cac
