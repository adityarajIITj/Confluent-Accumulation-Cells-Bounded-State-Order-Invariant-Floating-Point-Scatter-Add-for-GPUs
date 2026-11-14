// accum.h - accumulation schemes expressed as sequences of atomic primitives.
//
// A scatter kernel is modelled as: every lane owns one contribution
// (destination, value) and issues a short list of atomic primitives ("prims")
// against per-destination state cells. An accumulator defines
//   * its per-destination state layout,
//   * expand(): which prims a contribution issues (per pass),
//   * readout(): how the final value is produced at quiescence,
//   * merge_from(): how two device-local states combine (multi-device runs).
// The virtual GPU (vgpu.h) decides the interleaving.
//
// Schemes are templates over the floating-point format (Fmt32 / Fmt64); the
// binary32 instantiations keep their short names (FloatAcc, CacAcc, ...).
#pragma once
#include <algorithm>
#include <cmath>
#include <memory>
#include <type_traits>
#include "exact.h"

namespace cac {

template <class T>
struct ContributionT {
  uint32_t dst;
  T val;
};
typedef ContributionT<float> Contribution;
typedef ContributionT<double> Contribution64;

enum Op : uint8_t {
  OP_ADD_F32,        // native floating-point atomic add (binary32)
  OP_ADD_F32_CAS,    // the same update written as a load + compare-and-swap loop
  OP_ADD_F64,        // native floating-point atomic add (binary64)
  OP_ADD_I64,        // native 64-bit integer atomic add (wraps)
  OP_MAX_I64,        // native 64-bit signed atomic max
  OP_ORLAZY,         // plain load; atomic OR only if the bits are not yet set
  OP_TAGADD,         // CAC cell update as a load + compare-and-swap loop
  OP_TAGADD_NATIVE   // CAC cell update as one native atomic read-modify-write
};

struct Prim {
  uint32_t cell;
  uint8_t op;
  uint8_t tag;
  int64_t operand;
};
const int kMaxPrims = 6;

// ---------------------------------------------------------------------------
// Max-tagged bin (MTB) cell algebra.
//
// A cell is one 64-bit word: [ tag : 8 | value : 56 (two's complement) ].
// Tag 0 is the empty cell and the all-zero word is the identity element.
//   (t1,v1) (+) (t2,v2) = (t1, v1+v2 mod 2^56)   if t1 == t2
//                         the operand with the larger tag otherwise.
// (+) is associative and commutative, which is what makes the final cell
// state independent of the order in which deposits commit.
// ---------------------------------------------------------------------------
const uint64_t kMask56 = 0x00FFFFFFFFFFFFFFULL;
inline uint64_t mtb_tag(uint64_t c) { return c >> 56; }
inline int64_t mtb_val(uint64_t c) {
  uint64_t v = c & kMask56;
  if (v & (1ULL << 55)) v |= ~kMask56;
  return (int64_t)v;
}
inline uint64_t mtb_pack(uint64_t tag, int64_t v) { return (tag << 56) | ((uint64_t)v & kMask56); }
inline uint64_t mtb_combine(uint64_t a, uint64_t b) {
  const uint64_t ta = mtb_tag(a), tb = mtb_tag(b);
  if (ta > tb) return a;
  if (tb > ta) return b;
  if (ta == 0) return 0;
  return (ta << 56) | (((a & kMask56) + (b & kMask56)) & kMask56);
}

// Apply the full effect of one prim atomically (sequential deposits, merges).
inline void apply_native(std::vector<uint64_t>& cells, const Prim& p) {
  uint64_t& c = cells[p.cell];
  switch (p.op) {
    case OP_ADD_F32:
    case OP_ADD_F32_CAS: c = f2u(u2f((uint32_t)c) + u2f((uint32_t)p.operand)); break;
    case OP_ADD_F64: c = d2u(u2d(c) + u2d((uint64_t)p.operand)); break;
    case OP_ADD_I64: c += (uint64_t)p.operand; break;
    case OP_MAX_I64: if ((int64_t)c < p.operand) c = (uint64_t)p.operand; break;
    case OP_ORLAZY: c |= (uint64_t)p.operand; break;
    case OP_TAGADD:
    case OP_TAGADD_NATIVE: c = mtb_combine(c, mtb_pack(p.tag, p.operand)); break;
  }
}

// Split the magnitude (m << q) into base-2^W digits and call emit(bin, digit)
// for every non-zero digit. m < 2^53, 8 <= W <= 40; the shifted value is held
// in two 64-bit halves.
template <class F>
inline void for_each_slice(uint64_t m, int q, int W, F&& emit) {
  int j = q / W;
  const int off = q % W;
  uint64_t lo = m << off;
  uint64_t hi = off ? (m >> (64 - off)) : 0;
  const uint64_t mask = (1ULL << W) - 1;
  while (lo | hi) {
    const uint64_t dgt = lo & mask;
    if (dgt) emit(j, dgt);
    lo = (lo >> W) | (hi << (64 - W));
    hi >>= W;
    ++j;
  }
}

// Round-to-nearest-even of (m << q) / 2^L as a non-negative integer. The
// caller guarantees that the result fits in 63 bits.
inline uint64_t rne_shift(uint64_t m, int q, int L) {
  if (q >= L) return m << (q - L);
  const int s = L - q;
  if (s > 54) return 0;  // m < 2^53 is below half of 2^s
  uint64_t r = m >> s;
  const uint64_t rem = m & ((1ULL << s) - 1);
  const uint64_t half = 1ULL << (s - 1);
  if (rem > half || (rem == half && (r & 1u))) ++r;
  return r;
}

const uint64_t FL_NAN = 1, FL_PINF = 2, FL_NINF = 4, FL_PZERO = 8, FL_NZERO = 16, FL_REAL = 32;

struct Accumulator {
  std::vector<uint64_t> cells;
  size_t D = 0;
  virtual ~Accumulator() = default;
  virtual std::string name() const = 0;
  virtual int passes() const { return 1; }
  virtual size_t cells_per_dest() const = 0;
  // Declared per-destination storage in bytes for a real implementation.
  virtual double bytes_per_dest() const = 0;
  virtual bool uses_atomics() const { return true; }
  virtual void reset(size_t D_) { D = D_; cells.assign(D * cells_per_dest(), 0); }

  // binary32 interface
  virtual int expand(uint32_t, float, int, Prim*) const { die("scheme has no binary32 interface"); }
  virtual float readout(uint32_t) const { die("scheme has no binary32 interface"); }
  // binary64 interface
  virtual int expand64(uint32_t, double, int, Prim*) const { die("scheme has no binary64 interface"); }
  virtual double readout64(uint32_t) const { die("scheme has no binary64 interface"); }

  // Combine another device's state into this one (after the given pass).
  virtual void merge_from(const Accumulator& o, int pass) = 0;
  // If true, deposit_first() must be used for a destination's initial value.
  virtual bool initial_is_assignment() const { return false; }
  virtual void assign_initial(uint32_t, uint64_t) {}
  // Non-atomic schemes (sort-based) implement this instead of expand().
  virtual void sorted_sum(const std::vector<Contribution>&, const std::vector<float>&,
                          std::vector<float>&) const {
    die("sorted_sum not supported");
  }
};

// Overload set used by the format-generic runner.
inline int acc_expand(const Accumulator& A, uint32_t d, float x, int pass, Prim* o) { return A.expand(d, x, pass, o); }
inline int acc_expand(const Accumulator& A, uint32_t d, double x, int pass, Prim* o) { return A.expand64(d, x, pass, o); }
inline void acc_readout(const Accumulator& A, uint32_t d, float* out) { *out = A.readout(d); }
inline void acc_readout(const Accumulator& A, uint32_t d, double* out) { *out = A.readout64(d); }

// The destination's pre-existing value is one more contribution.
template <class T>
inline void acc_deposit_initial(Accumulator& A, uint32_t dst, T x, int pass) {
  if (A.initial_is_assignment()) {
    if (pass == 0) A.assign_initial(dst, FmtOf<T>::type::bits(x));
    return;
  }
  Prim p[kMaxPrims];
  const int n = acc_expand(A, dst, x, pass, p);
  for (int i = 0; i < n; ++i) apply_native(A.cells, p[i]);
}

// Helper: route the two virtual entry points of a format-generic scheme to
// one implementation and reject calls in the wrong format.
#define CAC_FORMAT_ENTRY_POINTS(Fm)                                                         \
  int expand(uint32_t d, float x, int pass, Prim* o) const override {                      \
    if (!std::is_same<typename Fm::T, float>::value) die("binary32 call on a binary64 scheme"); \
    return expand_t(d, (typename Fm::T)x, pass, o);                                        \
  }                                                                                         \
  int expand64(uint32_t d, double x, int pass, Prim* o) const override {                   \
    if (!std::is_same<typename Fm::T, double>::value) die("binary64 call on a binary32 scheme"); \
    return expand_t(d, (typename Fm::T)x, pass, o);                                        \
  }                                                                                         \
  float readout(uint32_t d) const override {                                               \
    if (!std::is_same<typename Fm::T, float>::value) die("binary32 call on a binary64 scheme"); \
    return (float)readout_t(d);                                                            \
  }                                                                                         \
  double readout64(uint32_t d) const override {                                            \
    if (!std::is_same<typename Fm::T, double>::value) die("binary64 call on a binary32 scheme"); \
    return (double)readout_t(d);                                                           \
  }

// --- A0: status quo, floating-point atomic add in the data's own format -----
template <class Fm>
struct FloatAccT : Accumulator {
  bool cas;
  explicit FloatAccT(bool cas_loop = false) : cas(cas_loop) {
    if (cas && Fm::kBytes != 4) die("the simulated CAS-loop float add is binary32 only");
  }
  std::string name() const override { return std::string("FLOAT") + Fm::suffix() + (cas ? "-CAS" : ""); }
  size_t cells_per_dest() const override { return 1; }
  double bytes_per_dest() const override { return Fm::kBytes; }
  bool initial_is_assignment() const override { return true; }
  void assign_initial(uint32_t dst, uint64_t bits) override { cells[dst] = bits; }
  int expand_t(uint32_t dst, typename Fm::T x, int, Prim* out) const {
    const uint8_t op = (Fm::kBytes == 8) ? (uint8_t)OP_ADD_F64 : (uint8_t)(cas ? OP_ADD_F32_CAS : OP_ADD_F32);
    out[0] = Prim{dst, op, 0, (int64_t)Fm::bits(x)};
    return 1;
  }
  typename Fm::T readout_t(uint32_t dst) const { return Fm::from_bits(cells[dst]); }
  CAC_FORMAT_ENTRY_POINTS(Fm)
  void merge_from(const Accumulator& o, int) override {
    for (size_t d = 0; d < D; ++d)
      cells[d] = Fm::bits(Fm::from_bits(cells[d]) + Fm::from_bits(o.cells[d]));
  }
};
typedef FloatAccT<Fmt32> FloatAcc;
typedef FloatAccT<Fmt64> Float64Acc;

// --- A1: binary32 data accumulated in a binary64 atomic, binary32 readout ---
struct DoubleAcc : Accumulator {
  std::string name() const override { return "DOUBLE-ACC"; }
  size_t cells_per_dest() const override { return 1; }
  double bytes_per_dest() const override { return 8; }
  bool initial_is_assignment() const override { return true; }
  void assign_initial(uint32_t dst, uint64_t bits) override { cells[dst] = d2u((double)u2f((uint32_t)bits)); }
  int expand(uint32_t dst, float x, int, Prim* out) const override {
    out[0] = Prim{dst, OP_ADD_F64, 0, (int64_t)d2u((double)x)};
    return 1;
  }
  float readout(uint32_t dst) const override { return (float)u2d(cells[dst]); }
  void merge_from(const Accumulator& o, int) override {
    for (size_t d = 0; d < D; ++d) cells[d] = d2u(u2d(cells[d]) + u2d(o.cells[d]));
  }
};

// --- A4: fixed-point accumulation with an a-priori scale 2^-F ---------------
// The practice used for deterministic force accumulation in GPU molecular
// dynamics: convert to a 64-bit integer with a fixed scale and use the integer
// atomic add. Order invariant, but only inside the representable window.
template <class Fm>
struct FixedAccT : Accumulator {
  int F;
  explicit FixedAccT(int frac_bits) : F(frac_bits) {}
  std::string name() const override { return std::string("FIXED") + Fm::suffix() + "-Q" + std::to_string(F); }
  size_t cells_per_dest() const override { return 1; }
  double bytes_per_dest() const override { return 8; }
  int expand_t(uint32_t dst, typename Fm::T x, int, Prim* out) const {
    const Dec d = decode(x);
    int64_t v = 0;
    if (d.cls == FClass::NaN) {
      v = INT64_MIN;  // what a float->int conversion instruction yields
    } else if (d.cls == FClass::Inf) {
      v = d.neg ? -(INT64_C(1) << 62) : (INT64_C(1) << 62);
    } else if (d.cls == FClass::Finite) {
      const int sh = d.q - Fm::kUnitExp + F;  // x * 2^F = m * 2^sh
      uint64_t mag;
      if (sh >= 0) mag = (sh + bit_width_u64(d.m) > 62) ? (1ULL << 62) : (d.m << sh);
      else mag = rne_shift(d.m, 0, -sh);
      v = d.neg ? -(int64_t)mag : (int64_t)mag;
    }
    if (v == 0) return 0;
    out[0] = Prim{dst, OP_ADD_I64, 0, v};
    return 1;
  }
  typename Fm::T readout_t(uint32_t dst) const {
    return Fm::round(Fm::Big::from_i64_shifted((int64_t)cells[dst], (unsigned)(Fm::kUnitExp - F)));
  }
  CAC_FORMAT_ENTRY_POINTS(Fm)
  void merge_from(const Accumulator& o, int) override {
    for (size_t d = 0; d < D; ++d) cells[d] += o.cells[d];
  }
};
typedef FixedAccT<Fmt32> FixedAcc;
typedef FixedAccT<Fmt64> Fixed64Acc;

// --- A5: exact long accumulator with integer atomics ------------------------
// Carry-save variant of the superaccumulator: NB digits of W payload bits, each
// stored in a 64-bit word so that carries can be left unresolved until
// readout. Exact, hence correctly rounded and order invariant.
template <class Fm>
struct SuperAccT : Accumulator {
  int W, NB;
  explicit SuperAccT(int w = 32) : W(w), NB((Fm::kGrid + w - 1) / w) {}
  std::string name() const override { return std::string("SUPERACC") + Fm::suffix(); }
  size_t cells_per_dest() const override { return (size_t)NB + 1; }
  double bytes_per_dest() const override { return 8.0 * NB + 4; }
  int expand_t(uint32_t dst, typename Fm::T x, int, Prim* out) const {
    const Dec d = decode(x);
    const uint32_t base = dst * (uint32_t)(NB + 1);
    const uint32_t fl = base + (uint32_t)NB;
    int n = 0;
    switch (d.cls) {
      case FClass::NaN: out[n++] = Prim{fl, OP_ORLAZY, 0, (int64_t)FL_NAN}; break;
      case FClass::Inf: out[n++] = Prim{fl, OP_ORLAZY, 0, (int64_t)(d.neg ? FL_NINF : FL_PINF)}; break;
      case FClass::Zero: out[n++] = Prim{fl, OP_ORLAZY, 0, (int64_t)(d.neg ? FL_NZERO : FL_PZERO)}; break;
      case FClass::Finite:
        for_each_slice(d.m, d.q, W, [&](int j, uint64_t dg) {
          out[n++] = Prim{base + (uint32_t)j, OP_ADD_I64, 0, d.neg ? -(int64_t)dg : (int64_t)dg};
        });
        out[n++] = Prim{fl, OP_ORLAZY, 0, (int64_t)FL_REAL};
        break;
    }
    return n;
  }
  typename Fm::Big exact_sum(uint32_t dst) const {
    const size_t base = (size_t)dst * (NB + 1);
    typename Fm::Big S;
    for (int j = 0; j < NB; ++j)
      if (cells[base + j]) S.add_i64_shifted((int64_t)cells[base + j], (unsigned)(j * W));
    return S;
  }
  typename Fm::T readout_t(uint32_t dst) const {
    const uint64_t fl = cells[(size_t)dst * (NB + 1) + NB];
    if ((fl & FL_NAN) || ((fl & FL_PINF) && (fl & FL_NINF))) return Fm::nan();
    if (fl & FL_PINF) return Fm::inf(false);
    if (fl & FL_NINF) return Fm::inf(true);
    const typename Fm::Big S = exact_sum(dst);
    if (S.is_zero()) return Fm::zero((fl & FL_NZERO) && !(fl & (FL_PZERO | FL_REAL)));
    return Fm::round(S);
  }
  CAC_FORMAT_ENTRY_POINTS(Fm)
  void merge_from(const Accumulator& o, int) override {
    for (size_t d = 0; d < D; ++d) {
      const size_t base = d * (NB + 1);
      for (int j = 0; j < NB; ++j) cells[base + j] += o.cells[base + j];
      cells[base + NB] |= o.cells[base + NB];
    }
  }
};
typedef SuperAccT<Fmt32> SuperAcc;
typedef SuperAccT<Fmt64> Super64Acc;

// --- A6: two-pass exponent-aligned window (binary32) ------------------------
// Pass 0 finds the largest magnitude per destination with an integer atomic
// max. Pass 1 rounds every addend to a grid P bits below that magnitude and
// adds the integers. Order invariant, 16 bytes of state, but the kernel has to
// visit every contribution twice.
struct Window2P : Accumulator {
  int P;
  explicit Window2P(int p = 40) : P(p) {}
  std::string name() const override { return "WINDOW-2P"; }
  int passes() const override { return 2; }
  size_t cells_per_dest() const override { return 3; }  // [top+1, acc, flags]
  double bytes_per_dest() const override { return 16; }
  int grid_lsb(uint32_t dst) const {
    const int T = (int)(int64_t)cells[(size_t)dst * 3] - 1;
    return std::max(0, T - P);
  }
  int expand(uint32_t dst, float x, int pass, Prim* out) const override {
    const Dec d = decode(x);
    const uint32_t base = dst * 3u;
    if (pass == 0) {
      switch (d.cls) {
        case FClass::NaN: out[0] = Prim{base + 2, OP_ORLAZY, 0, (int64_t)FL_NAN}; return 1;
        case FClass::Inf: out[0] = Prim{base + 2, OP_ORLAZY, 0, (int64_t)(d.neg ? FL_NINF : FL_PINF)}; return 1;
        case FClass::Zero: out[0] = Prim{base + 2, OP_ORLAZY, 0, (int64_t)(d.neg ? FL_NZERO : FL_PZERO)}; return 1;
        case FClass::Finite:
          out[0] = Prim{base, OP_MAX_I64, 0, (int64_t)(d.q + bit_width_u64(d.m))};
          return 1;
      }
      return 0;
    }
    if (d.cls != FClass::Finite) return 0;
    const uint64_t mag = rne_shift(d.m, d.q, grid_lsb(dst));
    if (!mag) return 0;
    out[0] = Prim{base + 1, OP_ADD_I64, 0, d.neg ? -(int64_t)mag : (int64_t)mag};
    return 1;
  }
  float readout(uint32_t dst) const override {
    const size_t base = (size_t)dst * 3;
    const uint64_t fl = cells[base + 2];
    if ((fl & FL_NAN) || ((fl & FL_PINF) && (fl & FL_NINF))) return u2f(kCanonNaN);
    if (fl & FL_PINF) return u2f(kPosInf);
    if (fl & FL_NINF) return u2f(kNegInf);
    if (cells[base] == 0) return u2f(((fl & FL_NZERO) && !(fl & FL_PZERO)) ? 0x80000000u : 0u);
    return round_to_f32(BigInt::from_i64_shifted((int64_t)cells[base + 1], (unsigned)grid_lsb(dst)));
  }
  void merge_from(const Accumulator& o, int pass) override {
    for (size_t d = 0; d < D; ++d) {
      const size_t b = d * 3;
      if (pass == 0) {
        if ((int64_t)cells[b] < (int64_t)o.cells[b]) cells[b] = o.cells[b];
      } else {
        cells[b + 1] += o.cells[b + 1];
      }
      cells[b + 2] |= o.cells[b + 2];
    }
  }
};

// --- Proposed: Confluent Accumulation Cells ---------------------------------
// K max-tagged bin cells per destination on an absolute grid of W-bit bins.
// Real bin j uses slot j mod K and tag j + 2. Reserved tags:
//   1        zero class   (value counts +0 in the low 28 bits, -0 above)
//   NB + 2   infinities   (same split for +Inf / -Inf)
//   NB + 3   NaN
// so that special values need no extra state and stay order invariant.
template <class Fm>
struct CacAccT : Accumulator {
  int K, W, NB;
  bool native;  // issue cell updates as one native RMW instead of a CAS loop
  CacAccT(int k = 2, int w = 32, bool native_rmw = false)
      : K(k), W(w), NB((Fm::kGrid + w - 1) / w), native(native_rmw) {
    if (K < 1 || K > 8 || W < 16 || W > 40) die("CAC parameters out of range");
    if (NB + 3 > 255) die("CAC: too many bins for an 8-bit tag");
  }
  std::string name() const override {
    return std::string("CAC") + Fm::suffix() + "-" + std::to_string(K) + "x" + std::to_string(W) +
           (native ? "-NATIVE" : "");
  }
  size_t cells_per_dest() const override { return (size_t)K; }
  double bytes_per_dest() const override { return 8.0 * K; }
  uint8_t tag_zero() const { return 1; }
  uint8_t tag_inf() const { return (uint8_t)(NB + 2); }
  uint8_t tag_nan() const { return (uint8_t)(NB + 3); }
  // Bits guaranteed to be kept below the leading bit of the largest addend.
  int guaranteed_bits() const { return (K - 1) * W; }
  // Deposits per destination that can never overflow a 56-bit cell value.
  uint64_t capacity() const { return 1ULL << (55 - W); }

  int expand_t(uint32_t dst, typename Fm::T x, int, Prim* out) const {
    const Dec d = decode(x);
    const uint32_t base = dst * (uint32_t)K;
    const uint8_t op = (uint8_t)(native ? OP_TAGADD_NATIVE : OP_TAGADD);
    int n = 0;
    switch (d.cls) {
      case FClass::Zero:
        out[n++] = Prim{base, op, tag_zero(), d.neg ? (INT64_C(1) << 28) : INT64_C(1)};
        break;
      case FClass::Inf:
        out[n++] = Prim{base + (uint32_t)(NB % K), op, tag_inf(),
                        d.neg ? (INT64_C(1) << 28) : INT64_C(1)};
        break;
      case FClass::NaN:
        out[n++] = Prim{base + (uint32_t)((NB + 1) % K), op, tag_nan(), INT64_C(1)};
        break;
      case FClass::Finite:
        for_each_slice(d.m, d.q, W, [&](int j, uint64_t dg) {
          out[n++] = Prim{base + (uint32_t)(j % K), op, (uint8_t)(j + 2),
                          d.neg ? -(int64_t)dg : (int64_t)dg};
        });
        break;
    }
    return n;
  }

  struct View {
    bool nan = false, pinf = false, ninf = false, any_real = false;
    bool pzero = false, nzero = false;
    int top_bin = -1;        // highest retained real bin
    typename Fm::Big sum;    // exact value of the retained bins, in grid units
  };
  View view(uint32_t dst) const {
    View v;
    const size_t base = (size_t)dst * K;
    for (int s = 0; s < K; ++s) {
      const uint64_t c = cells[base + s];
      const int t = (int)mtb_tag(c);
      if (t == 0) continue;
      const uint64_t raw = c & kMask56;
      const uint64_t lo = raw & ((1ULL << 28) - 1), hi = raw >> 28;
      if (t == tag_nan()) v.nan = true;
      else if (t == tag_inf()) { if (lo) v.pinf = true; if (hi) v.ninf = true; }
      else if (t == tag_zero()) { if (lo) v.pzero = true; if (hi) v.nzero = true; }
      else {
        v.any_real = true;
        const int bin = t - 2;
        if (bin > v.top_bin) v.top_bin = bin;
        v.sum.add_i64_shifted(mtb_val(c), (unsigned)(bin * W));
      }
    }
    return v;
  }
  typename Fm::T readout_t(uint32_t dst) const {
    const View v = view(dst);
    if (v.nan || (v.pinf && v.ninf)) return Fm::nan();
    if (v.pinf) return Fm::inf(false);
    if (v.ninf) return Fm::inf(true);
    if (v.sum.is_zero()) return Fm::zero(!v.any_real && v.nzero && !v.pzero);
    return Fm::round(v.sum);
  }
  CAC_FORMAT_ENTRY_POINTS(Fm)
  void merge_from(const Accumulator& o, int) override {
    for (size_t i = 0; i < cells.size(); ++i) cells[i] = mtb_combine(cells[i], o.cells[i]);
  }
};
typedef CacAccT<Fmt32> CacAcc;
typedef CacAccT<Fmt64> Cac64Acc;

// --- A2 / A3: sort-based deterministic reductions (no atomics, binary32) ----
// SORT-IDX: group by destination, add in source-index order.
// SORT-VAL: group by destination, add in a canonical value order.
struct SortAcc : Accumulator {
  bool by_value;
  explicit SortAcc(bool canonical_value_order) : by_value(canonical_value_order) {}
  std::string name() const override { return by_value ? "SORT-VAL" : "SORT-IDX"; }
  size_t cells_per_dest() const override { return 0; }
  double bytes_per_dest() const override { return 0; }
  bool uses_atomics() const override { return false; }
  void merge_from(const Accumulator&, int) override {}
  void sorted_sum(const std::vector<Contribution>& c, const std::vector<float>& init,
                  std::vector<float>& out) const override {
    const size_t Dn = init.size();
    // Stable counting sort by destination keeps source-index order per group.
    std::vector<uint32_t> start(Dn + 1, 0);
    for (const Contribution& x : c) ++start[x.dst + 1];
    for (size_t d = 0; d < Dn; ++d) start[d + 1] += start[d];
    std::vector<float> vals(c.size());
    {
      std::vector<uint32_t> pos(start.begin(), start.end() - 1);
      for (const Contribution& x : c) vals[pos[x.dst]++] = x.val;
    }
    out.assign(Dn, 0.0f);
    for (size_t d = 0; d < Dn; ++d) {
      float* b = vals.data() + start[d];
      float* e = vals.data() + start[d + 1];
      if (by_value) {
        std::sort(b, e, [](float a, float bb) {
          const uint32_t ua = f2u(a), ub = f2u(bb);
          const bool na = is_nan(a), nb = is_nan(bb);
          if (na != nb) return nb;  // NaNs last
          if (na) return ua < ub;
          const int64_t oa = ordinal(a), ob = ordinal(bb);
          if (oa != ob) return oa < ob;
          return ua < ub;
        });
      }
      float acc = init[d];
      for (float* p = b; p != e; ++p) acc += *p;
      out[d] = acc;
    }
  }
};

}  // namespace cac
