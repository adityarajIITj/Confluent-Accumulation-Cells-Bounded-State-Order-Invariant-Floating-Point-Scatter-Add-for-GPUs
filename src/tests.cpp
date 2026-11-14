// tests.cpp - registered test suite for the CAC reference simulator.
#include <cfloat>
#include "cac/threads.h"
#include "cac/train.h"

using namespace cac;

namespace {

struct TestCase {
  const char* name;
  bool (*fn)();
};
std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}
struct Reg {
  Reg(const char* n, bool (*f)()) { registry().push_back(TestCase{n, f}); }
};

#define TEST(name)                         \
  static bool name();                      \
  static Reg reg_##name(#name, name);      \
  static bool name()
#define CHECK(c)                                                                  \
  do {                                                                            \
    if (!(c)) {                                                                   \
      std::printf("    CHECK failed: %s  (line %d)\n", #c, __LINE__);            \
      return false;                                                               \
    }                                                                             \
  } while (0)

Workload single(const std::vector<float>& vals, float init = 0.0f) {
  Workload w;
  w.name = "single";
  w.D = 1;
  w.init.assign(1, init);
  for (float v : vals) w.c.push_back(Contribution{0, v});
  return w;
}

std::vector<Schedule> schedule_family(uint64_t seed) {
  std::vector<Schedule> s;
  s.push_back(Schedule{SchedKind::Seq, 1, 0});
  s.push_back(Schedule{SchedKind::Rev, 1, 0});
  s.push_back(Schedule{SchedKind::Window, 4, seed + 1});
  s.push_back(Schedule{SchedKind::Window, 64, seed + 2});
  s.push_back(Schedule{SchedKind::Lockstep, 16, seed + 3});
  s.push_back(Schedule{SchedKind::FullRandom, 0, seed + 4});
  return s;
}

Workload64 single64(const std::vector<double>& vals, double init = 0.0) {
  Workload64 w;
  w.name = "single64";
  w.D = 1;
  w.init.assign(1, init);
  for (double v : vals) w.c.push_back(Contribution64{0, v});
  return w;
}

template <class T>
std::vector<T> run(Accumulator& A, const WorkloadT<T>& w, Schedule s = Schedule(),
                   bool combine = false, Counters* out_cnt = nullptr) {
  RunCfg cfg;
  cfg.sched = s;
  cfg.combine = combine;
  Counters cnt;
  std::vector<T> r = run_once(A, w, cfg, cnt);
  if (out_cnt) *out_cnt = cnt;
  return r;
}

// True if the accumulator returns the same bits for every schedule in the family.
template <class T>
bool invariant_over_schedules(Accumulator& A, const WorkloadT<T>& w, uint64_t seed,
                              std::vector<T>* first = nullptr) {
  const std::vector<Schedule> fam = schedule_family(seed);
  const std::vector<T> base = run(A, w, fam[0]);
  if (first) *first = base;
  for (size_t i = 1; i < fam.size(); ++i)
    if (!same_bits(run(A, w, fam[i]), base)) return false;
  return true;
}

float p2(int e) { return std::ldexp(1.0f, e); }

// ---- 01-03: number representation ------------------------------------------

TEST(test_fp32_decode_roundtrip) {
  Dec d = decode(1.0f);
  CHECK(d.cls == FClass::Finite && !d.neg && d.m == (1u << 23) && d.q == 126);
  d = decode(u2f(1u));
  CHECK(d.cls == FClass::Finite && d.m == 1 && d.q == 0);
  d = decode(FLT_MAX);
  CHECK(d.m == (1u << 24) - 1 && d.q == 253);
  CHECK(decode(-0.0f).cls == FClass::Zero && decode(-0.0f).neg);
  CHECK(decode(u2f(kPosInf)).cls == FClass::Inf && decode(u2f(kCanonNaN)).cls == FClass::NaN);
  Rng rng(11);
  for (int i = 0; i < 200000; ++i) {
    const uint32_t u = (uint32_t)rng.next();
    const float f = u2f(u);
    if (!is_finite(f)) continue;
    const float back = round_to_f32(exact_of(f), (u >> 31) != 0);
    CHECK(f2u(back) == u);
  }
  return true;
}

TEST(test_bigint_shift_add) {
  BigInt a = BigInt::from_i64_shifted(1, 100);
  a.add(BigInt::from_i64_shifted(1, 100));
  CHECK(BigInt::cmp(a, BigInt::from_i64_shifted(1, 101)) == 0);
  BigInt b = BigInt::from_i64_shifted(-1, 0);  // all ones
  b.add(BigInt::from_i64_shifted(1, 0));
  CHECK(b.is_zero());
  BigInt c;
  c.w[0] = ~0ULL;
  c.add(BigInt::from_i64_shifted(1, 0));
  CHECK(c.w[0] == 0 && c.w[1] == 1 && c.bit_length() == 65);
  const BigInt e = BigInt::from_i64_shifted(0x5A5A5A, 133);
  CHECK(e.extract_bits(133, 24) == 0x5A5A5A && e.bit_length() == 133 + 23);
  CHECK(!e.any_bits_below(134) && e.any_bits_below(135) && e.bit(134) && !e.bit(133));
  CHECK(BigInt::cmp(BigInt::from_i64_shifted(3, 0).shl(200), BigInt::from_i64_shifted(3, 200)) == 0);
  return true;
}

TEST(test_bigint_negative_values) {
  BigInt a = BigInt::from_i64_shifted(-5, 70);
  CHECK(a.is_neg());
  a.add(BigInt::from_i64_shifted(5, 70));
  CHECK(a.is_zero());
  const BigInt n = BigInt::from_i64_shifted(-12345, 211);
  CHECK(BigInt::cmp(n.negated(), BigInt::from_i64_shifted(12345, 211)) == 0);
  CHECK(BigInt::cmp(n.abs(), BigInt::from_i64_shifted(12345, 211)) == 0);
  CHECK(BigInt::cmp(n, BigInt()) < 0 && BigInt::cmp(BigInt(), n) > 0);
  CHECK(BigInt::cmp(BigInt::from_i64_shifted(-2, 64), BigInt::from_i64_shifted(-1, 64)) < 0);
  Rng rng(5);
  for (int i = 0; i < 20000; ++i) {
    const int64_t x = (int64_t)(rng.next() >> 20) - (INT64_C(1) << 43);
    const int64_t y = (int64_t)(rng.next() >> 20) - (INT64_C(1) << 43);
    const unsigned s = rng.below(250);
    BigInt u = BigInt::from_i64_shifted(x, s);
    u.add(BigInt::from_i64_shifted(y, s));
    CHECK(BigInt::cmp(u, BigInt::from_i64_shifted(x + y, s)) == 0);
    CHECK(BigInt::cmp(BigInt::from_i64_shifted(x, s).sub(BigInt::from_i64_shifted(y, s)),
                      BigInt::from_i64_shifted(x - y, s)) == 0);
  }
  return true;
}

// ---- 04-08: correctly rounded conversion -----------------------------------

TEST(test_round_exact_small) {
  for (uint32_t k = 1; k < (1u << 24); k += 977) {
    const float got = round_to_f32(BigInt::from_i64_shifted((int64_t)k, 0));
    CHECK(f2u(got) == f2u(std::ldexp((float)k, -149)));
    const float neg = round_to_f32(BigInt::from_i64_shifted(-(int64_t)k, 0));
    CHECK(f2u(neg) == (f2u(got) | 0x80000000u));
  }
  CHECK(f2u(round_to_f32(BigInt())) == 0u && f2u(round_to_f32(BigInt(), true)) == 0x80000000u);
  return true;
}

TEST(test_round_ties_to_even) {
  // 2^24 + 1 lies exactly between two floats; the even significand wins.
  CHECK(f2u(round_to_f32(BigInt::from_i64_shifted((1 << 24) + 1, 0))) == 0x01000000u);
  CHECK(f2u(round_to_f32(BigInt::from_i64_shifted((1 << 24) + 2, 0))) == 0x01000001u);
  CHECK(f2u(round_to_f32(BigInt::from_i64_shifted((1 << 24) + 3, 0))) == 0x01000002u);
  // The same pattern 40 bits higher, plus a sticky bit that breaks the tie.
  CHECK(f2u(round_to_f32(BigInt::from_i64_shifted((1 << 24) + 1, 40))) ==
        f2u(std::ldexp(16777216.0f, 40 - 149)));
  BigInt t = BigInt::from_i64_shifted((1 << 24) + 1, 40);
  t.add(BigInt::from_i64_shifted(1, 0));
  CHECK(f2u(round_to_f32(t)) == f2u(std::ldexp(16777218.0f, 40 - 149)));
  return true;
}

TEST(test_round_subnormal_boundary) {
  CHECK(f2u(round_to_f32(BigInt::from_i64_shifted((1 << 23) - 1, 0))) == 0x007fffffu);
  CHECK(f2u(round_to_f32(BigInt::from_i64_shifted(1 << 23, 0))) == 0x00800000u);
  CHECK(f2u(round_to_f32(BigInt::from_i64_shifted((1 << 24) - 1, 0))) == 0x00ffffffu);
  CHECK(f2u(round_to_f32(BigInt::from_i64_shifted(1 << 24, 0))) == 0x01000000u);
  CHECK(f2u(round_to_f32(BigInt::from_i64_shifted(1, 0))) == 0x00000001u);
  return true;
}

TEST(test_round_overflow_to_inf) {
  const BigInt fmax = BigInt::from_i64_shifted((1 << 24) - 1, 253);
  CHECK(f2u(round_to_f32(fmax)) == 0x7f7fffffu);
  // Halfway between FLT_MAX and 2^128 rounds to infinity (ties to even).
  const BigInt half = BigInt::from_i64_shifted((INT64_C(1) << 25) - 1, 252);
  CHECK(f2u(round_to_f32(half)) == kPosInf);
  CHECK(f2u(round_to_f32(half.negated())) == kNegInf);
  BigInt below = half;
  below.add(BigInt::from_i64_shifted(-1, 0));
  CHECK(f2u(round_to_f32(below)) == 0x7f7fffffu);
  CHECK(f2u(round_to_f32(BigInt::from_i64_shifted(1, 300))) == kPosInf);
  return true;
}

TEST(test_round_matches_hardware_cast) {
  // For integers below 2^53 scaled by a power of two, a double holds the value
  // exactly and the hardware double->float cast is the correctly rounded answer.
  Rng rng(77);
  for (int i = 0; i < 300000; ++i) {
    const int64_t s = (int64_t)(rng.next() >> 11);
    const int bits = 1 + (int)rng.below(53);
    const int64_t v = s >> (53 - bits);
    const int e = (int)rng.below((uint32_t)(277 - bits));  // value stays below 2^128
    const bool neg = (rng.next() & 1u) != 0;
    const float want = (float)std::ldexp(neg ? -(double)v : (double)v, e - 149);
    // An exact zero has no sign as an integer; pass the sign the hardware cast sees.
    const float got = round_to_f32(BigInt::from_i64_shifted(neg ? -v : v, (unsigned)e), neg);
    CHECK(f2u(got) == f2u(want));
  }
  return true;
}

// ---- 09-11: oracle ----------------------------------------------------------

TEST(test_oracle_matches_double_sum_when_exact) {
  // Integer-valued floats below 2^20: every partial sum is exact in a double.
  Rng rng(3);
  Workload w;
  w.D = 16;
  w.init.assign(16, 0.0f);
  std::vector<double> ref(16, 0.0);
  for (int i = 0; i < 80000; ++i) {
    const uint32_t d = rng.below(16);
    const float v = (float)((int)rng.below(1u << 21) - (1 << 20));
    w.c.push_back(Contribution{d, v});
    ref[d] += (double)v;
  }
  const std::vector<float> o = oracle(w);
  for (int d = 0; d < 16; ++d) CHECK(f2u(o[(size_t)d]) == f2u((float)ref[(size_t)d]));
  return true;
}

TEST(test_oracle_special_values) {
  const float nan = u2f(kCanonNaN), inf = u2f(kPosInf), ninf = u2f(kNegInf);
  CHECK(is_nan(oracle(single({1.0f, nan, 2.0f}))[0]));
  CHECK(f2u(oracle(single({1.0f, inf, -5.0f}))[0]) == kPosInf);
  CHECK(f2u(oracle(single({ninf, 3.0f}))[0]) == kNegInf);
  CHECK(is_nan(oracle(single({inf, 1.0f, ninf}))[0]));
  CHECK(f2u(oracle(single({FLT_MAX, FLT_MAX}))[0]) == kPosInf);
  CHECK(f2u(oracle(single({FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, 1.5f}))[0]) == f2u(1.5f));
  return true;
}

TEST(test_oracle_negative_zero) {
  CHECK(f2u(oracle(single({-0.0f, -0.0f}, -0.0f))[0]) == 0x80000000u);
  CHECK(f2u(oracle(single({-0.0f, -0.0f}, 0.0f))[0]) == 0u);
  CHECK(f2u(oracle(single({-0.0f, 0.0f}, -0.0f))[0]) == 0u);
  CHECK(f2u(oracle(single({1.0f, -1.0f}, -0.0f))[0]) == 0u);
  CHECK(f2u(oracle(single({}, -0.0f))[0]) == 0x80000000u);
  return true;
}

// ---- 12-15: the problem, the scheduler, the counters ------------------------

TEST(test_float_atomic_order_dependence_witness) {
  const Workload w = single({1.0e8f, -1.0e8f, 1.0f});
  FloatAcc f;
  const float a = run(f, w, Schedule{SchedKind::Seq, 1, 0})[0];
  const float b = run(f, w, Schedule{SchedKind::Rev, 1, 0})[0];
  CHECK(a == 1.0f && b == 0.0f);
  CHECK(oracle(w)[0] == 1.0f);
  CacAcc c(2, 32);
  SuperAcc s;
  CHECK(run(c, w, Schedule{SchedKind::Seq, 1, 0})[0] == 1.0f);
  CHECK(run(c, w, Schedule{SchedKind::Rev, 1, 0})[0] == 1.0f);
  CHECK(run(s, w, Schedule{SchedKind::Rev, 1, 0})[0] == 1.0f);
  return true;
}

TEST(test_scheduler_seed_determinism) {
  const Workload w = make_workload(DstPattern::Zipf, ValDist::Normal, 8192, 64, 9);
  for (SchedKind k : {SchedKind::Window, SchedKind::Lockstep, SchedKind::FullRandom}) {
    FloatAcc f1, f2;
    CacAcc c1(2, 32), c2(2, 32);
    Counters n1, n2;
    const Schedule s{k, 16, 4242};
    CHECK(same_bits(run(f1, w, s), run(f2, w, s)));
    CHECK(same_bits(run(c1, w, s, false, &n1), run(c2, w, s, false, &n2)));
    CHECK(n1.cas_attempt == n2.cas_attempt && n1.cas_fail == n2.cas_fail && n1.load == n2.load);
  }
  return true;
}

TEST(test_scheduler_seeds_differ) {
  const Workload w = make_workload(DstPattern::Uniform, ValDist::Normal, 8192, 16, 10);
  FloatAcc f;
  const uint64_t h1 = output_hash(run(f, w, Schedule{SchedKind::Window, 64, 1}));
  const uint64_t h2 = output_hash(run(f, w, Schedule{SchedKind::Window, 64, 2}));
  const uint64_t h3 = output_hash(run(f, w, Schedule{SchedKind::FullRandom, 0, 3}));
  CHECK(h1 != h2 && h2 != h3 && h1 != h3);
  return true;
}

TEST(test_executor_counter_accounting) {
  const Workload w = make_workload(DstPattern::Uniform, ValDist::Wide, 4096, 32, 12);
  Counters c;
  FloatAcc f;
  run(f, w, Schedule(), false, &c);
  CHECK(c.contributions == w.c.size() && c.prims == w.c.size() && c.add_f32 == w.c.size());
  CHECK(c.atomics() == w.c.size() && c.cas_attempt == 0);
  // Count non-zero 32-bit digits independently.
  uint64_t slices = 0;
  for (const Contribution& x : w.c) {
    const Dec d = decode(x.val);
    for_each_slice(d.m, d.q, 32, [&](int, uint64_t) { ++slices; });
  }
  SuperAcc s;
  run(s, w, Schedule(), false, &c);
  CHECK(c.add_i64 == slices);
  CacAcc a(2, 32);
  run(a, w, Schedule(), false, &c);
  CHECK(c.prims == slices && c.cas_fail == 0);
  CHECK(c.cas_attempt + c.dropped == slices && c.load == slices);
  CacAcc an(2, 32, true);
  run(an, w, Schedule{SchedKind::FullRandom, 0, 5}, false, &c);
  CHECK(c.tagadd_native == slices && c.cas_attempt == 0 && c.load == 0);
  return true;
}

// ---- 16-19: the cell algebra -----------------------------------------------

uint64_t random_cell(Rng& rng) {
  const uint64_t tag = rng.below(6);
  if (tag == 0) return 0;
  return mtb_pack(tag, (int64_t)(rng.next() >> 8) - (INT64_C(1) << 55));
}

TEST(test_mtb_monoid_identity) {
  Rng rng(21);
  for (int i = 0; i < 100000; ++i) {
    const uint64_t a = random_cell(rng);
    CHECK(mtb_combine(a, 0) == a && mtb_combine(0, a) == a);
  }
  CHECK(mtb_val(mtb_pack(3, -7)) == -7 && mtb_tag(mtb_pack(3, -7)) == 3);
  CHECK(mtb_val(mtb_pack(200, (INT64_C(1) << 55) - 1)) == (INT64_C(1) << 55) - 1);
  return true;
}

TEST(test_mtb_monoid_commutative) {
  Rng rng(22);
  for (int i = 0; i < 300000; ++i) {
    const uint64_t a = random_cell(rng), b = random_cell(rng);
    CHECK(mtb_combine(a, b) == mtb_combine(b, a));
  }
  return true;
}

TEST(test_mtb_monoid_associative) {
  Rng rng(23);
  for (int i = 0; i < 300000; ++i) {
    const uint64_t a = random_cell(rng), b = random_cell(rng), c = random_cell(rng);
    CHECK(mtb_combine(mtb_combine(a, b), c) == mtb_combine(a, mtb_combine(b, c)));
  }
  return true;
}

TEST(test_mtb_cas_matches_algebraic_fold) {
  // Random tagged deposits on 4 cells; whatever the interleaving, every cell
  // must end as the (+)-fold of the deposits addressed to it.
  Rng rng(24);
  for (int trial = 0; trial < 20; ++trial) {
    const uint32_t n = 3000;
    std::vector<Lane> proto(n);
    std::vector<uint64_t> want(4, 0);
    for (uint32_t i = 0; i < n; ++i) {
      const int np = 1 + (int)rng.below(3);
      proto[i].n = (uint8_t)np;
      for (int j = 0; j < np; ++j) {
        const uint32_t cell = rng.below(4);
        const uint8_t tag = (uint8_t)(1 + rng.below(5));
        const int64_t val = (int64_t)rng.below(1u << 30) - (1 << 29);
        proto[i].p[j] = Prim{cell, OP_TAGADD, tag, val};
        want[cell] = mtb_combine(want[cell], mtb_pack(tag, val));
      }
    }
    for (const Schedule& s : schedule_family(100 + (uint64_t)trial)) {
      std::vector<Lane> lanes = proto;
      std::vector<uint64_t> cells(4, 0);
      Counters c;
      run_launch(lanes, cells, s, c);
      for (int k = 0; k < 4; ++k) CHECK(cells[(size_t)k] == want[(size_t)k]);
    }
  }
  return true;
}

// ---- 20-31: Confluent Accumulation Cells -----------------------------------

const DstPattern kDp[3] = {DstPattern::Uniform, DstPattern::Zipf, DstPattern::Splat};
const ValDist kVd[4] = {ValDist::Normal, ValDist::Wide, ValDist::Cancel, ValDist::Grad};

TEST(test_cac_schedule_invariance) {
  const int cfgs[4][2] = {{2, 32}, {3, 32}, {2, 24}, {4, 16}};
  uint64_t seed = 30;
  for (DstPattern dp : kDp)
    for (ValDist vd : kVd) {
      const Workload w = make_workload(dp, vd, 6000, 64, ++seed);
      for (const auto& kw : cfgs) {
        CacAcc a(kw[0], kw[1]);
        CHECK(invariant_over_schedules(a, w, seed));
      }
    }
  return true;
}

TEST(test_cac_permutation_invariance) {
  uint64_t seed = 50;
  for (DstPattern dp : kDp)
    for (ValDist vd : kVd) {
      Workload w = make_workload(dp, vd, 6000, 64, ++seed);
      CacAcc a(2, 32);
      const std::vector<float> base = run(a, w);
      for (uint64_t p = 1; p <= 3; ++p) {
        Workload wp = w;
        wp.c = permuted(w.c, seed * 10 + p);
        CHECK(same_bits(run(a, wp, Schedule{SchedKind::Window, 8, p}), base));
      }
    }
  return true;
}

TEST(test_cac_shard_merge_invariance) {
  uint64_t seed = 70;
  for (ValDist vd : kVd) {
    const Workload w = make_workload(DstPattern::Zipf, vd, 8000, 64, ++seed);
    for (const auto& kw : std::vector<std::pair<int, int>>{{2, 32}, {3, 24}}) {
      CacAcc a(kw.first, kw.second);
      const std::vector<float> base = run(a, w);
      const AccFactory mk = [&]() { return std::unique_ptr<Accumulator>(new CacAcc(kw.first, kw.second)); };
      for (int G : {2, 3, 8}) {
        RunCfg cfg;
        cfg.sched = Schedule{SchedKind::Window, 8, (uint64_t)G};
        Counters c;
        CHECK(same_bits(run_sharded(mk, w, G, seed + (uint64_t)G, cfg, c), base));
      }
    }
  }
  return true;
}

TEST(test_cac_warp_combining_invariance) {
  const Workload w = make_workload(DstPattern::Zipf, ValDist::Grad, 16384, 64, 81);
  CacAcc a(2, 32);
  Counters plain, comb;
  const std::vector<float> base = run(a, w, Schedule{SchedKind::Seq, 1, 0}, false, &plain);
  for (const Schedule& s : schedule_family(82))
    CHECK(same_bits(run(a, w, s, true), base));
  run(a, w, Schedule{SchedKind::Seq, 1, 0}, true, &comb);
  CHECK(comb.atomics() < plain.atomics());
  return true;
}

TEST(test_cac_native_equals_cas) {
  const Workload w = make_workload(DstPattern::Splat, ValDist::Wide, 8000, 64, 83);
  CacAcc cas(2, 32, false), nat(2, 32, true);
  Counters c;
  const std::vector<float> base = run(cas, w, Schedule{SchedKind::Lockstep, 16, 1});
  CHECK(same_bits(run(nat, w, Schedule{SchedKind::FullRandom, 0, 2}, false, &c), base));
  CHECK(c.cas_attempt == 0 && c.cas_fail == 0);
  return true;
}

// Per-destination facts needed to evaluate the CAC error bound independently.
template <class Big>
struct DestFactsT {
  Big exact;
  uint64_t n = 0;
  int top_bin = -1;
  int low_bit = 1 << 30;  // lowest set bit position over all finite addends
};
typedef DestFactsT<BigInt> DestFacts;
template <class T>
std::vector<DestFactsT<typename FmtOf<T>::type::Big>> dest_facts(const WorkloadT<T>& w, int W) {
  std::vector<DestFactsT<typename FmtOf<T>::type::Big>> f(w.D);
  auto add = [&](uint32_t d, T x) {
    const Dec dc = decode(x);
    if (dc.cls != FClass::Finite) return;
    auto& t = f[d];
    ++t.n;
    t.exact.add_i64_shifted(dc.neg ? -(int64_t)dc.m : (int64_t)dc.m, (unsigned)dc.q);
    const int hi = dc.q + bit_width_u64(dc.m) - 1;
    int lo = dc.q;
    uint64_t m = dc.m;
    while (!(m & 1u)) { m >>= 1; ++lo; }
    t.top_bin = std::max(t.top_bin, hi / W);
    t.low_bit = std::min(t.low_bit, lo);
  };
  for (uint32_t d = 0; d < w.D; ++d) add(d, w.init[d]);
  for (const ContributionT<T>& x : w.c) add(x.dst, x.val);
  return f;
}

// Check Theorem 3 on every destination of a finished CAC run:
// |retained sum - exact sum| < n_d * 2^cut with cut = (b* - K + 1) * W, and
// exactly 0 when no addend has a bit below the retained window.
template <class Fm>
bool cac_bound_holds(const CacAccT<Fm>& a, const WorkloadT<typename Fm::T>& w) {
  const auto f = dest_facts(w, a.W);
  for (uint32_t d = 0; d < w.D; ++d) {
    const typename Fm::Big diff = a.view(d).sum.sub(f[d].exact).abs();
    const int cut = (f[d].top_bin - a.K + 1) * a.W;
    if (f[d].n == 0 || f[d].low_bit >= cut) {
      if (!diff.is_zero()) return false;
    } else if (Fm::Big::cmp(diff, Fm::Big::from_i64_shifted((int64_t)f[d].n, (unsigned)cut)) >= 0) {
      return false;
    }
  }
  return true;
}

TEST(test_cac_error_bound) {
  // |retained sum - exact sum| < n_d * 2^((b* - K + 1) * W), and exactly 0
  // when no addend has a bit below the retained window.
  const int cfgs[5][2] = {{1, 32}, {2, 32}, {3, 32}, {2, 24}, {3, 16}};
  uint64_t seed = 90;
  for (ValDist vd : {ValDist::Wide, ValDist::Cancel, ValDist::Grad, ValDist::Subnormal}) {
    const Workload w = make_workload(DstPattern::Zipf, vd, 8000, 64, ++seed);
    for (const auto& kw : cfgs) {
      CacAcc a(kw[0], kw[1]);
      run(a, w, Schedule{SchedKind::Window, 8, seed});
      CHECK(cac_bound_holds(a, w));
    }
  }
  return true;
}

TEST(test_cac_exact_within_window) {
  // (a) Addends confined to one bin: nothing can be lost, so CAC is exact.
  Rng rng(95);
  Workload w;
  w.D = 8;
  w.init.assign(8, 0.0f);
  for (int i = 0; i < 20000; ++i) {
    const float v = std::ldexp((float)((int)rng.below(1u << 20) + 1), 96 - 149);  // bits 96..115
    w.c.push_back(Contribution{rng.below(8), (rng.next() & 1u) ? -v : v});
  }
  CacAcc a(2, 32);
  CHECK(same_bits(run(a, w, Schedule{SchedKind::FullRandom, 0, 1}), oracle(w)));
  // (b) With K >= number of bins CAC keeps every bin and is an exact
  // superaccumulator, whatever the dynamic range.
  uint64_t seed = 96;
  for (ValDist vd : {ValDist::Wide, ValDist::Cancel, ValDist::Overflow, ValDist::Subnormal}) {
    const Workload x = make_workload(DstPattern::Uniform, vd, 4000, 32, ++seed);
    CacAcc full(7, 40);
    CHECK(full.K >= full.NB);
    std::vector<float> first;
    CHECK(invariant_over_schedules(full, x, seed, &first));
    CHECK(same_bits(first, oracle(x)));
  }
  return true;
}

TEST(test_cac_guaranteed_precision) {
  // Worst alignment for K=2, W=32: the largest addend's leading bit is the
  // lowest bit of bin 4 (position 128). Bin 3 (positions 96..127) is still
  // kept, position 95 is not.
  const float big = p2(128 - 149), kept = p2(96 - 149), lost = p2(95 - 149);
  CacAcc a(2, 32);
  CHECK(a.guaranteed_bits() == 32);
  const Workload w1 = single({big, kept, kept, -kept});
  run(a, w1);
  CHECK(BigInt::cmp(a.view(0).sum, dest_facts(w1, 32)[0].exact) == 0);
  for (const Schedule& s : schedule_family(97)) {
    const Workload w2 = single({lost, big, kept, lost});
    run(a, w2, s);
    // Exactly the two `lost` addends are missing.
    BigInt want = exact_of(big);
    want.add(exact_of(kept));
    CHECK(BigInt::cmp(a.view(0).sum, want) == 0);
  }
  // K=3 keeps one more bin, so the same input is exact.
  CacAcc b(3, 32);
  const Workload w3 = single({lost, big, kept, lost});
  run(b, w3);
  CHECK(BigInt::cmp(b.view(0).sum, dest_facts(w3, 32)[0].exact) == 0);
  return true;
}

Workload special_cases() {
  const float nan = u2f(kCanonNaN), inf = u2f(kPosInf), ninf = u2f(kNegInf);
  struct Row { float init; std::vector<float> v; };
  const std::vector<Row> rows = {
      {0.0f, {nan, 1.0f}},
      {0.0f, {inf, 1.0f}},
      {0.0f, {ninf, -1.0e30f}},
      {0.0f, {inf, ninf}},
      {0.0f, {inf, inf, 5.0f}},
      {-0.0f, {-0.0f, -0.0f}},
      {-0.0f, {-0.0f, 0.0f}},
      {0.0f, {-0.0f, -0.0f}},
      {-0.0f, {1.0f, -1.0f}},
      {-0.0f, {}},
      {0.0f, {nan, inf, ninf}},
      {0.0f, {u2f(1u), u2f(3u), u2f(0x80000002u)}},
      {1.5f, {2.5f, -4.0f}},
      {0.0f, {FLT_MAX, FLT_MAX}},
      {0.0f, {-FLT_MAX, -FLT_MAX, 1.0f}},
      {u2f(kPosInf), {1.0f}},
  };
  Workload w;
  w.D = (uint32_t)rows.size();
  for (uint32_t d = 0; d < w.D; ++d) {
    w.init.push_back(rows[d].init);
    for (float v : rows[d].v) w.c.push_back(Contribution{d, v});
  }
  return w;
}

TEST(test_cac_special_values) {
  const Workload w = special_cases();
  const std::vector<float> ref = oracle(w);
  const int cfgs[4][2] = {{1, 32}, {2, 32}, {3, 24}, {4, 16}};
  for (const auto& kw : cfgs) {
    CacAcc a(kw[0], kw[1]);
    for (const Schedule& s : schedule_family(98)) CHECK(same_bits(run(a, w, s), ref));
    CacAcc n(kw[0], kw[1], true);
    CHECK(same_bits(run(n, w, Schedule{SchedKind::FullRandom, 0, 3}), ref));
  }
  return true;
}

TEST(test_cac_intermediate_overflow_immunity) {
  // The exact sum is 1.5. Floating-point accumulation overflows on the way in
  // most orders and can never recover.
  const Workload w = single({3.0e38f, 3.0e38f, -3.0e38f, -3.0e38f, 1.5f});
  FloatAcc f;
  CHECK(f2u(run(f, w, Schedule{SchedKind::Seq, 1, 0})[0]) == kPosInf);
  CHECK(oracle(w)[0] == 1.5f);
  SuperAcc s;
  CHECK(run(s, w)[0] == 1.5f);
  // CAC-2x32 keeps 32 bits below the 3e38 addends: the result is finite and
  // within its bound, but 1.5 is 127 binary orders below and is not retained.
  CacAcc a(2, 32);
  for (const Schedule& sc : schedule_family(99)) CHECK(run(a, w, sc)[0] == 0.0f);
  // A cell set that spans the whole grid returns the exact answer.
  CacAcc full(7, 40);
  CHECK(run(full, w, Schedule{SchedKind::Rev, 1, 0})[0] == 1.5f);
  return true;
}

uint64_t max_per_dest(const Workload& w) {
  std::vector<uint64_t> n(w.D, 1);
  for (const Contribution& x : w.c) ++n[x.dst];
  return *std::max_element(n.begin(), n.end());
}

TEST(test_cac_capacity_guard) {
  // W=40 leaves 15 bits of headroom: 2^15 deposits per destination are safe.
  CacAcc a(7, 40);
  CHECK(a.capacity() == (1ULL << 15));
  CHECK(CacAcc(2, 32).capacity() == (1ULL << 23) && CacAcc(2, 24).capacity() == (1ULL << 31));
  // A value whose digit nearly fills a 40-bit bin, deposited 2^16 times,
  // overflows the 56-bit cell value. The result is still order invariant but
  // no longer the exact sum - which is why capacity is a stated precondition.
  const float v = std::ldexp(16777215.0f, 16 + 80 - 149);  // digit (2^24-1) << 16 in bin 2
  std::vector<float> many((size_t)1 << 16, v);
  const Workload w = single(many);
  CHECK(max_per_dest(w) > a.capacity());
  std::vector<float> first;
  CHECK(invariant_over_schedules(a, w, 7, &first));
  CHECK(!same_bits(first, oracle(w)));
  // Within capacity the same value is summed exactly.
  std::vector<float> few((size_t)1 << 14, v);
  const Workload w2 = single(few);
  CHECK(max_per_dest(w2) <= a.capacity());
  CHECK(same_bits(run(a, w2), oracle(w2)));
  return true;
}

// ---- 32-39: baselines --------------------------------------------------------

TEST(test_superacc_exact_parity) {
  uint64_t seed = 110;
  for (const CaseSpec& cs : standard_cases()) {
    const Workload w = make_workload(cs.dp, cs.vd, 5000, 64, ++seed);
    SuperAcc s;
    std::vector<float> first;
    CHECK(invariant_over_schedules(s, w, seed, &first));
    CHECK(same_bits(first, oracle(w)));
  }
  CHECK(same_bits(run(*std::unique_ptr<SuperAcc>(new SuperAcc(24)), special_cases()),
                  oracle(special_cases())));
  return true;
}

TEST(test_superacc_transform_invariance) {
  const Workload w = make_workload(DstPattern::Zipf, ValDist::Cancel, 8000, 64, 130);
  SuperAcc s;
  const std::vector<float> base = run(s, w);
  Workload wp = w;
  wp.c = permuted(w.c, 131);
  CHECK(same_bits(run(s, wp, Schedule{SchedKind::Window, 8, 1}), base));
  CHECK(same_bits(run(s, w, Schedule{SchedKind::Lockstep, 8, 2}, true), base));
  const AccFactory mk = []() { return std::unique_ptr<Accumulator>(new SuperAcc()); };
  RunCfg cfg;
  Counters c;
  CHECK(same_bits(run_sharded(mk, w, 4, 132, cfg, c), base));
  return true;
}

TEST(test_window2p_invariance_and_bound) {
  uint64_t seed = 140;
  for (ValDist vd : {ValDist::Normal, ValDist::Wide, ValDist::Cancel, ValDist::Subnormal}) {
    const Workload w = make_workload(DstPattern::Zipf, vd, 8000, 64, ++seed);
    Window2P a(40);
    std::vector<float> base;
    CHECK(invariant_over_schedules(a, w, seed, &base));
    Workload wp = w;
    wp.c = permuted(w.c, seed + 1);
    CHECK(same_bits(run(a, wp, Schedule{SchedKind::Window, 8, 1}), base));
    CHECK(same_bits(run(a, w, Schedule{SchedKind::Window, 8, 2}, true), base));
    const AccFactory mk = []() { return std::unique_ptr<Accumulator>(new Window2P(40)); };
    RunCfg cfg;
    Counters c;
    CHECK(same_bits(run_sharded(mk, w, 4, seed + 2, cfg, c), base));
    // Each addend is rounded to the grid: |error| <= n_d * 2^(L-1).
    run(a, w);
    const std::vector<DestFacts> f = dest_facts(w, 32);
    for (uint32_t d = 0; d < w.D; ++d) {
      const int L = a.grid_lsb(d);
      const BigInt got = BigInt::from_i64_shifted((int64_t)a.cells[(size_t)d * 3 + 1], (unsigned)L);
      const BigInt twice = got.sub(f[d].exact).abs().shl(1);
      CHECK(BigInt::cmp(twice, BigInt::from_i64_shifted((int64_t)f[d].n, (unsigned)L)) <= 0);
    }
  }
  CHECK(same_bits(run(*std::unique_ptr<Window2P>(new Window2P(40)), special_cases()),
                  oracle(special_cases())));
  return true;
}

TEST(test_fixed_point_invariance_and_range_failure) {
  const Workload w = make_workload(DstPattern::Uniform, ValDist::Normal, 8000, 64, 150);
  FixedAcc a(32);
  CHECK(invariant_over_schedules(a, w, 151));
  // Inside its window the scheme is accurate to the grid.
  ErrStats es;
  es.add_all(run(a, w), oracle(w));
  CHECK(es.mismatch == 0 && es.max_ulp <= 4096);
  // Outside the window it is silently wrong: small values vanish, large ones saturate.
  CHECK(run(a, single({1.0e-20f, 1.0e-20f}))[0] == 0.0f);
  CHECK(oracle(single({1.0e-20f, 1.0e-20f}))[0] == 2.0e-20f);
  const float big = run(a, single({1.0e30f}))[0];
  CHECK(big != 1.0e30f);
  return true;
}

TEST(test_double_accumulator_not_invariant) {
  // 1 + 2^-24 is a rounding tie for binary32. Four addends of 2^-54 decide it,
  // but a binary64 accumulator only sees them if they arrive before the 1.
  const Workload w = single({1.0f, p2(-24), p2(-54), p2(-54), p2(-54), p2(-54)});
  DoubleAcc a;
  const float fwd = run(a, w, Schedule{SchedKind::Seq, 1, 0})[0];
  const float rev = run(a, w, Schedule{SchedKind::Rev, 1, 0})[0];
  CHECK(f2u(fwd) == 0x3f800000u && f2u(rev) == 0x3f800001u);
  CHECK(f2u(oracle(w)[0]) == 0x3f800001u);
  CacAcc c(3, 32);
  CHECK(f2u(run(c, w, Schedule{SchedKind::Seq, 1, 0})[0]) == 0x3f800001u);
  return true;
}

TEST(test_sort_idx_permutation_dependent) {
  SortAcc s(false);
  Workload a = single({1.0e8f, -1.0e8f, 1.0f});
  Workload b = single({1.0f, -1.0e8f, 1.0e8f});
  CHECK(run(s, a)[0] == 1.0f && run(s, b)[0] == 0.0f);
  // No atomics, so the schedule cannot matter.
  CHECK(run(s, a, Schedule{SchedKind::FullRandom, 0, 9})[0] == 1.0f);
  return true;
}

TEST(test_sort_val_shard_dependent) {
  const Workload w = make_workload(DstPattern::Uniform, ValDist::Normal, 8192, 16, 160);
  SortAcc s(true);
  const std::vector<float> base = run(s, w);
  Workload wp = w;
  wp.c = permuted(w.c, 161);
  CHECK(same_bits(run(s, wp), base));  // canonical order: permutation invariant
  const AccFactory mk = []() { return std::unique_ptr<Accumulator>(new SortAcc(true)); };
  RunCfg cfg;
  Counters c;
  CHECK(!same_bits(run_sharded(mk, w, 2, 162, cfg, c), base));  // partial sums are not
  return true;
}

TEST(test_float_combining_changes_bits) {
  const Workload w = make_workload(DstPattern::Zipf, ValDist::Normal, 8192, 16, 170);
  FloatAcc f;
  const Schedule s{SchedKind::Seq, 1, 0};
  CHECK(!same_bits(run(f, w, s, false), run(f, w, s, true)));
  return true;
}

// ---- 40: frozen workload digests (cross-platform determinism) ----------------

const uint64_t kFrozenDigest[14] = {
    0x840151b213c4eb8bULL, 0xf1e338cfa73f00a6ULL, 0xc7c5f15e32a793e4ULL, 0x2bdebe3da8021d20ULL,
    0x50b46a8061c9cc8aULL, 0x4c500773f494d334ULL, 0x5b1abeaa24906abaULL, 0x5b4158704bd3c540ULL,
    0x8a6f69aa43535c0eULL, 0xa58a3d947ca5543bULL, 0x540f739aff66fe63ULL, 0x5baa399d1e2edd19ULL,
    0x0bfd90a836c4306bULL, 0x724f4c5d602c1577ULL,
};
const uint64_t kFrozenDigest64[14] = {
    0xf4a0a2ff47598fa5ULL, 0x3c5d6dba041b8addULL, 0x2f92668f9059a39eULL, 0xbf1d52b5de76004fULL,
    0xfbe175e3fbd73c53ULL, 0x7cd9fc2a23dce5b8ULL, 0xf7ac4443db0dee81ULL, 0x52f1091d10157f84ULL,
    0x9d62752531bde3f0ULL, 0x34eae4af2ccb0142ULL, 0xb5d15b11ca786aebULL, 0x8004160203b9b623ULL,
    0xb0fee318ebf7f481ULL, 0x0c70715a2b5046caULL,
};

TEST(test_workload_digests_frozen) {
  const std::vector<CaseSpec> cs = standard_cases();
  bool all_zero = true;
  for (uint64_t d : kFrozenDigest) if (d) all_zero = false;
  for (size_t i = 0; i < cs.size(); ++i) {
    const Workload w = make_workload(cs[i].dp, cs[i].vd, 4096, 256, 1000 + i);
    const uint64_t dg = workload_digest(w);
    if (all_zero) std::printf("    digest[%zu] = 0x%016llxULL  (%s)\n", i, (unsigned long long)dg, w.name.c_str());
    else CHECK(dg == kFrozenDigest[i]);
  }
  return !all_zero;
}

TEST(test_workload64_digests_frozen) {
  const std::vector<CaseSpec> cs = standard_cases();
  bool all_zero = true;
  for (uint64_t d : kFrozenDigest64) if (d) all_zero = false;
  for (size_t i = 0; i < cs.size(); ++i) {
    const Workload64 w = make_workload64(cs[i].dp, cs[i].vd, 4096, 256, 1000 + i);
    const uint64_t dg = workload_digest(w);
    if (all_zero) std::printf("    digest64[%zu] = 0x%016llxULL  (%s)\n", i, (unsigned long long)dg, w.name.c_str());
    else CHECK(dg == kFrozenDigest64[i]);
  }
  return !all_zero;
}

// ---- 41-42: downstream training ---------------------------------------------

std::vector<ReplicaCfg> replica_set() {
  std::vector<ReplicaCfg> r(4);
  r[0].sched = Schedule{SchedKind::Seq, 1, 0};
  r[1].sched = Schedule{SchedKind::Window, 64, 11};
  r[1].seed = 1;
  r[2].sched = Schedule{SchedKind::Window, 16, 12};
  r[2].shards = 2;
  r[2].seed = 2;
  r[3].sched = Schedule{SchedKind::Lockstep, 8, 13};
  r[3].permute = true;
  r[3].seed = 3;
  return r;
}

double mean_loss(const std::vector<StepRecord>& rec, size_t from, size_t count) {
  double m = 0.0;
  for (size_t i = from; i < from + count; ++i) m += rec[i].loss0;
  return m / (double)count;
}

TEST(test_training_cac_replicas_identical) {
  TrainCfg cfg;
  cfg.users = 32; cfg.items = 64; cfg.rank = 4; cfg.batch = 128; cfg.steps = 120; cfg.lr = 4.0f;
  const TrainData data = make_train_data(cfg);
  const AccFactory mk = []() { return std::unique_ptr<Accumulator>(new CacAcc(2, 32)); };
  std::vector<std::vector<float>> fw;
  const std::vector<StepRecord> rec = train_compare(mk, cfg, data, replica_set(), &fw);
  for (const StepRecord& s : rec) CHECK(s.frac_diff == 0.0 && s.max_loss_diff == 0.0);
  for (size_t k = 1; k < fw.size(); ++k) CHECK(same_bits(fw[k], fw[0]));
  // The model is actually learning: the last 20 steps beat the first 20.
  CHECK(mean_loss(rec, rec.size() - 20, 20) < 0.8 * mean_loss(rec, 0, 20));
  return true;
}

TEST(test_training_float_replicas_diverge) {
  TrainCfg cfg;
  cfg.users = 32; cfg.items = 64; cfg.rank = 4; cfg.batch = 128; cfg.steps = 120; cfg.lr = 4.0f;
  const TrainData data = make_train_data(cfg);
  const AccFactory mk = []() { return std::unique_ptr<Accumulator>(new FloatAcc()); };
  const std::vector<StepRecord> rec = train_compare(mk, cfg, data, replica_set());
  CHECK(rec.back().frac_diff > 0.0 && rec.back().max_abs_diff > 0.0);
  return true;
}

// ---- 43: real threads --------------------------------------------------------

TEST(test_real_threads_invariant_schemes) {
  const Workload w = make_workload(DstPattern::Zipf, ValDist::Grad, 200000, 256, 180);
  const std::vector<float> ref = oracle(w);
  CacAcc sim(2, 32);
  const std::vector<float> want = run(sim, w);
  for (int rep = 0; rep < 3; ++rep) {
    CacAcc a(2, 32);
    CHECK(same_bits(run_threads(a, w, 2 + rep).out, want));
    SuperAcc s;
    CHECK(same_bits(run_threads(s, w, 2 + rep).out, ref));
  }
  return true;
}

// ---- binary64 ------------------------------------------------------------------

TEST(test_fp64_decode_roundtrip) {
  Dec d = decode(1.0);
  CHECK(d.cls == FClass::Finite && !d.neg && d.m == (1ULL << 52) && d.q == 1022);
  d = decode(u2d(1ULL));
  CHECK(d.cls == FClass::Finite && d.m == 1 && d.q == 0);
  d = decode(DBL_MAX);
  CHECK(d.m == (1ULL << 53) - 1 && d.q == 2045);
  CHECK(decode(-0.0).cls == FClass::Zero && decode(-0.0).neg);
  CHECK(decode(u2d(kPosInf64)).cls == FClass::Inf && decode(u2d(kCanonNaN64)).cls == FClass::NaN);
  Rng rng(301);
  for (int i = 0; i < 100000; ++i) {
    const uint64_t u = rng.next();
    const double f = u2d(u);
    if (!is_finite(f)) continue;
    CHECK(d2u(round_to_f64(exact_of(f), (u >> 63) != 0)) == u);
  }
  // Boundaries: largest subnormal, smallest normal, overflow threshold.
  CHECK(d2u(round_to_f64(BigInt64::from_i64_shifted((INT64_C(1) << 52) - 1, 0))) == 0x000fffffffffffffULL);
  CHECK(d2u(round_to_f64(BigInt64::from_i64_shifted(INT64_C(1) << 52, 0))) == 0x0010000000000000ULL);
  CHECK(d2u(round_to_f64(BigInt64::from_i64_shifted((INT64_C(1) << 53) - 1, 2045))) == d2u(DBL_MAX));
  CHECK(d2u(round_to_f64(BigInt64::from_i64_shifted((INT64_C(1) << 54) - 1, 2044))) == kPosInf64);
  CHECK(d2u(round_to_f64(BigInt64::from_i64_shifted((INT64_C(1) << 53) + 1, 7))) ==
        d2u(std::ldexp(9007199254740992.0, 7 - 1074)));
  return true;
}

TEST(test_round64_matches_hardware_cast) {
  // int64 -> double conversion is correctly rounded by the hardware, and
  // scaling a normal result by a power of two is exact.
  Rng rng(302);
  for (int i = 0; i < 200000; ++i) {
    const int bits = 1 + (int)rng.below(62);
    const int64_t v = (int64_t)(rng.next() >> (64 - bits));
    const int e = (int)rng.below((uint32_t)(2098 - bits));
    const bool neg = (rng.next() & 1u) != 0;
    const double want = std::ldexp(neg ? -(double)v : (double)v, e - 1074);
    const double got = round_to_f64(BigInt64::from_i64_shifted(neg ? -v : v, (unsigned)e), neg);
    CHECK(d2u(got) == d2u(want));
  }
  return true;
}

TEST(test_oracle64_exact_and_special) {
  Rng rng(303);
  Workload64 w;
  w.D = 8;
  w.init.assign(8, 0.0);
  std::vector<double> ref(8, 0.0);
  for (int i = 0; i < 60000; ++i) {
    const uint32_t d = rng.below(8);
    const double v = (double)((int64_t)(rng.next() >> 34) - (INT64_C(1) << 29));  // |v| < 2^29
    w.c.push_back(Contribution64{d, v});
    ref[d] += v;  // exact: partial sums stay below 2^53
  }
  const std::vector<double> o = oracle(w);
  for (int d = 0; d < 8; ++d) CHECK(d2u(o[(size_t)d]) == d2u(ref[(size_t)d]));
  const double nan = u2d(kCanonNaN64), inf = u2d(kPosInf64), ninf = u2d(kNegInf64);
  CHECK(is_nan(oracle(single64({1.0, nan}))[0]) && is_nan(oracle(single64({inf, ninf}))[0]));
  CHECK(d2u(oracle(single64({inf, -1.0}))[0]) == kPosInf64);
  CHECK(d2u(oracle(single64({DBL_MAX, DBL_MAX}))[0]) == kPosInf64);
  CHECK(oracle(single64({DBL_MAX, DBL_MAX, -DBL_MAX, -DBL_MAX, 1.5}))[0] == 1.5);
  CHECK(d2u(oracle(single64({-0.0}, -0.0))[0]) == 0x8000000000000000ULL);
  CHECK(d2u(oracle(single64({1.0, -1.0}, -0.0))[0]) == 0);
  return true;
}

TEST(test_double_atomic_order_dependence_witness) {
  const Workload64 w = single64({1.0e16, -1.0e16, 1.0});
  Float64Acc f;
  CHECK(run(f, w, Schedule{SchedKind::Seq, 1, 0})[0] == 1.0);
  CHECK(run(f, w, Schedule{SchedKind::Rev, 1, 0})[0] == 0.0);
  CHECK(oracle(w)[0] == 1.0);
  Cac64Acc c(3, 32);
  Super64Acc s;
  CHECK(run(c, w, Schedule{SchedKind::Seq, 1, 0})[0] == 1.0);
  CHECK(run(c, w, Schedule{SchedKind::Rev, 1, 0})[0] == 1.0);
  CHECK(run(s, w, Schedule{SchedKind::Rev, 1, 0})[0] == 1.0);
  return true;
}

TEST(test_cac64_invariance) {
  const int cfgs[4][2] = {{3, 32}, {4, 32}, {2, 32}, {4, 24}};
  uint64_t seed = 310;
  for (ValDist vd : kVd) {
    const Workload64 w = make_workload64(DstPattern::Zipf, vd, 6000, 64, ++seed);
    for (const auto& kw : cfgs) {
      Cac64Acc a(kw[0], kw[1]);
      std::vector<double> base;
      CHECK(invariant_over_schedules(a, w, seed, &base));
      Workload64 wp = w;
      wp.c = permuted(w.c, seed + 1);
      CHECK(same_bits(run(a, wp, Schedule{SchedKind::Window, 8, 1}), base));
      CHECK(same_bits(run(a, w, Schedule{SchedKind::Lockstep, 8, 2}, true), base));
      const AccFactory mk = [&]() { return std::unique_ptr<Accumulator>(new Cac64Acc(kw[0], kw[1])); };
      for (int G : {2, 8}) {
        RunCfg cfg;
        cfg.sched = Schedule{SchedKind::Window, 8, (uint64_t)G};
        Counters c;
        CHECK(same_bits(run_sharded(mk, w, G, seed + (uint64_t)G, cfg, c), base));
      }
    }
  }
  return true;
}

TEST(test_cac64_error_bound) {
  const int cfgs[4][2] = {{2, 32}, {3, 32}, {4, 32}, {3, 24}};
  uint64_t seed = 320;
  for (ValDist vd : {ValDist::Wide, ValDist::Cancel, ValDist::Grad, ValDist::Subnormal}) {
    const Workload64 w = make_workload64(DstPattern::Zipf, vd, 6000, 64, ++seed);
    for (const auto& kw : cfgs) {
      Cac64Acc a(kw[0], kw[1]);
      run(a, w, Schedule{SchedKind::Window, 8, seed});
      CHECK(cac_bound_holds(a, w));
    }
  }
  return true;
}

TEST(test_cac64_window_must_cover_significand) {
  // 1 + 2^-52 has bits at grid positions 1074 and 1022. With 32-bit bins the
  // leading bit is in bin 33; K=2 keeps bins 33 and 32 (positions >= 1024) and
  // loses the low bit, K=3 keeps it. A binary64 cell set needs (K-1)*W >= 52.
  const double x = 1.0 + std::ldexp(1.0, -52);
  const Workload64 w = single64({x});
  Cac64Acc k2(2, 32), k3(3, 32);
  CHECK(k2.guaranteed_bits() < 52 && k3.guaranteed_bits() >= 52);
  CHECK(run(k2, w)[0] == 1.0);
  CHECK(d2u(run(k3, w)[0]) == d2u(x));
  // With a sufficient window every single addend is returned unchanged.
  Rng rng(330);
  Workload64 many;
  many.D = 2000;
  many.init.assign(many.D, 0.0);
  for (uint32_t d = 0; d < many.D; ++d) {
    double v;
    do { v = u2d(rng.next()); } while (!is_finite(v));
    many.c.push_back(Contribution64{d, v});
  }
  std::vector<double> out = run(k3, many, Schedule{SchedKind::FullRandom, 0, 1});
  for (uint32_t d = 0; d < many.D; ++d) {
    // -0 + (+0 initial value) is +0, everything else must come back bit-exact.
    const double want = (many.c[d].val == 0.0) ? 0.0 : many.c[d].val;
    CHECK(d2u(out[d]) == d2u(want));
  }
  return true;
}

Workload64 special_cases64() {
  const double nan = u2d(kCanonNaN64), inf = u2d(kPosInf64), ninf = u2d(kNegInf64);
  struct Row { double init; std::vector<double> v; };
  const std::vector<Row> rows = {
      {0.0, {nan, 1.0}},
      {0.0, {inf, 1.0}},
      {0.0, {ninf, -1.0e300}},
      {0.0, {inf, ninf}},
      {-0.0, {-0.0, -0.0}},
      {-0.0, {-0.0, 0.0}},
      {-0.0, {1.0, -1.0}},
      {-0.0, {}},
      {0.0, {u2d(1ULL), u2d(3ULL), u2d(0x8000000000000002ULL)}},
      {1.5, {2.5, -4.0}},
      {0.0, {DBL_MAX, DBL_MAX}},
      {0.0, {-DBL_MAX, -DBL_MAX, 1.0}},
      {u2d(kPosInf64), {1.0}},
  };
  Workload64 w;
  w.D = (uint32_t)rows.size();
  for (uint32_t d = 0; d < w.D; ++d) {
    w.init.push_back(rows[d].init);
    for (double v : rows[d].v) w.c.push_back(Contribution64{d, v});
  }
  return w;
}

TEST(test_cac64_special_values) {
  const Workload64 w = special_cases64();
  const std::vector<double> ref = oracle(w);
  const int cfgs[4][2] = {{1, 32}, {3, 32}, {4, 32}, {4, 24}};
  for (const auto& kw : cfgs) {
    Cac64Acc a(kw[0], kw[1]);
    for (const Schedule& s : schedule_family(340)) CHECK(same_bits(run(a, w, s), ref));
  }
  Super64Acc sa;
  CHECK(same_bits(run(sa, w, Schedule{SchedKind::FullRandom, 0, 2}), ref));
  return true;
}

TEST(test_superacc64_exact_parity) {
  uint64_t seed = 350;
  for (const CaseSpec& cs : standard_cases()) {
    const Workload64 w = make_workload64(cs.dp, cs.vd, 3000, 64, ++seed);
    Super64Acc s;
    std::vector<double> first;
    CHECK(invariant_over_schedules(s, w, seed, &first));
    CHECK(same_bits(first, oracle(w)));
  }
  return true;
}

TEST(test_fixed64_invariance_and_range_failure) {
  const Workload64 w = make_workload64(DstPattern::Uniform, ValDist::Normal, 6000, 64, 360);
  Fixed64Acc a(48);
  CHECK(invariant_over_schedules(a, w, 361));
  CHECK(run(a, single64({1.0e-20, 1.0e-20}))[0] == 0.0);       // below the grid
  CHECK(run(a, single64({1.0e6}))[0] != 1.0e6);                 // above the window
  CHECK(run(a, single64({0.5, 0.25}))[0] == 0.75);              // inside the window
  return true;
}

TEST(test_real_threads_binary64) {
  const Workload64 w = make_workload64(DstPattern::Zipf, ValDist::Grad, 150000, 256, 370);
  const std::vector<double> ref = oracle(w);
  Cac64Acc sim(3, 32);
  const std::vector<double> want = run(sim, w);
  for (int rep = 0; rep < 2; ++rep) {
    Cac64Acc a(3, 32);
    CHECK(same_bits(run_threads(a, w, 2 + rep).out, want));
    Super64Acc s;
    CHECK(same_bits(run_threads(s, w, 2 + rep).out, ref));
  }
  return true;
}

// ---- reduced standard matrix ---------------------------------------------------

TEST(test_matrix_smoke) {
  uint64_t seed = 200;
  for (const CaseSpec& cs : standard_cases()) {
    const Workload w = make_workload(cs.dp, cs.vd, 4096, 256, ++seed);
    const std::vector<float> ref = oracle(w);
    CacAcc a2(2, 32), a3(3, 32);
    Window2P wp(40);
    SuperAcc sa;
    std::vector<float> o2, o3, ow, os;
    CHECK(invariant_over_schedules(a2, w, seed, &o2));
    CHECK(invariant_over_schedules(a3, w, seed, &o3));
    CHECK(invariant_over_schedules(wp, w, seed, &ow));
    CHECK(invariant_over_schedules(sa, w, seed, &os));
    CHECK(same_bits(os, ref));
    ErrStats e2, e3;
    e2.add_all(o2, ref);
    e3.add_all(o3, ref);
    CHECK(e2.mismatch == 0 && e3.mismatch == 0);
  }
  return true;
}

}  // namespace

int run_all_tests() {
  int pass = 0, idx = 0;
  const int total = (int)registry().size();
  for (const TestCase& t : registry()) {
    ++idx;
    const bool ok = t.fn();
    std::printf("[%02d/%02d] %-48s %s\n", idx, total, t.name, ok ? "PASS" : "FAIL");
    std::fflush(stdout);
    if (ok) ++pass;
  }
  std::printf("\n%d/%d registered tests passed\n", pass, total);
  return pass == total ? 0 : 1;
}
