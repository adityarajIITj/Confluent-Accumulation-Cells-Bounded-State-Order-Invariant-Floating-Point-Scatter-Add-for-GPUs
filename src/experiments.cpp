// experiments.cpp - experiment drivers. Each writes CSV files of raw counts;
// ratios and tables are derived from the CSVs by scripts/summarize.py so that
// every reported number can be traced to a machine-readable artifact.
#include <chrono>
#include "cac/threads.h"
#include "cac/train.h"

using namespace cac;

namespace {

const uint32_t kN = 131072;   // contributions per standard case
const uint32_t kD = 4096;     // destinations per standard case (64 x 64 grid)
const uint64_t kCaseSeed = 5000;

struct Scheme {
  std::string name;
  AccFactory make;
};

template <class A, class... Args>
Scheme scheme(Args... args) {
  AccFactory f = [args...]() { return std::unique_ptr<Accumulator>(new A(args...)); };
  return Scheme{f()->name(), f};
}

// Schemes evaluated on binary32 data.
std::vector<Scheme> standard_schemes(float*) {
  return {scheme<FloatAcc>(),        scheme<DoubleAcc>(),       scheme<SortAcc>(false),
          scheme<SortAcc>(true),     scheme<FixedAcc>(32),      scheme<Window2P>(40),
          scheme<SuperAcc>(32),      scheme<CacAcc>(2, 32),     scheme<CacAcc>(3, 32)};
}
// Schemes evaluated on binary64 data.
std::vector<Scheme> standard_schemes(double*) {
  return {scheme<Float64Acc>(),      scheme<Fixed64Acc>(48),    scheme<Super64Acc>(32),
          scheme<Cac64Acc>(2, 32),   scheme<Cac64Acc>(3, 32),   scheme<Cac64Acc>(4, 32)};
}

// 32 schedules: two sequential orders and thirty seeded interleavings.
std::vector<Schedule> standard_schedules() {
  std::vector<Schedule> s;
  s.push_back(Schedule{SchedKind::Seq, 1, 0});
  s.push_back(Schedule{SchedKind::Rev, 1, 0});
  for (uint64_t i = 0; i < 10; ++i) s.push_back(Schedule{SchedKind::Window, 4, 100 + i});
  for (uint64_t i = 0; i < 10; ++i) s.push_back(Schedule{SchedKind::Window, 64, 200 + i});
  for (uint64_t i = 0; i < 9; ++i) s.push_back(Schedule{SchedKind::Lockstep, 16, 300 + i});
  s.push_back(Schedule{SchedKind::FullRandom, 0, 400});
  return s;
}

template <class T>
WorkloadT<T> standard_workload(size_t idx) {
  const std::vector<CaseSpec> cs = standard_cases();
  return make_workload_t<T>(cs[idx].dp, cs[idx].vd, kN, kD, kCaseSeed + idx);
}

FILE* open_csv(const char* dir, const std::string& file) {
  const std::string path = std::string(dir) + "/" + file;
  FILE* f = std::fopen(path.c_str(), "w");
  if (!f) { std::fprintf(stderr, "cannot write %s (does the directory exist?)\n", path.c_str()); std::exit(2); }
  return f;
}

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <class T>
int cls_bit(T f) {
  if (is_nan(f)) return 8;
  if (is_inf(f)) return sign_bit(f) ? 4 : 2;
  return 1;
}

template <class T> const char* fmt_suffix(T*) { return FmtOf<T>::type::suffix(); }

// Right shift of a non-negative big integer (reporting helper).
template <class Big>
Big shr(const Big& a, int s) {
  Big r;
  const int NL = (int)(sizeof(a.w) / sizeof(a.w[0]));
  const int li = s / 64, lo = s % 64;
  for (int i = 0; i + li < NL; ++i) {
    uint64_t v = a.w[i + li] >> lo;
    if (lo && i + li + 1 < NL) v |= a.w[i + li + 1] << (64 - lo);
    r.w[i] = v;
  }
  return r;
}

// ---------------------------------------------------------------------------
// E1: standard matrix. For every case and scheme, run all schedules and record
// how many distinct outputs appear, how many destinations are unstable, and how
// far the results are from the exact oracle.
// ---------------------------------------------------------------------------
template <class T>
int matrix_t(const char* outdir) {
  const double t0 = now_s();
  const std::string tag = std::string("matrix") + fmt_suffix((T*)nullptr);
  FILE* f = open_csv(outdir, tag + ".csv");
  std::fprintf(f, "case,n,D,scheme,schedules,distinct_outputs,unstable_dests,max_spread_ulp,"
                  "class_flip_dests,results,exact_results,class_mismatch,max_err_ulp,sum_err_ulp,"
                  "contributions,prims,atomics,loads,cas_attempts,cas_fails,dropped,"
                  "bytes_per_dest,passes,oracle_hash,first_output_hash\n");
  const std::vector<Scheme> schemes = standard_schemes((T*)nullptr);
  const std::vector<Schedule> scheds = standard_schedules();
  const size_t ncases = standard_cases().size();
  for (size_t ci = 0; ci < ncases; ++ci) {
    const WorkloadT<T> w = standard_workload<T>(ci);
    const std::vector<T> ref = oracle(w);
    for (const Scheme& sc : schemes) {
      std::unique_ptr<Accumulator> A = sc.make();
      const size_t ns = A->uses_atomics() ? scheds.size() : 1;
      std::vector<uint64_t> hashes;
      std::vector<uint64_t> first(w.D);
      std::vector<int64_t> lo(w.D), hi(w.D);
      std::vector<uint8_t> cls(w.D, 0), unstable(w.D, 0);
      ErrStats es;
      Counters cnt;
      uint64_t first_hash = 0;
      for (size_t si = 0; si < ns; ++si) {
        RunCfg cfg;
        cfg.sched = scheds[si];
        const std::vector<T> out = run_once(*A, w, cfg, cnt);
        const uint64_t h = output_hash(out);
        if (si == 0) first_hash = h;
        if (std::find(hashes.begin(), hashes.end(), h) == hashes.end()) hashes.push_back(h);
        es.add_all(out, ref);
        for (uint32_t d = 0; d < w.D; ++d) {
          const uint64_t b = canon_bits(out[d]);
          if (si == 0) first[d] = b; else if (b != first[d]) unstable[d] = 1;
          cls[d] |= (uint8_t)cls_bit(out[d]);
          if (is_finite(out[d])) {
            const int64_t o = ordinal(out[d]);
            if (!(cls[d] & 16)) { lo[d] = hi[d] = o; cls[d] |= 16; }
            else { lo[d] = std::min(lo[d], o); hi[d] = std::max(hi[d], o); }
          }
        }
      }
      uint64_t n_unstable = 0, n_flip = 0;
      double spread = 0;
      for (uint32_t d = 0; d < w.D; ++d) {
        n_unstable += unstable[d];
        const int c = cls[d] & 15;
        if (c & (c - 1)) ++n_flip;  // more than one class seen across schedules
        if (cls[d] & 16) spread = std::max(spread, (double)hi[d] - (double)lo[d]);
      }
      std::fprintf(f, "%s,%zu,%u,%s,%zu,%zu,%llu,%.17g,%llu,%llu,%llu,%llu,%lld,%.17g,"
                      "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%.1f,%d,%016llx,%016llx\n",
                   w.name.c_str(), w.c.size(), w.D, sc.name.c_str(), ns, hashes.size(),
                   (unsigned long long)n_unstable, spread, (unsigned long long)n_flip,
                   (unsigned long long)es.n, (unsigned long long)es.exact,
                   (unsigned long long)es.mismatch, (long long)es.max_ulp, es.sum_ulp,
                   (unsigned long long)cnt.contributions, (unsigned long long)cnt.prims,
                   (unsigned long long)cnt.atomics(), (unsigned long long)cnt.load,
                   (unsigned long long)cnt.cas_attempt, (unsigned long long)cnt.cas_fail,
                   (unsigned long long)cnt.dropped, A->bytes_per_dest(), A->passes(),
                   (unsigned long long)output_hash(ref), (unsigned long long)first_hash);
      std::fflush(f);
    }
    std::printf("%s: case %2zu/%zu %-18s done (%.1fs)\n", tag.c_str(), ci + 1, ncases, w.name.c_str(),
                now_s() - t0);
    std::fflush(stdout);
  }
  std::fclose(f);
  return 0;
}

// ---------------------------------------------------------------------------
// E2: invariance gates. A scheme passes a transform on a case when the output
// bits equal its own single-device sequential output.
// ---------------------------------------------------------------------------
template <class T>
int invariance_t(const char* outdir) {
  const double t0 = now_s();
  const std::string tag = std::string("invariance") + fmt_suffix((T*)nullptr);
  FILE* f = open_csv(outdir, tag + ".csv");
  std::fprintf(f, "case,scheme,transform,identical\n");
  const std::vector<Scheme> schemes = standard_schemes((T*)nullptr);
  const size_t ncases = standard_cases().size();
  for (size_t ci = 0; ci < ncases; ++ci) {
    const WorkloadT<T> w = standard_workload<T>(ci);
    WorkloadT<T> wp1 = w, wp2 = w;
    wp1.c = permuted(w.c, 9001 + ci);
    wp2.c = permuted(w.c, 9101 + ci);
    for (const Scheme& sc : schemes) {
      std::unique_ptr<Accumulator> A = sc.make();
      Counters cnt;
      RunCfg seq;
      const std::vector<T> base = run_once(*A, w, seq, cnt);
      auto emit = [&](const std::string& tr, bool same) {
        std::fprintf(f, "%s,%s,%s,%d\n", w.name.c_str(), sc.name.c_str(), tr.c_str(), same ? 1 : 0);
      };
      RunCfg win;
      win.sched = Schedule{SchedKind::Window, 64, 77 + ci};
      RunCfg lock;
      lock.sched = Schedule{SchedKind::Lockstep, 16, 78 + ci};
      emit("schedule", same_bits(run_once(*A, w, win, cnt), base) &&
                           same_bits(run_once(*A, w, lock, cnt), base));
      emit("permutation", same_bits(run_once(*A, wp1, seq, cnt), base) &&
                              same_bits(run_once(*A, wp2, win, cnt), base));
      for (int G : {2, 4, 8})
        emit("devices" + std::to_string(G),
             same_bits(run_sharded(sc.make, w, G, 9201 + ci + (uint64_t)G, win, cnt), base));
      if (A->uses_atomics()) {
        RunCfg comb = win;
        comb.combine = true;
        emit("combining", same_bits(run_once(*A, w, comb, cnt), base));
      }
    }
    std::printf("%s: case %2zu/%zu done (%.1fs)\n", tag.c_str(), ci + 1, ncases, now_s() - t0);
    std::fflush(stdout);
  }
  std::fclose(f);
  return 0;
}

// ---------------------------------------------------------------------------
// E3: operation counts without contention (sequential schedule), so that the
// numbers are properties of the scheme rather than of an interleaving.
// ---------------------------------------------------------------------------
template <class T>
void ops_t(FILE* f, const std::vector<Scheme>& schemes) {
  for (size_t ci : {(size_t)0, (size_t)1, (size_t)3}) {  // uniform/normal, uniform/wide, uniform/grad
    const WorkloadT<T> w = standard_workload<T>(ci);
    for (const Scheme& sc : schemes) {
      std::unique_ptr<Accumulator> A = sc.make();
      Counters cnt;
      RunCfg seq;
      run_once(*A, w, seq, cnt);
      // A sort-based reduction needs the (destination, value) pairs in a
      // scratch buffer: 8 bytes per contribution.
      const int temp = A->uses_atomics() ? 0 : 8;
      std::fprintf(f, "%s,%s,%s,%llu,%llu,%llu,%llu,%llu,%.1f,%d,%d\n",
                   sizeof(T) == 4 ? "binary32" : "binary64", w.name.c_str(), sc.name.c_str(),
                   (unsigned long long)cnt.contributions, (unsigned long long)cnt.prims,
                   (unsigned long long)cnt.atomics(), (unsigned long long)cnt.load,
                   (unsigned long long)cnt.dropped, A->bytes_per_dest(), A->passes(), temp);
    }
  }
}

// D destinations, npd addends each, magnitudes [1,2) * 2^e with e uniform over
// range_bits binary orders. shape 0: random signs. shape 1: nearly cancelling
// pairs (residual about 2^-20 of the operands for binary32, 2^-40 for binary64).
template <class T>
WorkloadT<T> sens_workload(uint32_t D, uint32_t npd, int range_bits, int shape, uint64_t seed) {
  WorkloadT<T> w;
  w.D = D;
  w.init.assign(D, (T)0);
  Rng rng(seed);
  const double dscale = sizeof(T) == 4 ? (1.0 / 1048576.0) : (1.0 / 1099511627776.0);
  for (uint32_t d = 0; d < D; ++d) {
    for (uint32_t k = 0; k < npd; ++k) {
      const int e = range_bits ? (int)rng.below((uint32_t)range_bits + 1) - range_bits / 2 : 0;
      const T a = std::ldexp((T)(1.0 + rng.uniform01()), e);
      if (shape == 0) {
        w.c.push_back(ContributionT<T>{d, (rng.next() & 1u) ? -a : a});
      } else {
        const T delta = (T)(rng.normal() * dscale);
        w.c.push_back(ContributionT<T>{d, a});
        w.c.push_back(ContributionT<T>{d, -(a + a * delta)});
        ++k;
      }
    }
  }
  shuffle_vec(w.c, rng);
  return w;
}

// ---------------------------------------------------------------------------
// E5: sensitivity grid over the CAC parameters and the input statistics.
// ---------------------------------------------------------------------------
template <class Fm>
int sensitivity_t(const char* outdir, const std::vector<int>& Ks, const std::vector<int>& Ws,
                  const std::vector<int>& Rs) {
  typedef typename Fm::T T;
  typedef typename Fm::Big Big;
  const double t0 = now_s();
  const std::string tag = std::string("sensitivity") + Fm::suffix();
  FILE* f = open_csv(outdir, tag + ".csv");
  std::fprintf(f, "K,W,npd,range_bits,shape,bytes_per_dest,guaranteed_bits,capacity,invariant,"
                  "dests,exact_dests,max_err_ulp,max_bound_ratio,max_rel_to_max_addend,"
                  "simple_bound,float_exact_dests,float_max_err_ulp\n");
  const uint32_t D = 64;
  uint64_t evals = 0, violations = 0;
  for (int K : Ks) {
    for (int W : Ws) {
      for (uint32_t npd : {8u, 64u, 512u, 4096u}) {
        for (int R : Rs) {
          for (int shape = 0; shape <= 1; ++shape) {
            const uint64_t seed = 8000 + (uint64_t)npd * 131 + (uint64_t)R * 7 + (uint64_t)shape;
            const WorkloadT<T> w = sens_workload<T>(D, npd, R, shape, seed);
            const std::vector<T> ref = oracle(w);
            CacAccT<Fm> a(K, W);
            Counters cnt;
            RunCfg cfg;
            const std::vector<T> base = run_once(a, w, cfg, cnt);
            // Exact pre-rounding error against the bound of Theorem 3.
            std::vector<Big> exact(D);
            std::vector<uint64_t> nd(D, 0);
            std::vector<int> top(D, -1);
            std::vector<double> maxabs(D, 0.0);
            for (const ContributionT<T>& x : w.c) {
              const Dec dc = decode(x.val);
              exact[x.dst].add_i64_shifted(dc.neg ? -(int64_t)dc.m : (int64_t)dc.m, (unsigned)dc.q);
              ++nd[x.dst];
              top[x.dst] = std::max(top[x.dst], (dc.q + bit_width_u64(dc.m) - 1) / W);
              maxabs[x.dst] = std::max(maxabs[x.dst], std::fabs((double)x.val));
            }
            double max_ratio = 0.0, max_rel = 0.0;
            for (uint32_t d = 0; d < D; ++d) {
              const Big diff = a.view(d).sum.sub(exact[d]).abs();
              const int cut = (top[d] - K + 1) * W;
              bool ok;
              if (cut > 0) {
                ok = Big::cmp(diff, Big::from_i64_shifted((int64_t)nd[d], (unsigned)cut)) < 0;
                // diff / (n_d * 2^cut), computed after dropping the bits that
                // lie more than 60 below the cut (keeps the quotient in range).
                const int lowb = std::max(0, cut - 60);
                max_ratio = std::max(max_ratio, shr(diff, lowb).to_double_approx() /
                                                    ((double)nd[d] * std::ldexp(1.0, cut - lowb)));
              } else {
                ok = diff.is_zero();
              }
              if (!ok) { ++violations; max_ratio = 2.0; }
              // Error relative to the largest addend, from the top 60 bits of diff.
              const int hb = diff.bit_length();
              if (hb > 0) {
                const int s2 = std::max(0, hb - 60);
                const double mant = (double)diff.extract_bits((unsigned)s2, 60);
                max_rel = std::max(max_rel, std::ldexp(mant, s2 - Fm::kUnitExp) / maxabs[d]);
              }
            }
            bool inv = true;
            for (const Schedule& s : {Schedule{SchedKind::Rev, 1, 0}, Schedule{SchedKind::Window, 16, seed},
                                      Schedule{SchedKind::Lockstep, 8, seed + 1}}) {
              RunCfg c2;
              c2.sched = s;
              if (!same_bits(run_once(a, w, c2, cnt), base)) inv = false;
            }
            ErrStats es, fs;
            es.add_all(base, ref);
            FloatAccT<Fm> fl;
            fs.add_all(run_once(fl, w, cfg, cnt), ref);
            // Simple form of the bound: n * 2^-(K-1)W relative to the largest addend.
            const double simple = (double)npd * std::ldexp(1.0, -(K - 1) * W);
            std::fprintf(f, "%d,%d,%u,%d,%d,%.1f,%d,%llu,%d,%u,%llu,%lld,%.6g,%.6g,%.6g,%llu,%lld\n", K, W,
                         npd, R, shape, a.bytes_per_dest(), a.guaranteed_bits(),
                         (unsigned long long)a.capacity(), inv ? 1 : 0, D, (unsigned long long)es.exact,
                         (long long)es.max_ulp, max_ratio, max_rel, simple,
                         (unsigned long long)fs.exact, (long long)fs.max_ulp);
            ++evals;
          }
        }
      }
    }
    std::printf("%s: K=%d done (%.1fs)\n", tag.c_str(), K, now_s() - t0);
    std::fflush(stdout);
  }
  std::fclose(f);
  std::printf("%s: %llu evaluations, %llu bound violations\n", tag.c_str(), (unsigned long long)evals,
              (unsigned long long)violations);
  return violations ? 1 : 0;
}

// ---------------------------------------------------------------------------
// E7 helper: real threads for one format.
// ---------------------------------------------------------------------------
template <class T>
void threads_t(FILE* f, const std::vector<Scheme>& schemes, unsigned hc) {
  const WorkloadT<T> w = make_workload_t<T>(DstPattern::Zipf, ValDist::Grad, 1u << 22, 4096, 7777);
  for (const Scheme& sc : schemes) {
    // Reference for "matches": the scheme's own simulated sequential output.
    std::unique_ptr<Accumulator> S = sc.make();
    Counters cnt;
    RunCfg seq;
    const std::vector<T> want = run_once(*S, w, seq, cnt);
    for (unsigned Tn : {1u, hc}) {
      std::vector<uint64_t> hashes;
      for (int run = 0; run < 12; ++run) {
        std::unique_ptr<Accumulator> A = sc.make();
        const ThreadRun<T> r = run_threads(*A, w, (int)Tn);
        if (std::find(hashes.begin(), hashes.end(), r.hash) == hashes.end()) hashes.push_back(r.hash);
        std::fprintf(f, "%s,%u,%d,%.6f,%zu,%llu,%016llx,%d\n", sc.name.c_str(), Tn, run, r.seconds,
                     w.c.size(), (unsigned long long)r.cas_fail, (unsigned long long)r.hash,
                     same_bits(r.out, want) ? 1 : 0);
      }
      std::printf("threads: %-12s T=%u  distinct outputs over 12 runs: %zu\n", sc.name.c_str(), Tn,
                  hashes.size());
      std::fflush(stdout);
    }
  }
}

}  // namespace

int exp_matrix(const char* outdir) { return matrix_t<float>(outdir) | matrix_t<double>(outdir); }
int exp_invariance(const char* outdir) { return invariance_t<float>(outdir) | invariance_t<double>(outdir); }

int exp_ops(const char* outdir) {
  FILE* f = open_csv(outdir, "ops.csv");
  std::fprintf(f, "format,case,scheme,contributions,prims,atomics,loads,dropped,bytes_per_dest,passes,"
                  "temp_bytes_per_contribution\n");
  std::vector<Scheme> s32 = standard_schemes((float*)nullptr);
  s32.push_back(scheme<CacAcc>(2, 24));
  s32.push_back(scheme<CacAcc>(2, 32, true));
  ops_t<float>(f, s32);
  std::vector<Scheme> s64 = standard_schemes((double*)nullptr);
  s64.push_back(scheme<Cac64Acc>(3, 32, true));
  ops_t<double>(f, s64);
  std::fclose(f);
  std::printf("ops: done\n");
  return 0;
}

// ---------------------------------------------------------------------------
// E4: contention. Compare-and-swap retries as a function of destination skew
// and of how many warps are in flight, and the effect of warp-level combining.
// ---------------------------------------------------------------------------
int exp_contention(const char* outdir) {
  const double t0 = now_s();
  FILE* f = open_csv(outdir, "contention.csv");
  std::fprintf(f, "alpha_x2,kind,window,scheme,combine,prims_issued,atomics,cas_attempts,cas_fails,"
                  "loads,dropped,identical_to_seq\n");
  const std::vector<Scheme> schemes = {scheme<FloatAcc>(true), scheme<CacAcc>(2, 32),
                                       scheme<CacAcc>(2, 32, true)};
  for (int a2 = 0; a2 <= 4; ++a2) {
    const Workload w = make_workload(DstPattern::Zipf, ValDist::Grad, kN, kD, 7000 + (uint64_t)a2, a2);
    for (const Scheme& sc : schemes) {
      std::unique_ptr<Accumulator> A = sc.make();
      Counters c0;
      RunCfg seq;
      const std::vector<float> base = run_once(*A, w, seq, c0);
      for (SchedKind kind : {SchedKind::Window, SchedKind::Lockstep}) {
        for (uint32_t win : {1u, 8u, 64u, 512u}) {
          for (int comb = 0; comb <= 1; ++comb) {
            RunCfg cfg;
            cfg.sched = Schedule{kind, win, 600 + win + (uint64_t)a2};
            cfg.combine = comb != 0;
            Counters c;
            // Count the prims that actually reach the atomic unit.
            std::vector<Lane> lanes(w.c.size());
            uint64_t issued = 0;
            for (size_t i = 0; i < w.c.size(); ++i)
              lanes[i].n = (uint8_t)acc_expand(*A, w.c[i].dst, w.c[i].val, 0, lanes[i].p);
            if (cfg.combine) combine_warps(lanes, cfg.warp_width);
            for (const Lane& L : lanes) issued += L.n;
            const std::vector<float> out = run_once(*A, w, cfg, c);
            std::fprintf(f, "%d,%s,%u,%s,%d,%llu,%llu,%llu,%llu,%llu,%llu,%d\n", a2,
                         kind == SchedKind::Window ? "window" : "lockstep", win, sc.name.c_str(), comb,
                         (unsigned long long)issued, (unsigned long long)c.atomics(),
                         (unsigned long long)c.cas_attempt, (unsigned long long)c.cas_fail,
                         (unsigned long long)c.load, (unsigned long long)c.dropped,
                         same_bits(out, base) ? 1 : 0);
          }
        }
      }
    }
    std::printf("contention: alpha=%.1f done (%.1fs)\n", a2 / 2.0, now_s() - t0);
    std::fflush(stdout);
  }
  std::fclose(f);
  return 0;
}

int exp_sensitivity(const char* outdir) {
  int rc = sensitivity_t<Fmt32>(outdir, {1, 2, 3, 4}, {16, 24, 32}, {0, 8, 16, 32, 64, 96});
  rc |= sensitivity_t<Fmt64>(outdir, {2, 3, 4, 5}, {24, 32, 40}, {0, 16, 64, 128, 256, 512});
  return rc;
}

// ---------------------------------------------------------------------------
// E6: downstream training divergence (binary32 model).
// ---------------------------------------------------------------------------
int exp_train(const char* outdir) {
  const double t0 = now_s();
  FILE* f = open_csv(outdir, "train.csv");
  std::fprintf(f, "scheme,variation,step,loss0,frac_params_differ,max_abs_diff,max_loss_diff\n");
  TrainCfg cfg;
  const TrainData data = make_train_data(cfg);
  std::vector<Scheme> schemes;
  for (const Scheme& s : standard_schemes((float*)nullptr))
    if (s.name != "SORT-VAL" && s.name != "CAC-3x32") schemes.push_back(s);
  struct Var { const char* name; std::vector<ReplicaCfg> reps; };
  std::vector<Var> vars(3);
  vars[0].name = "schedule";
  for (uint64_t k = 0; k < 4; ++k) {
    ReplicaCfg r;
    r.sched = k == 0 ? Schedule{SchedKind::Seq, 1, 0} : Schedule{SchedKind::Window, 64, 40 + k};
    r.seed = k;
    vars[0].reps.push_back(r);
  }
  vars[1].name = "devices";
  for (int g : {1, 2, 4, 8}) {
    ReplicaCfg r;
    r.shards = g;
    r.seed = (uint64_t)g;
    vars[1].reps.push_back(r);
  }
  vars[2].name = "order";
  for (uint64_t k = 0; k < 4; ++k) {
    ReplicaCfg r;
    r.permute = k != 0;
    r.seed = 50 + k;
    vars[2].reps.push_back(r);
  }
  for (const Scheme& sc : schemes) {
    for (const Var& v : vars) {
      const std::vector<StepRecord> rec = train_compare(sc.make, cfg, data, v.reps);
      for (const StepRecord& r : rec)
        std::fprintf(f, "%s,%s,%u,%.17g,%.17g,%.17g,%.17g\n", sc.name.c_str(), v.name, r.step, r.loss0,
                     r.frac_diff, r.max_abs_diff, r.max_loss_diff);
      std::fflush(f);
    }
    std::printf("train: %-10s done (%.1fs)\n", sc.name.c_str(), now_s() - t0);
    std::fflush(stdout);
  }
  std::fclose(f);
  return 0;
}

// ---------------------------------------------------------------------------
// E7: real threads. Timings and the number of distinct floating-point outputs
// depend on the machine and the moment; only the "matches" column of the
// invariant schemes is expected to repeat.
// ---------------------------------------------------------------------------
int exp_threads(const char* outdir) {
  FILE* f = open_csv(outdir, "threads.csv");
  std::fprintf(f, "scheme,threads,run,seconds,contributions,cas_fails,output_hash,matches_sequential\n");
  unsigned hc = std::thread::hardware_concurrency();
  if (hc < 2) hc = 2;
  threads_t<float>(f, {scheme<FloatAcc>(), scheme<DoubleAcc>(), scheme<CacAcc>(2, 32),
                       scheme<CacAcc>(3, 32), scheme<SuperAcc>(32)}, hc);
  threads_t<double>(f, {scheme<Float64Acc>(), scheme<Cac64Acc>(3, 32), scheme<Super64Acc>(32)}, hc);
  std::fclose(f);
  return 0;
}

// ---------------------------------------------------------------------------
// E8: how often does a binary64 accumulator change the binary32 result when
// the order changes? This is a rare event, so it is measured with a dedicated
// Monte Carlo loop instead of the simulator: each trial sums `addends` values
// of the form +-[1,2) * 2^e, e uniform over range_bits binary orders, forward
// and backward in binary64, rounds both to binary32 and compares.
// The stream seeds are fixed, so the counts are reproducible regardless of
// how many threads execute the streams.
// ---------------------------------------------------------------------------
int exp_rare(const char* outdir) {
  const double t0 = now_s();
  FILE* f = open_csv(outdir, "rare.csv");
  std::fprintf(f, "range_bits,addends,trials,double_acc_order_dependent,forward_not_exact,"
                  "backward_not_exact,float_trials,float_acc_order_dependent\n");
  const int addends = 16, nstreams = 16;
  struct Cfg { int range; uint64_t trials_per_stream; };
  const Cfg cfgs[5] = {{0, 1ULL << 22}, {24, 1ULL << 22}, {32, 1ULL << 22}, {48, 1ULL << 24}, {80, 1ULL << 24}};
  const uint64_t float_trials = 1ULL << 20;
  unsigned hc = std::thread::hardware_concurrency();
  if (hc < 1) hc = 1;
  for (const Cfg& cf : cfgs) {
    std::vector<uint64_t> ev(nstreams, 0), fw(nstreams, 0), bw(nstreams, 0), fl(nstreams, 0);
    std::atomic<int> next(0);
    auto work = [&]() {
      for (;;) {
        const int s = next.fetch_add(1);
        if (s >= nstreams) return;
        Rng rng(0xC0FFEEULL + (uint64_t)cf.range * 1000 + (uint64_t)s);
        float x[addends];
        for (uint64_t t = 0; t < cf.trials_per_stream; ++t) {
          for (int i = 0; i < addends; ++i) {
            const uint64_t r = rng.next();
            const int e = cf.range ? (int)(((r >> 32) * (uint64_t)(cf.range + 1)) >> 32) - cf.range / 2 : 0;
            const uint32_t bits = ((uint32_t)(r >> 31) & 0x80000000u) | ((uint32_t)(127 + e) << 23) |
                                  ((uint32_t)r & 0x7fffffu);
            x[i] = u2f(bits);
          }
          double a = 0.0, b = 0.0;
          for (int i = 0; i < addends; ++i) a += (double)x[i];
          for (int i = addends - 1; i >= 0; --i) b += (double)x[i];
          const float fa = (float)a, fb = (float)b;
          if (f2u(fa) != f2u(fb)) {
            ++ev[(size_t)s];
            ExactAcc ex;
            for (int i = 0; i < addends; ++i) ex.add(x[i]);
            const float want = ex.result();
            if (f2u(fa) != f2u(want)) ++fw[(size_t)s];
            if (f2u(fb) != f2u(want)) ++bw[(size_t)s];
          }
          if (s == 0 && t < float_trials) {
            float c = 0.0f, d = 0.0f;
            for (int i = 0; i < addends; ++i) c += x[i];
            for (int i = addends - 1; i >= 0; --i) d += x[i];
            if (f2u(c) != f2u(d)) ++fl[(size_t)s];
          }
        }
      }
    };
    std::vector<std::thread> th;
    for (unsigned t = 0; t < hc; ++t) th.emplace_back(work);
    for (std::thread& t : th) t.join();
    uint64_t e = 0, a = 0, b = 0, c = 0;
    for (int s = 0; s < nstreams; ++s) { e += ev[(size_t)s]; a += fw[(size_t)s]; b += bw[(size_t)s]; c += fl[(size_t)s]; }
    std::fprintf(f, "%d,%d,%llu,%llu,%llu,%llu,%llu,%llu\n", cf.range, addends,
                 (unsigned long long)(cf.trials_per_stream * nstreams), (unsigned long long)e,
                 (unsigned long long)a, (unsigned long long)b, (unsigned long long)float_trials,
                 (unsigned long long)c);
    std::fflush(f);
    std::printf("rare: range=%2d bits  trials=%llu  order-dependent=%llu  (%.1fs)\n", cf.range,
                (unsigned long long)(cf.trials_per_stream * nstreams), (unsigned long long)e, now_s() - t0);
    std::fflush(stdout);
  }
  std::fclose(f);
  return 0;
}

// ---------------------------------------------------------------------------
// Dump workloads and results for scripts/crosscheck.py, which recomputes the
// exact sums and the CAC specification independently with Python integers.
// ---------------------------------------------------------------------------
int exp_dump(const char* outdir) {
  const std::vector<CaseSpec> cs = standard_cases();
  for (size_t ci = 0; ci < cs.size(); ++ci) {
    // Smaller instances keep the dump files and the Python run short.
    RunCfg cfg;
    cfg.sched = Schedule{SchedKind::Window, 32, 5 + ci};
    Counters cnt;
    {
      const Workload w = make_workload(cs[ci].dp, cs[ci].vd, 16384, 1024, kCaseSeed + ci);
      const std::vector<float> ref = oracle(w);
      CacAcc a2(2, 32), a3(3, 24);
      Window2P wp(40);
      const std::vector<float> o2 = run_once(a2, w, cfg, cnt);
      const std::vector<float> o3 = run_once(a3, w, cfg, cnt);
      const std::vector<float> ow = run_once(wp, w, cfg, cnt);
      const std::string base = std::string(outdir) + "/dump32_" + std::to_string(ci);
      FILE* fc = std::fopen((base + "_contrib.txt").c_str(), "w");
      FILE* fr = std::fopen((base + "_result.txt").c_str(), "w");
      if (!fc || !fr) { std::fprintf(stderr, "cannot write dump files in %s\n", outdir); return 2; }
      std::fprintf(fc, "# %s  D=%u\n", w.name.c_str(), w.D);
      for (const Contribution& x : w.c) std::fprintf(fc, "%u %08x\n", x.dst, f2u(x.val));
      std::fprintf(fr, "# dst init oracle cac2x32 cac3x24 window2p40\n");
      for (uint32_t d = 0; d < w.D; ++d)
        std::fprintf(fr, "%u %08x %08x %08x %08x %08x\n", d, f2u(w.init[d]), (uint32_t)canon_bits(ref[d]),
                     (uint32_t)canon_bits(o2[d]), (uint32_t)canon_bits(o3[d]), (uint32_t)canon_bits(ow[d]));
      std::fclose(fc);
      std::fclose(fr);
    }
    {
      const Workload64 w = make_workload64(cs[ci].dp, cs[ci].vd, 16384, 1024, kCaseSeed + ci);
      const std::vector<double> ref = oracle(w);
      Cac64Acc a3(3, 32), a4(4, 24);
      const std::vector<double> o3 = run_once(a3, w, cfg, cnt);
      const std::vector<double> o4 = run_once(a4, w, cfg, cnt);
      const std::string base = std::string(outdir) + "/dump64_" + std::to_string(ci);
      FILE* fc = std::fopen((base + "_contrib.txt").c_str(), "w");
      FILE* fr = std::fopen((base + "_result.txt").c_str(), "w");
      if (!fc || !fr) { std::fprintf(stderr, "cannot write dump files in %s\n", outdir); return 2; }
      std::fprintf(fc, "# %s  D=%u\n", w.name.c_str(), w.D);
      for (const Contribution64& x : w.c)
        std::fprintf(fc, "%u %016llx\n", x.dst, (unsigned long long)d2u(x.val));
      std::fprintf(fr, "# dst init oracle cac3x32 cac4x24\n");
      for (uint32_t d = 0; d < w.D; ++d)
        std::fprintf(fr, "%u %016llx %016llx %016llx %016llx\n", d, (unsigned long long)d2u(w.init[d]),
                     (unsigned long long)canon_bits(ref[d]), (unsigned long long)canon_bits(o3[d]),
                     (unsigned long long)canon_bits(o4[d]));
      std::fclose(fc);
      std::fclose(fr);
    }
  }
  std::printf("dump: wrote %zu binary32 and %zu binary64 cases to %s\n", cs.size(), cs.size(), outdir);
  return 0;
}
