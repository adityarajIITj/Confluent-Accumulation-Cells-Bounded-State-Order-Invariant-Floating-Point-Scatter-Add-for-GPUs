// runner.h - drives an accumulator over a workload on the virtual GPU and
// provides the oracle and comparison helpers used by tests and experiments.
// Everything is generic over the value type (float or double).
#pragma once
#include <functional>
#include "vgpu.h"
#include "workload.h"

namespace cac {

struct RunCfg {
  Schedule sched;
  bool combine = false;
  uint32_t warp_width = 32;
};

typedef std::function<std::unique_ptr<Accumulator>()> AccFactory;

template <class T>
inline void run_pass(Accumulator& A, const std::vector<ContributionT<T>>& c, int pass,
                     const RunCfg& cfg, Counters& cnt) {
  std::vector<Lane> lanes(c.size());
  for (size_t i = 0; i < c.size(); ++i) {
    lanes[i].n = (uint8_t)acc_expand(A, c[i].dst, c[i].val, pass, lanes[i].p);
    cnt.prims += lanes[i].n;
  }
  if (pass == 0) cnt.contributions += c.size();
  if (cfg.combine) combine_warps(lanes, cfg.warp_width);
  Schedule s = cfg.sched;
  s.seed = s.seed * 1000003ULL + (uint64_t)pass;
  run_launch(lanes, A.cells, s, cnt, cfg.warp_width);
}

inline void sorted_dispatch(const Accumulator& A, const std::vector<Contribution>& c,
                            const std::vector<float>& init, std::vector<float>& out) {
  A.sorted_sum(c, init, out);
}
inline void sorted_dispatch(const Accumulator&, const std::vector<Contribution64>&,
                            const std::vector<double>&, std::vector<double>&) {
  die("sort-based schemes are binary32 only");
}

// Single-device run: returns the final destination values.
template <class T>
inline std::vector<T> run_once(Accumulator& A, const WorkloadT<T>& w, const RunCfg& cfg,
                               Counters& cnt) {
  std::vector<T> out(w.D);
  if (!A.uses_atomics()) {
    sorted_dispatch(A, w.c, w.init, out);
    cnt.contributions += w.c.size();
    return out;
  }
  A.reset(w.D);
  for (int pass = 0; pass < A.passes(); ++pass) {
    for (uint32_t d = 0; d < w.D; ++d) acc_deposit_initial(A, d, w.init[d], pass);
    run_pass(A, w.c, pass, cfg, cnt);
  }
  for (uint32_t d = 0; d < w.D; ++d) acc_readout(A, d, &out[d]);
  return out;
}

// Multi-device run: the contributions are split at random over G devices,
// each device accumulates into its own state, and the states are merged.
// Multi-pass schemes exchange their pass results between passes.
template <class T>
inline std::vector<T> run_sharded(const AccFactory& make, const WorkloadT<T>& w, int G,
                                  uint64_t shard_seed, const RunCfg& cfg, Counters& cnt) {
  std::vector<std::vector<ContributionT<T>>> shard((size_t)G);
  Rng rng(shard_seed);
  for (const ContributionT<T>& x : w.c) shard[rng.below((uint32_t)G)].push_back(x);
  std::vector<T> out(w.D);
  std::unique_ptr<Accumulator> probe = make();
  if (!probe->uses_atomics()) {
    std::vector<T> zeros(w.D, (T)0), part;
    for (int g = 0; g < G; ++g) {
      sorted_dispatch(*probe, shard[(size_t)g], g == 0 ? w.init : zeros, part);
      if (g == 0) out = part;
      else for (uint32_t d = 0; d < w.D; ++d) out[d] = out[d] + part[d];
    }
    cnt.contributions += w.c.size();
    return out;
  }
  std::vector<std::unique_ptr<Accumulator>> dev;
  for (int g = 0; g < G; ++g) { dev.push_back(make()); dev.back()->reset(w.D); }
  const int np = dev[0]->passes();
  for (int pass = 0; pass < np; ++pass) {
    for (uint32_t d = 0; d < w.D; ++d) acc_deposit_initial(*dev[0], d, w.init[d], pass);
    for (int g = 0; g < G; ++g) {
      RunCfg c2 = cfg;
      c2.sched.seed = cfg.sched.seed * 131ULL + (uint64_t)g + 1;
      run_pass(*dev[(size_t)g], shard[(size_t)g], pass, c2, cnt);
    }
    for (int g = 1; g < G; ++g) dev[0]->merge_from(*dev[(size_t)g], pass);
    if (pass + 1 < np)
      for (int g = 1; g < G; ++g) dev[(size_t)g]->cells = dev[0]->cells;
  }
  for (uint32_t d = 0; d < w.D; ++d) acc_readout(*dev[0], d, &out[d]);
  return out;
}

// Exact reference for every destination.
template <class T>
inline std::vector<T> oracle(const WorkloadT<T>& w) {
  std::vector<ExactAccT<typename FmtOf<T>::type>> acc(w.D);
  for (uint32_t d = 0; d < w.D; ++d) acc[d].add(w.init[d]);
  for (const ContributionT<T>& x : w.c) acc[x.dst].add(x.val);
  std::vector<T> out(w.D);
  for (uint32_t d = 0; d < w.D; ++d) out[d] = acc[d].result();
  return out;
}

template <class T>
inline uint64_t output_hash(const std::vector<T>& v) {
  Fnv h;
  for (T f : v) h.u64(canon_bits(f));
  return h.h;
}

template <class T>
inline bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) if (canon_bits(a[i]) != canon_bits(b[i])) return false;
  return true;
}

// Distance between a result and the oracle, in representable values of the
// format. Returns -1 when exactly one of the two is non-finite or their
// non-finite classes differ ("class mismatch": e.g. Inf where the exact sum is
// finite).
template <class T>
inline int64_t ulp_distance(T r, T ref) {
  const bool fr = is_finite(r), ff = is_finite(ref);
  if (fr && ff) {
    const int64_t a = ordinal(r), b = ordinal(ref);
    if ((a < 0) == (b < 0)) return a > b ? a - b : b - a;
    // Opposite signs: |a| + |b| can exceed int64 for binary64, so saturate.
    const uint64_t s = (uint64_t)(a < 0 ? -a : a) + (uint64_t)(b < 0 ? -b : b);
    return s > (uint64_t)INT64_MAX ? INT64_MAX : (int64_t)s;
  }
  if (!fr && !ff && canon_bits(r) == canon_bits(ref)) return 0;
  return -1;
}

struct ErrStats {
  uint64_t n = 0, exact = 0, mismatch = 0;
  int64_t max_ulp = 0;
  double sum_ulp = 0;
  template <class T>
  void add(T r, T ref) {
    ++n;
    const int64_t d = ulp_distance(r, ref);
    if (d < 0) { ++mismatch; return; }
    if (d == 0) ++exact;
    if (d > max_ulp) max_ulp = d;
    sum_ulp += (double)d;
  }
  template <class T>
  void add_all(const std::vector<T>& r, const std::vector<T>& ref) {
    for (size_t i = 0; i < r.size(); ++i) add(r[i], ref[i]);
  }
};

template <class T>
inline std::vector<ContributionT<T>> permuted(const std::vector<ContributionT<T>>& c, uint64_t seed) {
  std::vector<ContributionT<T>> p = c;
  Rng rng(seed);
  shuffle_vec(p, rng);
  return p;
}

}  // namespace cac
