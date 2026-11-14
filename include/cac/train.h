// train.h - downstream experiment. A matrix-factorisation model is trained by
// mini-batch SGD; the only non-sequential step is the scatter accumulation of
// per-sample gradient contributions into embedding rows, which runs on the
// virtual GPU through the accumulator under test. Everything else is plain
// sequential binary32 arithmetic, so any difference between two replicas is
// caused by the accumulation step alone.
#pragma once
#include "runner.h"

namespace cac {

struct TrainCfg {
  uint32_t users = 128, items = 512, rank = 8, batch = 512, steps = 400;
  float lr = 16.0f;  // step applied to the batch-mean gradient (per-sample step lr / batch)
  uint64_t data_seed = 2026;
};

struct TrainData {
  std::vector<uint32_t> u, i;
  std::vector<float> y;
  std::vector<float> w0;  // initial parameters, (users + items) * rank
};

inline TrainData make_train_data(const TrainCfg& cfg) {
  TrainData d;
  Rng rng(cfg.data_seed);
  const uint32_t r = cfg.rank;
  std::vector<double> ut((size_t)cfg.users * r), vt((size_t)cfg.items * r);
  for (double& x : ut) x = rng.normal() * 0.5;
  for (double& x : vt) x = rng.normal() * 0.5;
  ZipfSampler zs(cfg.items, 2, rng);
  const size_t total = (size_t)cfg.steps * cfg.batch;
  d.u.resize(total); d.i.resize(total); d.y.resize(total);
  for (size_t s = 0; s < total; ++s) {
    const uint32_t u = rng.below(cfg.users), it = zs.draw(rng);
    double dot = 0.0;
    for (uint32_t k = 0; k < r; ++k) dot += ut[(size_t)u * r + k] * vt[(size_t)it * r + k];
    d.u[s] = u; d.i[s] = it;
    d.y[s] = (float)(dot + 0.1 * rng.normal());
  }
  d.w0.resize((size_t)(cfg.users + cfg.items) * r);
  for (float& x : d.w0) x = (float)(rng.normal() * 0.1);
  return d;
}

// What differs between replicas of the same training run.
struct ReplicaCfg {
  Schedule sched;          // interleaving of the accumulation kernel
  int shards = 1;          // number of devices the batch is split over
  bool permute = false;    // shuffle the contribution list (e.g. loader order)
  uint64_t seed = 0;
};

struct StepRecord {
  uint32_t step;
  double loss0;           // batch loss of replica 0
  double frac_diff;       // max over replicas: fraction of parameters whose bits differ from replica 0
  double max_abs_diff;    // max over replicas and parameters of |w_r - w_0|
  double max_loss_diff;   // max over replicas of |loss_r - loss_0|
};

inline std::vector<StepRecord> train_compare(const AccFactory& make, const TrainCfg& cfg,
                                             const TrainData& data,
                                             const std::vector<ReplicaCfg>& reps,
                                             std::vector<std::vector<float>>* final_w = nullptr) {
  const uint32_t r = cfg.rank;
  const uint32_t D = (cfg.users + cfg.items) * r;
  const size_t R = reps.size();
  std::vector<std::vector<float>> W(R, data.w0);
  std::vector<std::unique_ptr<Accumulator>> acc;
  for (size_t k = 0; k < R; ++k) acc.push_back(make());
  std::vector<StepRecord> rec;
  const float scale = cfg.lr / (float)cfg.batch;
  Workload w;
  w.D = D;
  w.init.assign(D, 0.0f);
  std::vector<double> loss(R);
  for (uint32_t step = 0; step < cfg.steps; ++step) {
    for (size_t k = 0; k < R; ++k) {
      std::vector<float>& P = W[k];
      w.c.clear();
      double l = 0.0;
      for (uint32_t b = 0; b < cfg.batch; ++b) {
        const size_t s = (size_t)step * cfg.batch + b;
        const uint32_t u = data.u[s], it = data.i[s];
        const float* pu = &P[(size_t)u * r];
        const float* pv = &P[(size_t)(cfg.users + it) * r];
        float pred = 0.0f;
        for (uint32_t q = 0; q < r; ++q) pred += pu[q] * pv[q];
        const float e = pred - data.y[s];
        l += (double)e * (double)e;
        for (uint32_t q = 0; q < r; ++q) {
          w.c.push_back(Contribution{u * r + q, e * pv[q]});
          w.c.push_back(Contribution{(cfg.users + it) * r + q, e * pu[q]});
        }
      }
      loss[k] = l / (double)cfg.batch;
      const ReplicaCfg& rc = reps[k];
      if (rc.permute) {
        Rng prng(rc.seed * 7919ULL + step);
        shuffle_vec(w.c, prng);
      }
      RunCfg run;
      run.sched = rc.sched;
      run.sched.seed = rc.sched.seed * 104729ULL + step;
      Counters cnt;
      std::vector<float> g;
      if (rc.shards > 1) g = run_sharded(make, w, rc.shards, rc.seed * 31ULL + step, run, cnt);
      else g = run_once(*acc[k], w, run, cnt);
      for (uint32_t d = 0; d < D; ++d) P[d] = P[d] - scale * g[d];
    }
    StepRecord sr{step, loss[0], 0.0, 0.0, 0.0};
    for (size_t k = 1; k < R; ++k) {
      uint64_t nd = 0;
      double mx = 0.0;
      for (uint32_t d = 0; d < D; ++d) {
        if (canon_bits(W[k][d]) != canon_bits(W[0][d])) {
          ++nd;
          const double a = std::fabs((double)W[k][d] - (double)W[0][d]);
          if (a > mx) mx = a;
        }
      }
      sr.frac_diff = std::max(sr.frac_diff, (double)nd / (double)D);
      sr.max_abs_diff = std::max(sr.max_abs_diff, mx);
      sr.max_loss_diff = std::max(sr.max_loss_diff, std::fabs(loss[k] - loss[0]));
    }
    rec.push_back(sr);
  }
  if (final_w) *final_w = W;
  return rec;
}

}  // namespace cac
