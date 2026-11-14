// threads.h - real-concurrency backend. The same prims are executed by
// operating-system threads on std::atomic words, so the interleaving is chosen
// by the hardware and the OS scheduler rather than by a seed. This is used to
// check, outside the simulator, that (a) floating-point compare-and-swap
// accumulation really is non-reproducible and (b) the invariant schemes really
// return identical bits under genuine races.
#pragma once
#include <atomic>
#include <chrono>
#include <thread>
#include "runner.h"

namespace cac {

template <class V>
struct ThreadRun {
  double seconds = 0.0;
  uint64_t hash = 0;
  uint64_t cas_fail = 0;
  std::vector<V> out;
};

// Runs accumulator A (single-pass schemes only) with T real threads.
template <class V>
inline ThreadRun<V> run_threads(Accumulator& A, const WorkloadT<V>& w, int T) {
  if (A.passes() != 1 || !A.uses_atomics()) die("run_threads: single-pass atomic schemes only");
  A.reset(w.D);
  for (uint32_t d = 0; d < w.D; ++d) acc_deposit_initial(A, d, w.init[d], 0);
  const size_t nc = A.cells.size();
  std::unique_ptr<std::atomic<uint64_t>[]> cells(new std::atomic<uint64_t>[nc]);
  for (size_t i = 0; i < nc; ++i) cells[i].store(A.cells[i], std::memory_order_relaxed);
  std::atomic<bool> go(false);
  std::atomic<int> ready(0);
  std::vector<uint64_t> fails((size_t)T, 0);
  const size_t n = w.c.size();
  auto worker = [&](int t) {
    const size_t b = n * (size_t)t / (size_t)T, e = n * (size_t)(t + 1) / (size_t)T;
    ready.fetch_add(1);
    while (!go.load(std::memory_order_acquire)) {}
    Prim p[kMaxPrims];
    uint64_t nf = 0;
    for (size_t i = b; i < e; ++i) {
      const int np = acc_expand(A, w.c[i].dst, w.c[i].val, 0, p);
      for (int j = 0; j < np; ++j) {
        std::atomic<uint64_t>& cell = cells[p[j].cell];
        switch (p[j].op) {
          case OP_ADD_F32:
          case OP_ADD_F32_CAS: {
            uint64_t old = cell.load(std::memory_order_relaxed);
            for (;;) {
              const uint64_t nw = f2u(u2f((uint32_t)old) + u2f((uint32_t)p[j].operand));
              if (cell.compare_exchange_weak(old, nw)) break;
              ++nf;
            }
            break;
          }
          case OP_TAGADD:
          case OP_TAGADD_NATIVE: {
            const uint64_t add = mtb_pack(p[j].tag, p[j].operand);
            uint64_t old = cell.load(std::memory_order_relaxed);
            for (;;) {
              if (mtb_tag(old) > p[j].tag) break;
              if (cell.compare_exchange_weak(old, mtb_combine(old, add))) break;
              ++nf;
            }
            break;
          }
          case OP_ADD_I64:
            cell.fetch_add((uint64_t)p[j].operand);
            break;
          case OP_ORLAZY:
            if ((cell.load(std::memory_order_relaxed) & (uint64_t)p[j].operand) !=
                (uint64_t)p[j].operand)
              cell.fetch_or((uint64_t)p[j].operand);
            break;
          case OP_ADD_F64: {
            uint64_t old = cell.load(std::memory_order_relaxed);
            for (;;) {
              const uint64_t nw = d2u(u2d(old) + u2d((uint64_t)p[j].operand));
              if (cell.compare_exchange_weak(old, nw)) break;
              ++nf;
            }
            break;
          }
          case OP_MAX_I64: {
            uint64_t old = cell.load(std::memory_order_relaxed);
            while ((int64_t)old < p[j].operand &&
                   !cell.compare_exchange_weak(old, (uint64_t)p[j].operand)) {}
            break;
          }
        }
      }
    }
    fails[(size_t)t] = nf;
  };
  std::vector<std::thread> th;
  for (int t = 0; t < T; ++t) th.emplace_back(worker, t);
  while (ready.load() < T) {}
  const auto t0 = std::chrono::steady_clock::now();
  go.store(true, std::memory_order_release);
  for (std::thread& x : th) x.join();
  const auto t1 = std::chrono::steady_clock::now();
  ThreadRun<V> r;
  r.seconds = std::chrono::duration<double>(t1 - t0).count();
  for (size_t i = 0; i < nc; ++i) A.cells[i] = cells[i].load(std::memory_order_relaxed);
  r.out.resize(w.D);
  for (uint32_t d = 0; d < w.D; ++d) acc_readout(A, d, &r.out[d]);
  r.hash = output_hash(r.out);
  for (uint64_t f : fails) r.cas_fail += f;
  return r;
}

}  // namespace cac
