// vgpu.h - the virtual GPU: lanes, warps, an atomic unit with counters, and a
// seeded scheduler that chooses the interleaving of atomic micro-operations.
//
// Model. A launch has one lane per contribution; consecutive lanes form warps
// of warp_width lanes. A lane executes its prims in program order; a prim is
// one or more micro-operations, each of which touches shared memory at most
// once and is therefore atomic:
//   native atomics (add/max/or/tagged-add) ......... 1 micro-op
//   OP_ORLAZY ........ plain load, then atomic OR only if needed
//   CAS-loop prims ... plain load, then compare-and-swap attempts; a failed
//                      attempt returns the current word, as atomicCAS does
// The schedule decides which lane performs its next micro-op. Any schedule is
// a legal GPU execution: atomics guarantee that every update is applied, not
// the order in which updates are applied.
#pragma once
#include "accum.h"

namespace cac {

struct Counters {
  uint64_t contributions = 0, prims = 0, microsteps = 0;
  uint64_t add_f32 = 0, add_f64 = 0, add_i64 = 0, max_i64 = 0, or64 = 0, tagadd_native = 0;
  uint64_t cas_attempt = 0, cas_fail = 0, load = 0, dropped = 0;
  // Atomic memory transactions issued to the atomic unit.
  uint64_t atomics() const {
    return add_f32 + add_f64 + add_i64 + max_i64 + or64 + tagadd_native + cas_attempt;
  }
  void add(const Counters& o) {
    contributions += o.contributions; prims += o.prims; microsteps += o.microsteps;
    add_f32 += o.add_f32; add_f64 += o.add_f64; add_i64 += o.add_i64; max_i64 += o.max_i64;
    or64 += o.or64; tagadd_native += o.tagadd_native;
    cas_attempt += o.cas_attempt; cas_fail += o.cas_fail; load += o.load; dropped += o.dropped;
  }
};

enum class SchedKind : uint8_t {
  Seq,        // lanes run to completion in index order
  Rev,        // lanes run to completion in reverse index order
  Window,     // `window` warps in flight; a random in-flight lane steps
  Lockstep,   // `window` warps in flight; a random warp steps all its lanes (SIMT)
  FullRandom  // every lane in flight; a random lane steps
};

struct Schedule {
  SchedKind kind = SchedKind::Seq;
  uint32_t window = 1;
  uint64_t seed = 0;
  std::string label() const {
    switch (kind) {
      case SchedKind::Seq: return "seq";
      case SchedKind::Rev: return "rev";
      case SchedKind::Window: return "window" + std::to_string(window);
      case SchedKind::Lockstep: return "lockstep" + std::to_string(window);
      case SchedKind::FullRandom: return "fullrandom";
    }
    return "?";
  }
};

struct Lane {
  Prim p[kMaxPrims];
  uint8_t n = 0, i = 0, phase = 0;
  uint64_t assumed = 0;
};

// Execute one micro-operation of a lane. Returns true when the lane is done.
inline bool lane_step(Lane& L, std::vector<uint64_t>& cells, Counters& c) {
  const Prim& pr = L.p[L.i];
  uint64_t& cell = cells[pr.cell];
  bool done_prim = true;
  ++c.microsteps;
  switch (pr.op) {
    case OP_ADD_F32:
      cell = f2u(u2f((uint32_t)cell) + u2f((uint32_t)pr.operand));
      ++c.add_f32;
      break;
    case OP_ADD_F64:
      cell = d2u(u2d(cell) + u2d((uint64_t)pr.operand));
      ++c.add_f64;
      break;
    case OP_ADD_I64:
      cell += (uint64_t)pr.operand;
      ++c.add_i64;
      break;
    case OP_MAX_I64:
      if ((int64_t)cell < pr.operand) cell = (uint64_t)pr.operand;
      ++c.max_i64;
      break;
    case OP_TAGADD_NATIVE:
      cell = mtb_combine(cell, mtb_pack(pr.tag, pr.operand));
      ++c.tagadd_native;
      break;
    case OP_ORLAZY:
      if (L.phase == 0) {
        ++c.load;
        if ((cell & (uint64_t)pr.operand) != (uint64_t)pr.operand) { L.phase = 1; done_prim = false; }
      } else {
        cell |= (uint64_t)pr.operand;
        ++c.or64;
      }
      break;
    case OP_ADD_F32_CAS:
      if (L.phase == 0) {
        L.assumed = cell; ++c.load; L.phase = 1; done_prim = false;
      } else {
        ++c.cas_attempt;
        if (cell == L.assumed) {
          cell = f2u(u2f((uint32_t)L.assumed) + u2f((uint32_t)pr.operand));
        } else {
          ++c.cas_fail; L.assumed = cell; done_prim = false;
        }
      }
      break;
    case OP_TAGADD:
      if (L.phase == 0) {
        L.assumed = cell; ++c.load; L.phase = 1;
        // Tags only grow, so a deposit below the observed tag can never be
        // retained: it is dropped without issuing an atomic.
        if (mtb_tag(L.assumed) > pr.tag) ++c.dropped; else done_prim = false;
      } else {
        ++c.cas_attempt;
        if (cell == L.assumed) {
          cell = mtb_combine(L.assumed, mtb_pack(pr.tag, pr.operand));
        } else {
          ++c.cas_fail; L.assumed = cell;
          if (mtb_tag(L.assumed) > pr.tag) ++c.dropped; else done_prim = false;
        }
      }
      break;
  }
  if (done_prim) { ++L.i; L.phase = 0; }
  return L.i >= L.n;
}

// Run one kernel launch under the given schedule.
inline void run_launch(std::vector<Lane>& lanes, std::vector<uint64_t>& cells, const Schedule& s,
                       Counters& c, uint32_t warp_width = 32) {
  const uint32_t n = (uint32_t)lanes.size();
  Rng rng(s.seed * 0x9E3779B97F4A7C15ULL + 0x1234567ULL);
  switch (s.kind) {
    case SchedKind::Seq:
      for (uint32_t i = 0; i < n; ++i)
        if (lanes[i].n) while (!lane_step(lanes[i], cells, c)) {}
      break;
    case SchedKind::Rev:
      for (uint32_t i = n; i-- > 0;)
        if (lanes[i].n) while (!lane_step(lanes[i], cells, c)) {}
      break;
    case SchedKind::FullRandom: {
      std::vector<uint32_t> act;
      act.reserve(n);
      for (uint32_t i = 0; i < n; ++i) if (lanes[i].n) act.push_back(i);
      while (!act.empty()) {
        const uint32_t k = rng.below((uint32_t)act.size());
        if (lane_step(lanes[act[k]], cells, c)) { act[k] = act.back(); act.pop_back(); }
      }
      break;
    }
    case SchedKind::Window: {
      const uint32_t nw = (n + warp_width - 1) / warp_width;
      std::vector<uint32_t> order(nw);
      for (uint32_t i = 0; i < nw; ++i) order[i] = i;
      shuffle_vec(order, rng);
      std::vector<uint32_t> remaining(nw, 0), act;
      uint32_t next = 0, in_flight = 0;
      auto admit = [&]() -> bool {
        while (next < nw) {
          const uint32_t w = order[next++];
          const uint32_t b = w * warp_width, e = std::min(n, b + warp_width);
          uint32_t cnt = 0;
          for (uint32_t i = b; i < e; ++i) if (lanes[i].n) { act.push_back(i); ++cnt; }
          remaining[w] = cnt;
          if (cnt) return true;
        }
        return false;
      };
      while (in_flight < s.window && admit()) ++in_flight;
      while (!act.empty()) {
        const uint32_t k = rng.below((uint32_t)act.size());
        const uint32_t id = act[k];
        if (lane_step(lanes[id], cells, c)) {
          act[k] = act.back(); act.pop_back();
          if (--remaining[id / warp_width] == 0) { --in_flight; if (admit()) ++in_flight; }
        }
      }
      break;
    }
    case SchedKind::Lockstep: {
      const uint32_t nw = (n + warp_width - 1) / warp_width;
      std::vector<uint32_t> order(nw);
      for (uint32_t i = 0; i < nw; ++i) order[i] = i;
      shuffle_vec(order, rng);
      std::vector<uint32_t> actw;
      uint32_t next = 0;
      auto warp_live = [&](uint32_t w) {
        const uint32_t b = w * warp_width, e = std::min(n, b + warp_width);
        for (uint32_t i = b; i < e; ++i) if (lanes[i].i < lanes[i].n) return true;
        return false;
      };
      auto admit = [&]() -> bool {
        while (next < nw) {
          const uint32_t w = order[next++];
          if (warp_live(w)) { actw.push_back(w); return true; }
        }
        return false;
      };
      while (actw.size() < s.window && admit()) {}
      while (!actw.empty()) {
        const uint32_t k = rng.below((uint32_t)actw.size());
        const uint32_t w = actw[k];
        const uint32_t b = w * warp_width, e = std::min(n, b + warp_width);
        bool live = false;
        // One instruction for the whole warp; same-address atomics of a warp
        // are serialised in lane order.
        for (uint32_t i = b; i < e; ++i) {
          if (lanes[i].i < lanes[i].n) {
            if (!lane_step(lanes[i], cells, c)) live = true;
          }
        }
        if (!live) { actw[k] = actw.back(); actw.pop_back(); admit(); }
      }
      break;
    }
  }
}

// Warp-level combining: prims of one warp that address the same cell with the
// same operation are merged before they reach the atomic unit. For integer
// and tagged updates this is an application of the same associative,
// commutative operation and cannot change the final state. For floating-point
// adds it is a different (still legal) rounding order.
inline void combine_warps(std::vector<Lane>& lanes, uint32_t warp_width = 32) {
  const uint32_t n = (uint32_t)lanes.size();
  struct Owner { uint32_t cell; uint8_t opclass; uint32_t lane; uint8_t idx; };
  std::vector<Owner> owners;
  for (uint32_t b = 0; b < n; b += warp_width) {
    const uint32_t e = std::min(n, b + warp_width);
    owners.clear();
    for (uint32_t i = b; i < e; ++i) {
      Lane& L = lanes[i];
      uint8_t keep = 0;
      for (uint8_t j = 0; j < L.n; ++j) {
        Prim p = L.p[j];
        const uint8_t oc = (p.op == OP_TAGADD_NATIVE) ? (uint8_t)OP_TAGADD : p.op;
        Owner* own = nullptr;
        for (Owner& o : owners) if (o.cell == p.cell && o.opclass == oc) { own = &o; break; }
        if (!own) {
          L.p[keep] = p;
          owners.push_back(Owner{p.cell, oc, i, keep});
          ++keep;
          continue;
        }
        Prim& t = lanes[own->lane].p[own->idx];
        switch (p.op) {
          case OP_ADD_I64: t.operand = (int64_t)((uint64_t)t.operand + (uint64_t)p.operand); break;
          case OP_MAX_I64: if (t.operand < p.operand) t.operand = p.operand; break;
          case OP_ORLAZY: t.operand = (int64_t)((uint64_t)t.operand | (uint64_t)p.operand); break;
          case OP_TAGADD:
          case OP_TAGADD_NATIVE: {
            const uint64_t m = mtb_combine(mtb_pack(t.tag, t.operand), mtb_pack(p.tag, p.operand));
            t.tag = (uint8_t)mtb_tag(m);
            t.operand = mtb_val(m);
            break;
          }
          case OP_ADD_F32:
          case OP_ADD_F32_CAS:
            t.operand = (int64_t)f2u(u2f((uint32_t)t.operand) + u2f((uint32_t)p.operand));
            break;
          case OP_ADD_F64:
            t.operand = (int64_t)d2u(u2d((uint64_t)t.operand) + u2d((uint64_t)p.operand));
            break;
        }
      }
      L.n = keep;
    }
  }
}

}  // namespace cac
