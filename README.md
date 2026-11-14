# Confluent Accumulation Cells

**Bounded-state, order-invariant floating-point scatter-add for GPUs**

Aditya Raj, School of Artificial Intelligence and Data Science, IIT Jodhpur

Floating-point scatter-add (`atomicAdd`, `index_add`, `scatter_add`, embedding-gradient
accumulation) returns different bits from run to run on a GPU, because atomics fix *that*
every update is applied but not the *order*, and floating-point addition is not associative.

Confluent Accumulation Cells (CAC) give every destination **K 64-bit cells (K = 2 to 4)**.
A cell holds an 8-bit bin tag and a 56-bit integer, and two cells combine by the rule
*"the larger tag wins, equal tags add"*. That operation is a commutative monoid, so the final
state cannot depend on commit order, warp-level combining or multi-device merging. CAC reads
each contribution once, needs no prior knowledge of the value range, and has a proven error
bound of `n · 2^-(K-1)W · max|x_i|`, with exact results when the dynamic range fits the window.

This repository contains the paper and the **working reference model** that produces every
number in it.

## Key results (from the reference model)

| | |
|---|---|
| Plain atomic add | 32 different output vectors under 32 schedules, in all 28 format/case combinations |
| CAC | one output in every case; all invariance checks passed |
| binary64, CAC with 24 B per destination | correctly rounded on all but one of 53,248 destinations in 13 of 14 cases (exact superaccumulator: 532 B) |
| Known failure | cancellation of near-overflow pairs defeats CAC |
| binary32 | accumulating in binary64 is as accurate as CAC on 13 of 14 cases; CAC adds a guarantee, not accuracy |

All evidence comes from the simulator and from CPU threads. **No physical-GPU timing is claimed.**

## Build and run

Requires a C++17 compiler (g++ or clang++). The scripts need Python 3 with matplotlib.

```sh
make            # builds ./cac (with -ffp-contract=off, required for bit-identical results)
make test       # 55 tests: oracle, monoid laws, invariance, error bound, frozen workload digests
make results    # every experiment -> results/*.csv, then cross-check, tables and figures
```

Individual experiments:

```sh
./cac matrix|invariance|ops|contention|sensitivity|train|threads|rare|dump [outdir]
python3 scripts/check_corollary2.py   # randomised check of the exactness corollary
```

`make results` takes a few minutes on two cores.

## Repository layout

```
include/cac/   common.h fp32.h bigint.h exact.h   PRNG, IEEE decoding, big integers, exact oracle
               accum.h                            CAC and all baselines
               vgpu.h runner.h workload.h         virtual GPU, schedulers, workloads
               train.h threads.h                  SGD experiment, real-thread backend
src/           main.cpp tests.cpp experiments.cpp
scripts/       crosscheck.py        independent Python-integer re-derivation of oracle and CAC
               summarize.py         tables (results/tables.md) and figures (results/figures)
               check_corollary2.py  randomised check of Corollary 2
results/       raw CSVs, tables.md and figures; the paper's numbers come only from here
paper/         Confluent_Accumulation_Cells.pdf and its LaTeX source (Elsevier elsarticle)
```

## Reproducibility

- All pseudo-randomness is integer xoshiro256**; workload digests are frozen in the tests,
  so a platform that generates different data fails `make test`.
- The build disables floating-point contraction (`-ffp-contract=off`).
- `threads.csv` depends on the machine and the moment; only the invariant schemes are
  expected to repeat bit for bit there.
- `crosscheck.py` recomputes the oracle and the CAC specification with Python integers and
  compares them with the C++ output for 28,672 destinations.
- The raw dump files used by the cross-check (about 9 MB) are not committed; `make results`
  regenerates them.

## Paper

The paper is in [`paper/Confluent_Accumulation_Cells.pdf`](paper/Confluent_Accumulation_Cells.pdf).
To rebuild it: `cd paper/latex && xelatex Confluent_Accumulation_Cells.tex` (run twice).

## Citation

```bibtex
@misc{raj2026cac,
  author = {Raj, Aditya},
  title  = {Confluent Accumulation Cells: Bounded-State, Order-Invariant
            Floating-Point Scatter-Add for GPUs},
  year   = {2026},
  note   = {School of AI and Data Science, IIT Jodhpur},
  howpublished = {\url{https://github.com/adityarajIITj/Confluent-Accumulation-Cells-Bounded-State-Order-Invariant-Floating-Point-Scatter-Add-for-GPUs}}
}
```
