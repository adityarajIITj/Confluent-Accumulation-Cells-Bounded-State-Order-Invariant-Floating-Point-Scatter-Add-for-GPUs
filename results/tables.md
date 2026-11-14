### T1. The status quo (plain floating-point atomicAdd) across 32 schedules

| format | case | schedules | distinct output vectors | % destinations with >1 value | destinations whose Inf/NaN class flips |
|---|---|---|---|---|---|
| binary32 | uniform/normal | 32 | 32 | 100.0 | 0 |
| binary32 | uniform/wide | 32 | 32 | 98.8 | 0 |
| binary32 | uniform/cancel | 32 | 32 | 100.0 | 0 |
| binary32 | uniform/grad | 32 | 32 | 100.0 | 0 |
| binary32 | zipf/normal | 32 | 32 | 85.1 | 0 |
| binary32 | zipf/wide | 32 | 32 | 50.5 | 0 |
| binary32 | zipf/cancel | 32 | 32 | 82.5 | 0 |
| binary32 | zipf/grad | 32 | 32 | 85.5 | 0 |
| binary32 | splat/normal | 32 | 32 | 100.0 | 0 |
| binary32 | splat/wide | 32 | 32 | 98.6 | 0 |
| binary32 | splat/cancel | 32 | 32 | 100.0 | 0 |
| binary32 | splat/grad | 32 | 32 | 100.0 | 0 |
| binary32 | uniform/overflow | 32 | 32 | 100.0 | 4096 |
| binary32 | uniform/subnormal | 32 | 32 | 99.9 | 0 |
| binary64 | uniform/normal | 32 | 32 | 71.8 | 0 |
| binary64 | uniform/wide | 32 | 32 | 96.9 | 0 |
| binary64 | uniform/cancel | 32 | 32 | 100.0 | 0 |
| binary64 | uniform/grad | 32 | 32 | 100.0 | 0 |
| binary64 | zipf/normal | 32 | 32 | 20.5 | 0 |
| binary64 | zipf/wide | 32 | 32 | 43.6 | 0 |
| binary64 | zipf/cancel | 32 | 32 | 80.0 | 0 |
| binary64 | zipf/grad | 32 | 32 | 66.3 | 0 |
| binary64 | splat/normal | 32 | 32 | 100.0 | 0 |
| binary64 | splat/wide | 32 | 32 | 96.3 | 0 |
| binary64 | splat/cancel | 32 | 32 | 100.0 | 0 |
| binary64 | splat/grad | 32 | 32 | 100.0 | 0 |
| binary64 | uniform/overflow | 32 | 32 | 100.0 | 4096 |
| binary64 | uniform/subnormal | 32 | 32 | 98.3 | 0 |

### T2a. All schemes, binary32 (14 cases x 32 schedules; gates = schedule, permutation, 2/4/8 devices, combining)

| scheme | state B/dest | passes | % dests order-unstable | invariance gates passed | % results correctly rounded | results with wrong Inf/NaN class |
|---|---|---|---|---|---|---|
| FLOAT | 4 | 1 | 92.9 | 0/84 | 32.44 | 111878 |
| DOUBLE-ACC | 8 | 1 | 6.8 | 78/84 | 92.95 | 0 |
| SORT-IDX | 0 | 1 | 0.0 | 14/70 | 32.41 | 3488 |
| SORT-VAL | 0 | 1 | 0.0 | 28/70 | 27.44 | 4096 |
| FIXED-Q32 | 8 | 1 | 0.0 | 84/84 | 60.31 | 0 |
| WINDOW-2P | 16 | 2 | 0.0 | 84/84 | 92.92 | 0 |
| SUPERACC | 76 | 1 | 0.0 | 84/84 | 100.00 | 0 |
| CAC-2x32 | 16 | 1 | 0.0 | 84/84 | 92.92 | 0 |
| CAC-3x32 | 24 | 1 | 0.0 | 84/84 | 92.93 | 0 |

### T2b. All schemes, binary64

| scheme | state B/dest | passes | % dests order-unstable | invariance gates passed | % results correctly rounded | results with wrong Inf/NaN class |
|---|---|---|---|---|---|---|
| FLOAT64 | 8 | 1 | 83.8 | 0/84 | 42.09 | 111878 |
| FIXED64-Q48 | 8 | 1 | 0.0 | 84/84 | 2.94 | 0 |
| SUPERACC64 | 532 | 1 | 0.0 | 84/84 | 100.00 | 0 |
| CAC64-2x32 | 16 | 1 | 0.0 | 84/84 | 36.95 | 0 |
| CAC64-3x32 | 24 | 1 | 0.0 | 84/84 | 92.93 | 0 |
| CAC64-4x32 | 32 | 1 | 0.0 | 84/84 | 92.93 | 0 |

### T3a. Percentage of destination results that are correctly rounded (binary32; mean over schedules)

| case | FLOAT | DOUBLE-ACC | FIXED-Q32 | WINDOW-2P | CAC-2x32 | CAC-3x32 | SUPERACC |
|---|---|---|---|---|---|---|---|
| uniform/normal | 27.1 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| uniform/wide | 49.5 | 100.0 | 1.3 | 100.0 | 100.0 | 100.0 | 100.0 |
| uniform/cancel | 1.1 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| uniform/grad | 27.8 | 100.0 | 50.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| zipf/normal | 51.5 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| zipf/wide | 74.5 | 100.0 | 31.0 | 99.9 | 100.0 | 100.0 | 100.0 |
| zipf/cancel | 30.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| zipf/grad | 51.6 | 100.0 | 48.6 | 100.0 | 100.0 | 100.0 | 100.0 |
| splat/normal | 28.3 | 100.0 | 97.7 | 100.0 | 100.0 | 100.0 | 100.0 |
| splat/wide | 51.3 | 100.0 | 11.3 | 100.0 | 100.0 | 100.0 | 100.0 |
| splat/cancel | 0.5 | 100.0 | 100.0 | 100.0 | 99.9 | 100.0 | 100.0 |
| splat/grad | 28.9 | 100.0 | 4.5 | 100.0 | 100.0 | 100.0 | 100.0 |
| uniform/overflow | 0.0 | 1.2 | 100.0 | 1.0 | 1.0 | 1.0 | 100.0 |
| uniform/subnormal | 31.2 | 100.0 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 |

### T3b. Same, binary64

| case | FLOAT64 | FIXED64-Q48 | CAC64-2x32 | CAC64-3x32 | CAC64-4x32 | SUPERACC64 |
|---|---|---|---|---|---|---|
| uniform/normal | 78.2 | 6.3 | 82.9 | 100.0 | 100.0 | 100.0 |
| uniform/wide | 56.2 | 0.0 | 20.8 | 100.0 | 100.0 | 100.0 |
| uniform/cancel | 1.5 | 0.0 | 88.3 | 100.0 | 100.0 | 100.0 |
| uniform/grad | 31.9 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 |
| zipf/normal | 91.6 | 12.5 | 91.0 | 100.0 | 100.0 | 100.0 |
| zipf/wide | 78.5 | 1.8 | 29.4 | 100.0 | 100.0 | 100.0 |
| zipf/cancel | 34.3 | 4.2 | 93.4 | 100.0 | 100.0 | 100.0 |
| zipf/grad | 64.6 | 0.5 | 0.5 | 100.0 | 100.0 | 100.0 |
| splat/normal | 31.0 | 0.3 | 0.4 | 100.0 | 100.0 | 100.0 |
| splat/wide | 56.1 | 0.0 | 6.6 | 100.0 | 100.0 | 100.0 |
| splat/cancel | 0.5 | 0.0 | 3.0 | 100.0 | 100.0 | 100.0 |
| splat/grad | 29.8 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 |
| uniform/overflow | 0.1 | 15.4 | 1.0 | 1.0 | 1.0 | 100.0 |
| uniform/subnormal | 34.9 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 |

### T4. Cost model on uniform/normal (sequential schedule, no contention): per contribution

| format | scheme | atomic primitives issued | atomic ops executed | loads | dropped (tag already larger) | state B/dest | passes | scratch B/contribution |
|---|---|---|---|---|---|---|---|---|
| binary32 | FLOAT | 1.00 | 1.00 | 0.00 | 0.000 | 4.0 | 1 | 0 |
| binary32 | DOUBLE-ACC | 1.00 | 1.00 | 0.00 | 0.000 | 8.0 | 1 | 0 |
| binary32 | SORT-IDX | 0.00 | 0.00 | 0.00 | 0.000 | 0.0 | 1 | 8 |
| binary32 | SORT-VAL | 0.00 | 0.00 | 0.00 | 0.000 | 0.0 | 1 | 8 |
| binary32 | FIXED-Q32 | 1.00 | 1.00 | 0.00 | 0.000 | 8.0 | 1 | 0 |
| binary32 | WINDOW-2P | 2.00 | 2.00 | 0.00 | 0.000 | 16.0 | 2 | 0 |
| binary32 | SUPERACC | 2.86 | 1.89 | 1.00 | 0.000 | 76.0 | 1 | 0 |
| binary32 | CAC-2x32 | 1.86 | 1.86 | 1.86 | 0.000 | 16.0 | 1 | 0 |
| binary32 | CAC-3x32 | 1.86 | 1.86 | 1.86 | 0.000 | 24.0 | 1 | 0 |
| binary32 | CAC-2x24 | 1.98 | 1.97 | 1.98 | 0.008 | 16.0 | 1 | 0 |
| binary32 | CAC-2x32-NATIVE | 1.86 | 1.86 | 0.00 | 0.000 | 16.0 | 1 | 0 |
| binary64 | FLOAT64 | 1.00 | 1.00 | 0.00 | 0.000 | 8.0 | 1 | 0 |
| binary64 | FIXED64-Q48 | 1.00 | 1.00 | 0.00 | 0.000 | 8.0 | 1 | 0 |
| binary64 | SUPERACC64 | 3.01 | 2.04 | 1.00 | 0.000 | 532.0 | 1 | 0 |
| binary64 | CAC64-2x32 | 2.01 | 2.00 | 2.01 | 0.008 | 16.0 | 1 | 0 |
| binary64 | CAC64-3x32 | 2.01 | 2.01 | 2.01 | 0.000 | 24.0 | 1 | 0 |
| binary64 | CAC64-4x32 | 2.01 | 2.01 | 2.01 | 0.000 | 32.0 | 1 | 0 |
| binary64 | CAC64-3x32-NATIVE | 2.01 | 2.01 | 0.00 | 0.000 | 24.0 | 1 | 0 |

### T4b. Atomic operations per contribution on the other op-count cases (wide dynamic range and gradient-like)

| format | case | scheme | atomic ops/contribution | dropped/contribution |
|---|---|---|---|---|
| binary32 | uniform/wide | FLOAT | 1.00 | 0.000 |
| binary32 | uniform/wide | SUPERACC | 1.68 | 0.000 |
| binary32 | uniform/wide | CAC-2x32 | 1.08 | 0.571 |
| binary32 | uniform/wide | CAC-3x32 | 1.55 | 0.101 |
| binary32 | uniform/wide | CAC-2x24 | 0.92 | 0.997 |
| binary32 | uniform/wide | CAC-2x32-NATIVE | 1.65 | 0.000 |
| binary32 | uniform/grad | FLOAT | 1.00 | 0.000 |
| binary32 | uniform/grad | SUPERACC | 2.03 | 0.000 |
| binary32 | uniform/grad | CAC-2x32 | 2.00 | 0.000 |
| binary32 | uniform/grad | CAC-3x32 | 2.00 | 0.000 |
| binary32 | uniform/grad | CAC-2x24 | 1.93 | 0.009 |
| binary32 | uniform/grad | CAC-2x32-NATIVE | 2.00 | 0.000 |
| binary64 | uniform/wide | FLOAT64 | 1.00 | 0.000 |
| binary64 | uniform/wide | SUPERACC64 | 2.61 | 0.000 |
| binary64 | uniform/wide | CAC64-2x32 | 0.61 | 1.963 |
| binary64 | uniform/wide | CAC64-3x32 | 1.01 | 1.566 |
| binary64 | uniform/wide | CAC64-4x32 | 1.40 | 1.176 |
| binary64 | uniform/wide | CAC64-3x32-NATIVE | 2.57 | 0.000 |
| binary64 | uniform/grad | FLOAT64 | 1.00 | 0.000 |
| binary64 | uniform/grad | SUPERACC64 | 3.03 | 0.000 |
| binary64 | uniform/grad | CAC64-2x32 | 2.03 | 0.968 |
| binary64 | uniform/grad | CAC64-3x32 | 3.00 | 0.000 |
| binary64 | uniform/grad | CAC64-4x32 | 3.00 | 0.000 |
| binary64 | uniform/grad | CAC64-3x32-NATIVE | 3.00 | 0.000 |

### T5. Compare-and-swap behaviour on a Zipf destination distribution (window schedule; fail % = failed / attempted)

| zipf alpha | warps in flight | scheme | warp combining | atomic ops per issued primitive | CAS failure % | output identical to sequential |
|---|---|---|---|---|---|---|
| 0.0 | 1 | FLOAT-CAS | 0 | 1.002 | 0.2 | 0 |
| 0.0 | 1 | FLOAT-CAS | 1 | 1.000 | 0.0 | 0 |
| 0.0 | 64 | FLOAT-CAS | 0 | 1.080 | 7.4 | 0 |
| 0.0 | 64 | FLOAT-CAS | 1 | 1.079 | 7.3 | 0 |
| 0.0 | 512 | FLOAT-CAS | 0 | 1.538 | 35.0 | 0 |
| 0.0 | 512 | FLOAT-CAS | 1 | 1.541 | 35.1 | 0 |
| 0.0 | 1 | CAC-2x32 | 0 | 1.001 | 0.1 | 1 |
| 0.0 | 1 | CAC-2x32 | 1 | 1.000 | 0.0 | 1 |
| 0.0 | 64 | CAC-2x32 | 0 | 1.051 | 4.9 | 1 |
| 0.0 | 64 | CAC-2x32 | 1 | 1.050 | 4.7 | 1 |
| 0.0 | 512 | CAC-2x32 | 0 | 1.383 | 27.7 | 1 |
| 0.0 | 512 | CAC-2x32 | 1 | 1.383 | 27.7 | 1 |
| 0.0 | 1 | CAC-2x32-NATIVE | 0 | 1.000 | 0.0 | 1 |
| 0.0 | 1 | CAC-2x32-NATIVE | 1 | 1.000 | 0.0 | 1 |
| 0.0 | 64 | CAC-2x32-NATIVE | 0 | 1.000 | 0.0 | 1 |
| 0.0 | 64 | CAC-2x32-NATIVE | 1 | 1.000 | 0.0 | 1 |
| 0.0 | 512 | CAC-2x32-NATIVE | 0 | 1.000 | 0.0 | 1 |
| 0.0 | 512 | CAC-2x32-NATIVE | 1 | 1.000 | 0.0 | 1 |
| 0.5 | 1 | FLOAT-CAS | 0 | 1.004 | 0.4 | 0 |
| 0.5 | 1 | FLOAT-CAS | 1 | 1.000 | 0.0 | 0 |
| 0.5 | 64 | FLOAT-CAS | 0 | 1.174 | 14.8 | 0 |
| 0.5 | 64 | FLOAT-CAS | 1 | 1.162 | 13.9 | 0 |
| 0.5 | 512 | FLOAT-CAS | 0 | 1.868 | 46.5 | 0 |
| 0.5 | 512 | FLOAT-CAS | 1 | 1.839 | 45.6 | 0 |
| 0.5 | 1 | CAC-2x32 | 0 | 1.003 | 0.3 | 1 |
| 0.5 | 1 | CAC-2x32 | 1 | 1.000 | 0.0 | 1 |
| 0.5 | 64 | CAC-2x32 | 0 | 1.114 | 10.2 | 1 |
| 0.5 | 64 | CAC-2x32 | 1 | 1.105 | 9.5 | 1 |
| 0.5 | 512 | CAC-2x32 | 0 | 1.624 | 38.4 | 1 |
| 0.5 | 512 | CAC-2x32 | 1 | 1.605 | 37.7 | 1 |
| 0.5 | 1 | CAC-2x32-NATIVE | 0 | 1.000 | 0.0 | 1 |
| 0.5 | 1 | CAC-2x32-NATIVE | 1 | 1.000 | 0.0 | 1 |
| 0.5 | 64 | CAC-2x32-NATIVE | 0 | 1.000 | 0.0 | 1 |
| 0.5 | 64 | CAC-2x32-NATIVE | 1 | 1.000 | 0.0 | 1 |
| 0.5 | 512 | CAC-2x32-NATIVE | 0 | 1.000 | 0.0 | 1 |
| 0.5 | 512 | CAC-2x32-NATIVE | 1 | 1.000 | 0.0 | 1 |
| 1.0 | 1 | FLOAT-CAS | 0 | 1.138 | 12.1 | 0 |
| 1.0 | 1 | FLOAT-CAS | 1 | 1.000 | 0.0 | 0 |
| 1.0 | 64 | FLOAT-CAS | 0 | 3.139 | 68.1 | 0 |
| 1.0 | 64 | FLOAT-CAS | 1 | 1.979 | 49.5 | 0 |
| 1.0 | 512 | FLOAT-CAS | 0 | 7.933 | 87.4 | 0 |
| 1.0 | 512 | FLOAT-CAS | 1 | 4.532 | 77.9 | 0 |
| 1.0 | 1 | CAC-2x32 | 0 | 1.106 | 9.6 | 1 |
| 1.0 | 1 | CAC-2x32 | 1 | 1.000 | 0.0 | 1 |
| 1.0 | 64 | CAC-2x32 | 0 | 2.594 | 61.4 | 1 |
| 1.0 | 64 | CAC-2x32 | 1 | 1.708 | 41.5 | 1 |
| 1.0 | 512 | CAC-2x32 | 0 | 6.306 | 84.1 | 1 |
| 1.0 | 512 | CAC-2x32 | 1 | 3.649 | 72.6 | 1 |
| 1.0 | 1 | CAC-2x32-NATIVE | 0 | 1.000 | 0.0 | 1 |
| 1.0 | 1 | CAC-2x32-NATIVE | 1 | 1.000 | 0.0 | 1 |
| 1.0 | 64 | CAC-2x32-NATIVE | 0 | 1.000 | 0.0 | 1 |
| 1.0 | 64 | CAC-2x32-NATIVE | 1 | 1.000 | 0.0 | 1 |
| 1.0 | 512 | CAC-2x32-NATIVE | 0 | 1.000 | 0.0 | 1 |
| 1.0 | 512 | CAC-2x32-NATIVE | 1 | 1.000 | 0.0 | 1 |
| 1.5 | 1 | FLOAT-CAS | 0 | 1.923 | 48.0 | 0 |
| 1.5 | 1 | FLOAT-CAS | 1 | 1.000 | 0.0 | 0 |
| 1.5 | 64 | FLOAT-CAS | 0 | 9.674 | 89.7 | 0 |
| 1.5 | 64 | FLOAT-CAS | 1 | 3.226 | 69.0 | 0 |
| 1.5 | 512 | FLOAT-CAS | 0 | 27.967 | 96.4 | 0 |
| 1.5 | 512 | FLOAT-CAS | 1 | 8.596 | 88.4 | 0 |
| 1.5 | 1 | CAC-2x32 | 0 | 1.725 | 42.0 | 1 |
| 1.5 | 1 | CAC-2x32 | 1 | 1.000 | 0.0 | 1 |
| 1.5 | 64 | CAC-2x32 | 0 | 7.886 | 87.3 | 1 |
| 1.5 | 64 | CAC-2x32 | 1 | 2.601 | 61.6 | 1 |
| 1.5 | 512 | CAC-2x32 | 0 | 22.654 | 95.6 | 1 |
| 1.5 | 512 | CAC-2x32 | 1 | 6.744 | 85.2 | 1 |
| 1.5 | 1 | CAC-2x32-NATIVE | 0 | 1.000 | 0.0 | 1 |
| 1.5 | 1 | CAC-2x32-NATIVE | 1 | 1.000 | 0.0 | 1 |
| 1.5 | 64 | CAC-2x32-NATIVE | 0 | 1.000 | 0.0 | 1 |
| 1.5 | 64 | CAC-2x32-NATIVE | 1 | 1.000 | 0.0 | 1 |
| 1.5 | 512 | CAC-2x32-NATIVE | 0 | 1.000 | 0.0 | 1 |
| 1.5 | 512 | CAC-2x32-NATIVE | 1 | 1.000 | 0.0 | 1 |
| 2.0 | 1 | FLOAT-CAS | 0 | 2.731 | 63.4 | 0 |
| 2.0 | 1 | FLOAT-CAS | 1 | 1.000 | 0.0 | 0 |
| 2.0 | 64 | FLOAT-CAS | 0 | 15.623 | 93.6 | 0 |
| 2.0 | 64 | FLOAT-CAS | 1 | 4.227 | 76.3 | 0 |
| 2.0 | 512 | FLOAT-CAS | 0 | 45.658 | 97.8 | 0 |
| 2.0 | 512 | FLOAT-CAS | 1 | 11.757 | 91.5 | 0 |
| 2.0 | 1 | CAC-2x32 | 0 | 2.377 | 57.9 | 1 |
| 2.0 | 1 | CAC-2x32 | 1 | 1.000 | 0.0 | 1 |
| 2.0 | 64 | CAC-2x32 | 0 | 12.780 | 92.2 | 1 |
| 2.0 | 64 | CAC-2x32 | 1 | 3.336 | 70.0 | 1 |
| 2.0 | 512 | CAC-2x32 | 0 | 37.415 | 97.3 | 1 |
| 2.0 | 512 | CAC-2x32 | 1 | 9.111 | 89.0 | 1 |
| 2.0 | 1 | CAC-2x32-NATIVE | 0 | 1.000 | 0.0 | 1 |
| 2.0 | 1 | CAC-2x32-NATIVE | 1 | 1.000 | 0.0 | 1 |
| 2.0 | 64 | CAC-2x32-NATIVE | 0 | 1.000 | 0.0 | 1 |
| 2.0 | 64 | CAC-2x32-NATIVE | 1 | 1.000 | 0.0 | 1 |
| 2.0 | 512 | CAC-2x32-NATIVE | 0 | 1.000 | 0.0 | 1 |
| 2.0 | 512 | CAC-2x32-NATIVE | 1 | 1.000 | 0.0 | 1 |

### T6. Sensitivity grid: correctly rounded % over all (addends, range, shape) per (K, W)

| format | K | W | guaranteed bits | state B/dest | CAC % correctly rounded | plain-float % correctly rounded | max (observed error / Theorem-3 bound) |
|---|---|---|---|---|---|---|---|
| binary32 | 1 | 16 | 0 | 8 | 0.0 | 14.1 | 0.538 |
| binary32 | 1 | 24 | 0 | 8 | 0.5 | 14.1 | 0.538 |
| binary32 | 1 | 32 | 0 | 8 | 16.0 | 14.1 | 0.573 |
| binary32 | 2 | 16 | 16 | 16 | 17.7 | 14.1 | 0.573 |
| binary32 | 2 | 24 | 24 | 16 | 75.2 | 14.1 | 0.234 |
| binary32 | 2 | 32 | 32 | 16 | 86.4 | 14.1 | 0.415 |
| binary32 | 3 | 16 | 32 | 24 | 79.3 | 14.1 | 0.415 |
| binary32 | 3 | 24 | 48 | 24 | 99.9 | 14.1 | 0.363 |
| binary32 | 3 | 32 | 64 | 24 | 100.0 | 14.1 | 0.395 |
| binary32 | 4 | 16 | 48 | 32 | 99.8 | 14.1 | 0.308 |
| binary32 | 4 | 24 | 72 | 32 | 100.0 | 14.1 | 0.395 |
| binary32 | 4 | 32 | 96 | 32 | 100.0 | 14.1 | 0.206 |
| binary32 | all |  |  |  | 576 configs | invariant in 576 | 0 violations of Theorem 3 |
| binary64 | 2 | 24 | 24 | 16 | 0.1 | 18.4 | 0.676 |
| binary64 | 2 | 32 | 32 | 16 | 12.7 | 18.4 | 0.438 |
| binary64 | 2 | 40 | 40 | 16 | 39.0 | 18.4 | 0.447 |
| binary64 | 3 | 24 | 48 | 24 | 39.1 | 18.4 | 0.433 |
| binary64 | 3 | 32 | 64 | 24 | 72.1 | 18.4 | 0.412 |
| binary64 | 3 | 40 | 80 | 24 | 88.2 | 18.4 | 0.311 |
| binary64 | 4 | 24 | 72 | 32 | 74.1 | 18.4 | 0.391 |
| binary64 | 4 | 32 | 96 | 32 | 99.9 | 18.4 | 0.346 |
| binary64 | 4 | 40 | 120 | 32 | 100.0 | 18.4 | 0.286 |
| binary64 | 5 | 24 | 96 | 40 | 99.1 | 18.4 | 0.313 |
| binary64 | 5 | 32 | 128 | 40 | 100.0 | 18.4 | 0.304 |
| binary64 | 5 | 40 | 160 | 40 | 100.0 | 18.4 | 0.213 |
| binary64 | all |  |  |  | 576 configs | invariant in 576 | 0 violations of Theorem 3 |

### T7. Matrix-factorisation SGD after 400 steps: fraction of parameters that differ from replica 0 / max loss difference

| scheme | different schedule | different device count | different input order |
|---|---|---|---|
| FLOAT | 0.845 / 3.1e-10 | 0.846 / 2.8e-09 | 0.836 / 3.1e-09 |
| DOUBLE-ACC | 0.000 / 0 | 0.000 / 0 | 0.000 / 0 |
| SORT-IDX | 0.000 / 0 | 0.846 / 2.8e-09 | 0.836 / 3.1e-09 |
| FIXED-Q32 | 0.000 / 0 | 0.000 / 0 | 0.000 / 0 |
| WINDOW-2P | 0.000 / 0 | 0.000 / 0 | 0.000 / 0 |
| SUPERACC | 0.000 / 0 | 0.000 / 0 | 0.000 / 0 |
| CAC-2x32 | 0.000 / 0 | 0.000 / 0 | 0.000 / 0 |

### T8. Real operating-system threads (std::atomic), 2^22 contributions, Zipf destinations, gradient-like values

| scheme | threads | runs | distinct output vectors | runs equal to simulated sequential output | median seconds |
|---|---|---|---|---|---|
| FLOAT | 1 | 12 | 1 | 12 | 0.041 |
| FLOAT | 2 | 12 | 12 | 0 | 0.050 |
| DOUBLE-ACC | 1 | 12 | 1 | 12 | 0.042 |
| DOUBLE-ACC | 2 | 12 | 1 | 12 | 0.049 |
| CAC-2x32 | 1 | 12 | 1 | 12 | 0.103 |
| CAC-2x32 | 2 | 12 | 1 | 12 | 0.109 |
| CAC-3x32 | 1 | 12 | 1 | 12 | 0.105 |
| CAC-3x32 | 2 | 12 | 1 | 12 | 0.110 |
| SUPERACC | 1 | 12 | 1 | 12 | 0.098 |
| SUPERACC | 2 | 12 | 1 | 12 | 0.093 |
| FLOAT64 | 1 | 12 | 1 | 12 | 0.040 |
| FLOAT64 | 2 | 12 | 12 | 0 | 0.049 |
| CAC64-3x32 | 1 | 12 | 1 | 12 | 0.151 |
| CAC64-3x32 | 2 | 12 | 1 | 12 | 0.137 |
| SUPERACC64 | 1 | 12 | 1 | 12 | 0.135 |
| SUPERACC64 | 2 | 12 | 1 | 12 | 0.123 |

Wall-clock seconds are for the machine the experiment ran on and are not a GPU performance claim.

### T9. Monte Carlo: fp32 result changes when 16 addends are summed forward vs backward

| dynamic range (binary orders) | addends | trials | fp64 accumulator: results that differ | fp64 forward not correctly rounded | fp64 backward not correctly rounded | plain fp32 accumulator: % of trials that differ |
|---|---|---|---|---|---|---|
| 0 | 16 | 67108864 | 0 | 0 | 0 | 71.4 |
| 24 | 16 | 67108864 | 0 | 0 | 0 | 64.2 |
| 32 | 16 | 67108864 | 0 | 0 | 0 | 61.7 |
| 48 | 16 | 268435456 | 0 | 0 | 0 | 54.7 |
| 80 | 16 | 268435456 | 3 | 1 | 2 | 42.9 |
