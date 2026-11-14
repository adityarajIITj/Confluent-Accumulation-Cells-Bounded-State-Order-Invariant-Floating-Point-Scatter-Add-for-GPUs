#!/usr/bin/env python3
"""Randomised check of Corollary 2 (exactness when max/min <= 2^((K-1)W - p + 1)).

Uses the Python-integer oracle and CAC specification of crosscheck.py; it does not
run the C++ simulator.  Run:  python3 -I scripts/check_corollary2.py
"""
import sys, random
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crosscheck as cc
random.seed(12345)
def mk(fmt, e, neg):
    p, eb, _ = cc.FORMATS[fmt]
    bias = (1 << (eb - 1)) - 1
    frac = random.getrandbits(p - 1)
    be = e + bias
    if be <= 0:   # subnormal: value in [2^e, 2^(e+1)) on the subnormal grid
        emin = 1 - bias
        m = (1 << (p - 1)) | frac
        sh = emin - e
        m >>= sh
        if m == 0: m = 1
        return (neg << (p - 1 + eb)) | m
    return (neg << (p - 1 + eb)) | (be << (p - 1)) | frac
cfgs = [(32,2,32),(32,3,32),(32,2,24),(32,3,24),(32,3,16),(32,4,16),(64,3,32),(64,4,32),(64,3,40),(64,4,24)]
for fmt,K,W in cfgs:
    p, eb, _ = cc.FORMATS[fmt]
    R = (K-1)*W - p + 1
    bias = (1 << (eb - 1)) - 1
    lo_e, hi_e = (1 - bias) - (p - 1), bias - 6      # leave headroom so the sum stays finite
    bad = tot = 0
    for _ in range(6000):
        span = random.randint(1, R) if R >= 1 else 0
        if span == 0: break
        e0 = random.randint(lo_e, hi_e - span)
        n = random.randint(2, 40)
        xs = [mk(fmt, e0 + random.randint(0, span - 1), random.getrandbits(1)) for _ in range(n)]
        tot += 1
        if cc.cac(xs, fmt, K, W) != cc.oracle(xs, fmt): bad += 1
    # tightness probe: allow a much wider span and count deviations
    wide = wbad = 0
    for _ in range(3000):
        span = R + W + 8
        e0 = random.randint(lo_e, hi_e - span)
        n = random.randint(2, 40)
        xs = [mk(fmt, e0 + random.randint(0, span - 1), random.getrandbits(1)) for _ in range(n)]
        wide += 1
        if cc.cac(xs, fmt, K, W) != cc.oracle(xs, fmt): wbad += 1
    print(f"binary{fmt} K={K} W={W} R=2^{R}: within range {bad}/{tot} mismatches; span R+W+8: {wbad}/{wide} mismatches")
