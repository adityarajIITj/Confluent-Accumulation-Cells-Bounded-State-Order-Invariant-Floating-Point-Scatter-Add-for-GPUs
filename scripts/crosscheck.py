#!/usr/bin/env python3
"""Independent cross-check of the C++ simulator.

Reads the workloads and results written by `./cac dump <dir>` and recomputes,
with nothing but Python integers (no numpy, no floats, no shared code):

  1. the exact correctly rounded sum (the oracle) of every destination, with the
     IEEE-754 rules for Inf, NaN and the sign of zero, and
  2. the final state and read-out of Confluent Accumulation Cells directly from
     the specification (max-tagged-bin cells), formulated as "per cell, keep the
     largest tag and add the digits that carry it" rather than as a sequence of
     compare-and-swap operations.

Both are compared bit for bit with the columns the C++ code produced.
Run:  python3 -I scripts/crosscheck.py results
Exit status 0 means every destination of every case agrees.
"""
import glob
import os
import sys

FORMATS = {
    # name: (precision p, exponent bits, grid size in bits)
    32: (24, 8, 277),    # grid unit 2^-149, largest finite value < 2^128
    64: (53, 11, 2098),  # grid unit 2^-1074, largest finite value < 2^1024
}
MASK56 = (1 << 56) - 1


def decode(bits, fmt):
    p, eb, _ = FORMATS[fmt]
    neg = (bits >> (p - 1 + eb)) & 1
    be = (bits >> (p - 1)) & ((1 << eb) - 1)
    frac = bits & ((1 << (p - 1)) - 1)
    if be == (1 << eb) - 1:
        return ("inf" if frac == 0 else "nan", neg, 0, 0)
    if be == 0:
        return ("zero", neg, 0, 0) if frac == 0 else ("fin", neg, frac, 0)
    return ("fin", neg, frac | (1 << (p - 1)), be - 1)


def encode_special(kind, neg, fmt):
    p, eb, _ = FORMATS[fmt]
    top = ((1 << eb) - 1) << (p - 1)
    if kind == "nan":
        return top | (1 << (p - 2))          # canonical quiet NaN, sign clear
    if kind == "inf":
        return top | (neg << (p - 1 + eb))
    return neg << (p - 1 + eb)               # zero


def round_grid(S, fmt):
    """Round the integer S (in grid units) to the nearest representable value,
    ties to even, and return the bit pattern."""
    p, eb, kgrid = FORMATS[fmt]
    if S == 0:
        return 0
    neg = 1 if S < 0 else 0
    M = -S if neg else S
    nb = M.bit_length()
    if nb > p:
        sh = nb - p
        r = M >> sh
        rem = M & ((1 << sh) - 1)
        half = 1 << (sh - 1)
        if rem > half or (rem == half and (r & 1)):
            r += 1
        V = r << sh
    else:
        V = M
    if V >= (1 << kgrid):                    # overflow to infinity
        return encode_special("inf", neg, fmt)
    nv = V.bit_length()
    if nv < p:                               # subnormal (or exact small value)
        return (neg << (p - 1 + eb)) | V
    be = nv - p + 1
    mant = (V >> (nv - p)) & ((1 << (p - 1)) - 1)
    return (neg << (p - 1 + eb)) | (be << (p - 1)) | mant


def oracle(addends, fmt):
    nan = pinf = ninf = False
    all_negzero, S = True, 0
    for b in addends:
        kind, neg, m, q = decode(b, fmt)
        if kind == "nan":
            nan, all_negzero = True, False
        elif kind == "inf":
            if neg:
                ninf = True
            else:
                pinf = True
            all_negzero = False
        elif kind == "zero":
            if not neg:
                all_negzero = False
        else:
            all_negzero = False
            S += -(m << q) if neg else (m << q)
    if nan or (pinf and ninf):
        return encode_special("nan", 0, fmt)
    if pinf:
        return encode_special("inf", 0, fmt)
    if ninf:
        return encode_special("inf", 1, fmt)
    if S == 0:
        return encode_special("zero", 1 if (addends and all_negzero) else 0, fmt)
    return round_grid(S, fmt)


def slices(m, q, W):
    """Base-2^W digits of m * 2^q as (bin, digit) with digit != 0."""
    j, off = divmod(q, W)
    v = m << off
    out = []
    while v:
        d = v & ((1 << W) - 1)
        if d:
            out.append((j, d))
        v >>= W
        j += 1
    return out


def cac(addends, fmt, K, W):
    """CAC read-out from the specification."""
    _, _, kgrid = FORMATS[fmt]
    NB = (kgrid + W - 1) // W
    tag_zero, tag_inf, tag_nan = 1, NB + 2, NB + 3
    deposits = {}  # cell slot -> list of (tag, signed value)

    def put(slot, tag, val):
        deposits.setdefault(slot, []).append((tag, val))

    for b in addends:
        kind, neg, m, q = decode(b, fmt)
        if kind == "zero":
            put(0, tag_zero, (1 << 28) if neg else 1)
        elif kind == "inf":
            put(NB % K, tag_inf, (1 << 28) if neg else 1)
        elif kind == "nan":
            put((NB + 1) % K, tag_nan, 1)
        else:
            for j, d in slices(m, q, W):
                put(j % K, j + 2, -d if neg else d)
    nan = pinf = ninf = pzero = nzero = any_real = False
    S = 0
    for slot, dep in deposits.items():
        t = max(tag for tag, _ in dep)
        raw = sum(v for tag, v in dep if tag == t) & MASK56   # sum mod 2^56
        lo, hi = raw & ((1 << 28) - 1), raw >> 28
        if t == tag_nan:
            nan = True
        elif t == tag_inf:
            pinf |= bool(lo)
            ninf |= bool(hi)
        elif t == tag_zero:
            pzero |= bool(lo)
            nzero |= bool(hi)
        else:
            any_real = True
            sv = raw - (1 << 56) if raw & (1 << 55) else raw   # two's complement
            S += sv << ((t - 2) * W)
    if nan or (pinf and ninf):
        return encode_special("nan", 0, fmt)
    if pinf:
        return encode_special("inf", 0, fmt)
    if ninf:
        return encode_special("inf", 1, fmt)
    if S == 0:
        return encode_special("zero", 1 if (not any_real and nzero and not pzero) else 0, fmt)
    return round_grid(S, fmt)


def canon(bits, fmt):
    p, eb, _ = FORMATS[fmt]
    kind, neg, _, _ = decode(bits, fmt)
    return encode_special("nan", 0, fmt) if kind == "nan" else bits


def load_case(path_prefix, fmt):
    cname = None
    per_dst = {}
    with open(path_prefix + "_contrib.txt") as f:
        for line in f:
            if line.startswith("#"):
                cname = line[1:].split("D=")[0].strip()
                continue
            d, b = line.split()
            per_dst.setdefault(int(d), []).append(int(b, 16))
    rows = []
    with open(path_prefix + "_result.txt") as f:
        cols = None
        for line in f:
            if line.startswith("#"):
                cols = line[1:].split()
                continue
            rows.append([int(x, 16) if i else int(x) for i, x in enumerate(line.split())])
    return cname, cols, per_dst, rows


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "results"
    cac_cfg = {
        32: {"cac2x32": (2, 32), "cac3x24": (3, 24)},
        64: {"cac3x32": (3, 32), "cac4x24": (4, 24)},
    }
    total = bad = 0
    for fmt in (32, 64):
        for ci in range(14):
            prefix = os.path.join(outdir, "dump%d_%d" % (fmt, ci))
            if not os.path.exists(prefix + "_contrib.txt"):
                print("missing", prefix)
                return 2
            cname, cols, per_dst, rows = load_case(prefix, fmt)
            mism = {c: 0 for c in ["oracle"] + list(cac_cfg[fmt])}
            for row in rows:
                rec = dict(zip(cols, row))
                d = rec["dst"]
                addends = [rec["init"]] + per_dst.get(d, [])
                want = oracle(addends, fmt)
                if canon(want, fmt) != canon(rec["oracle"], fmt):
                    mism["oracle"] += 1
                for name, (K, W) in cac_cfg[fmt].items():
                    if canon(cac(addends, fmt, K, W), fmt) != canon(rec[name], fmt):
                        mism[name] += 1
                total += 1
            n_bad = sum(mism.values())
            bad += n_bad
            print("binary%d %-18s dests=%4d  mismatches: %s  %s" % (
                fmt, cname, len(rows),
                ", ".join("%s=%d" % kv for kv in mism.items()),
                "OK" if n_bad == 0 else "FAIL"))
    print("\n%d destination checks, %d mismatching columns" % (total, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
