#!/usr/bin/env python3
"""Derive every table and figure of the paper from the CSV files in results/.

Nothing here recomputes an experiment: it only aggregates the raw counts that
`./cac all <dir>` wrote.  Output: <dir>/tables.md and <dir>/figures/*.png|pdf.
Run:  python3 -I scripts/summarize.py results
"""
import csv
import collections
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

# Fixed categorical colour per scheme (slot order of the reference palette), so a
# scheme keeps its colour in every figure.  Marker shape is the second encoding.
COLOR = {
    "FLOAT": "#e34948", "FLOAT64": "#e34948", "FLOAT-CAS": "#e34948",
    "DOUBLE-ACC": "#eb6834",
    "SORT-IDX": "#6b6b66", "SORT-VAL": "#9a9a93",
    "FIXED-Q32": "#6250d6", "FIXED64-Q48": "#6250d6",
    "WINDOW-2P": "#eda100",
    "SUPERACC": "#1baf7a", "SUPERACC64": "#1baf7a",
    "CAC-2x32": "#2a78d6", "CAC-3x32": "#184f95", "CAC-2x24": "#6da7ec",
    "CAC64-2x32": "#6da7ec", "CAC64-3x32": "#2a78d6", "CAC64-4x32": "#184f95",
    "CAC-2x32-NATIVE": "#5598e7", "CAC64-3x32-NATIVE": "#5598e7",
}
MARK = {"FLOAT": "v", "FLOAT64": "v", "DOUBLE-ACC": "D", "SORT-IDX": "s", "SORT-VAL": "s",
        "FIXED-Q32": "P", "FIXED64-Q48": "P", "WINDOW-2P": "^", "SUPERACC": "X", "SUPERACC64": "X"}
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#2b2b29", "xtick.color": "#4a4a47",
    "ytick.color": "#4a4a47", "axes.grid": True, "grid.color": "#e4e3de", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.dpi": 200,
})


def rd(name):
    with open(os.path.join(OUT, name)) as f:
        return list(csv.DictReader(f))


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


md = []


def table(title, header, rows, note=None):
    md.append("### " + title + "\n")
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        md.append("| " + " | ".join(str(x) for x in r) + " |")
    if note:
        md.append("\n" + note)
    md.append("")


m32, m64 = rd("matrix.csv"), rd("matrix64.csv")
i32, i64 = rd("invariance.csv"), rd("invariance64.csv")

# ---- T1: the problem ---------------------------------------------------------
rows = []
for fmt, mm in (("binary32", m32), ("binary64", m64)):
    for r in mm:
        if r["scheme"] in ("FLOAT", "FLOAT64"):
            rows.append([fmt, r["case"], r["schedules"], r["distinct_outputs"],
                         "%.1f" % pct(int(r["unstable_dests"]), int(r["D"])),
                         r["class_flip_dests"]])
table("T1. The status quo (plain floating-point atomicAdd) across 32 schedules",
      ["format", "case", "schedules", "distinct output vectors", "% destinations with >1 value",
       "destinations whose Inf/NaN class flips"], rows)

# ---- T2: scheme comparison (aggregated over the 14 cases) ------------------------
def scheme_rows(mm, inv):
    agg = collections.OrderedDict()
    for r in mm:
        a = agg.setdefault(r["scheme"], collections.Counter())
        a["D"] += int(r["D"]); a["unst"] += int(r["unstable_dests"])
        a["res"] += int(r["results"]); a["ex"] += int(r["exact_results"])
        a["flip"] += int(r["class_flip_dests"]); a["mis"] += int(r["class_mismatch"])
        a["bytes"] = float(r["bytes_per_dest"]); a["passes"] = int(r["passes"])
    gate = collections.defaultdict(lambda: [0, 0])
    for r in inv:
        gate[r["scheme"]][1] += 1
        gate[r["scheme"]][0] += int(r["identical"])
    out = []
    for s, a in agg.items():
        g = gate[s]
        out.append([s, "%.0f" % a["bytes"], a["passes"], "%.1f" % pct(a["unst"], a["D"]),
                    "%d/%d" % (g[0], g[1]), "%.2f" % pct(a["ex"], a["res"]), a["mis"]])
    return out


hdr = ["scheme", "state B/dest", "passes", "% dests order-unstable", "invariance gates passed",
       "% results correctly rounded", "results with wrong Inf/NaN class"]
table("T2a. All schemes, binary32 (14 cases x 32 schedules; gates = schedule, permutation, 2/4/8 devices, combining)",
      hdr, scheme_rows(m32, i32))
table("T2b. All schemes, binary64", hdr, scheme_rows(m64, i64))

# ---- T3: per case correctly-rounded percentages -------------------------------------------
cases = []
for r in m32:
    if r["case"] not in cases:
        cases.append(r["case"])
sel32 = ["FLOAT", "DOUBLE-ACC", "FIXED-Q32", "WINDOW-2P", "CAC-2x32", "CAC-3x32", "SUPERACC"]
by = {(r["case"], r["scheme"]): r for r in m32}
rows = []
for c in cases:
    rows.append([c] + ["%.1f" % pct(int(by[(c, s)]["exact_results"]), int(by[(c, s)]["results"])) for s in sel32])
table("T3a. Percentage of destination results that are correctly rounded (binary32; mean over schedules)",
      ["case"] + sel32, rows)
by64 = {(r["case"], r["scheme"]): r for r in m64}
sel64 = ["FLOAT64", "FIXED64-Q48", "CAC64-2x32", "CAC64-3x32", "CAC64-4x32", "SUPERACC64"]
rows = []
for c in cases:
    rows.append([c] + ["%.1f" % pct(int(by64[(c, s)]["exact_results"]), int(by64[(c, s)]["results"])) for s in sel64])
table("T3b. Same, binary64", ["case"] + sel64, rows)

# ---- T4: operation counts -------------------------------------------------------------------
ops = rd("ops.csv")
rows = []
for r in ops:
    if r["case"] != "uniform/normal":
        continue
    n = int(r["contributions"])
    rows.append([r["format"], r["scheme"], "%.2f" % (int(r["prims"]) / n), "%.2f" % (int(r["atomics"]) / n),
                 "%.2f" % (int(r["loads"]) / n), "%.3f" % (int(r["dropped"]) / n),
                 r["bytes_per_dest"], r["passes"], r["temp_bytes_per_contribution"]])
table("T4. Cost model on uniform/normal (sequential schedule, no contention): per contribution",
      ["format", "scheme", "atomic primitives issued", "atomic ops executed", "loads", "dropped (tag already larger)",
       "state B/dest", "passes", "scratch B/contribution"], rows)
rows = []
for fmt in ("binary32", "binary64"):
    for r in ops:
        if r["format"] == fmt and r["scheme"].startswith(("CAC", "SUPER", "FLOAT")) and r["case"] != "uniform/normal":
            n = int(r["contributions"])
            rows.append([fmt, r["case"], r["scheme"], "%.2f" % (int(r["atomics"]) / n), "%.3f" % (int(r["dropped"]) / n)])
table("T4b. Atomic operations per contribution on the other op-count cases (wide dynamic range and gradient-like)",
      ["format", "case", "scheme", "atomic ops/contribution", "dropped/contribution"], rows)

# ---- T5: contention --------------------------------------------------------------------------------
con = rd("contention.csv")
rows = []
for r in con:
    if r["kind"] == "window" and r["window"] in ("1", "64", "512"):
        att = int(r["cas_attempts"])
        rows.append([int(r["alpha_x2"]) / 2.0, r["window"], r["scheme"], r["combine"],
                     "%.3f" % (int(r["atomics"]) / int(r["prims_issued"])),
                     "%.1f" % pct(int(r["cas_fails"]), att), r["identical_to_seq"]])
table("T5. Compare-and-swap behaviour on a Zipf destination distribution (window schedule; fail % = failed / attempted)",
      ["zipf alpha", "warps in flight", "scheme", "warp combining", "atomic ops per issued primitive",
       "CAS failure %", "output identical to sequential"], rows)

# ---- T6: sensitivity ----------------------------------------------------------------------------------
sens = {"binary32": rd("sensitivity.csv"), "binary64": rd("sensitivity64.csv")}
rows = []
for fmt, ss in sens.items():
    viol = sum(1 for r in ss if float(r["max_bound_ratio"]) > 1.0)
    inv = sum(1 for r in ss if r["invariant"] == "1")
    byK = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r in ss:
        a = byK[(int(r["K"]), int(r["W"]))]
        a[0] += int(r["dests"]); a[1] += int(r["exact_dests"]); a[2] += int(r["float_exact_dests"])
        a[3] = max(a[3], float(r["max_bound_ratio"]))
    for (K, W), a in sorted(byK.items()):
        rows.append([fmt, K, W, (K - 1) * W, "%.0f" % (8.0 * K), "%.1f" % pct(a[1], a[0]),
                     "%.1f" % pct(a[2], a[0]), "%.3f" % a[3]])
    rows.append([fmt, "all", "", "", "", "%d configs" % len(ss), "invariant in %d" % inv,
                 "%d violations of Theorem 3" % viol])
table("T6. Sensitivity grid: correctly rounded % over all (addends, range, shape) per (K, W)",
      ["format", "K", "W", "guaranteed bits", "state B/dest", "CAC % correctly rounded",
       "plain-float % correctly rounded", "max (observed error / Theorem-3 bound)"], rows)

# ---- T7: training divergence ------------------------------------------------------------------------
tr = rd("train.csv")
last = max(int(r["step"]) for r in tr)
rows = []
for s in dict.fromkeys(r["scheme"] for r in tr):
    cells = []
    for v in ("schedule", "devices", "order"):
        x = [r for r in tr if r["scheme"] == s and r["variation"] == v and int(r["step"]) == last]
        cells.append("%.3f / %.2g" % (float(x[0]["frac_params_differ"]), float(x[0]["max_loss_diff"])))
    rows.append([s] + cells)
table("T7. Matrix-factorisation SGD after %d steps: fraction of parameters that differ from replica 0 / max loss difference"
      % (last + 1), ["scheme", "different schedule", "different device count", "different input order"], rows)

# ---- T8: real threads ----------------------------------------------------------------------------------
th = rd("threads.csv")
rows = []
for s in dict.fromkeys(r["scheme"] for r in th):
    for T in sorted({int(r["threads"]) for r in th}):
        x = [r for r in th if r["scheme"] == s and int(r["threads"]) == T]
        hs = {r["output_hash"] for r in x}
        rows.append([s, T, len(x), len(hs), sum(int(r["matches_sequential"]) for r in x),
                     "%.3f" % statistics.median(float(r["seconds"]) for r in x)])
table("T8. Real operating-system threads (std::atomic), 2^22 contributions, Zipf destinations, gradient-like values",
      ["scheme", "threads", "runs", "distinct output vectors", "runs equal to simulated sequential output",
       "median seconds"], rows,
      "Wall-clock seconds are for the machine the experiment ran on and are not a GPU performance claim.")

# ---- T9: how rare is order dependence of an fp64 accumulator ------------------------------------------
rare = rd("rare.csv")
rows = [[r["range_bits"], r["addends"], r["trials"], r["double_acc_order_dependent"],
         r["forward_not_exact"], r["backward_not_exact"], "%.1f" % pct(int(r["float_acc_order_dependent"]), int(r["float_trials"]))]
        for r in rare]
table("T9. Monte Carlo: fp32 result changes when 16 addends are summed forward vs backward",
      ["dynamic range (binary orders)", "addends", "trials", "fp64 accumulator: results that differ",
       "fp64 forward not correctly rounded", "fp64 backward not correctly rounded",
       "plain fp32 accumulator: % of trials that differ"], rows)

with open(os.path.join(OUT, "tables.md"), "w") as f:
    f.write("\n".join(md))

# =============================== figures =====================================
# F1: how bad is the status quo
fig, ax = plt.subplots(figsize=(6.4, 2.8))
f1 = [r for r in m32 if r["scheme"] == "FLOAT"]
ax.bar(range(len(f1)), [pct(int(r["unstable_dests"]), int(r["D"])) for r in f1], width=0.6, color=COLOR["FLOAT"])
ax.set_xticks(range(len(f1)))
ax.set_xticklabels([r["case"] for r in f1], rotation=40, ha="right", fontsize=7)
ax.set_ylabel("% destinations with\nmore than one result")
ax.set_ylim(0, 105)
ax.set_title("Plain float atomicAdd, binary32: results that change with the schedule (32 schedules)", fontsize=8, loc="left")
save(fig, "fig1_status_quo")

# F2: state vs correctness
fig, ax = plt.subplots(figsize=(6.0, 3.4))
agg = {}
for r in m32:
    a = agg.setdefault(r["scheme"], [0, 0, 0.0])
    a[0] += int(r["results"]); a[1] += int(r["exact_results"]); a[2] = float(r["bytes_per_dest"])
LABEL = {  # (text, dx, dy) in points; schemes that coincide share one label
    "FLOAT": ("FLOAT", 6, 4), "SORT-IDX": ("SORT-IDX (no persistent state)", 6, 4),
    "SORT-VAL": ("SORT-VAL (no persistent state)", 6, -10), "FIXED-Q32": ("FIXED-Q32", 6, 4),
    "DOUBLE-ACC": ("DOUBLE-ACC", -6, 9), "WINDOW-2P": ("WINDOW-2P = CAC-2x32", 0, -15),
    "CAC-2x32": None, "CAC-3x32": ("CAC-3x32", 4, 9), "SUPERACC": ("SUPERACC", -6, -14),
}
for s, (n, e, b) in agg.items():
    if s in ("SORT-IDX", "SORT-VAL"):
        b = 0.5  # no persistent per-destination state (drawn at the left edge)
    ax.scatter(b, pct(e, n), s=46, color=COLOR.get(s, "#555"), marker=MARK.get(s, "o"),
               edgecolor="white", linewidth=0.8, zorder=3)
    lab = LABEL.get(s)
    if lab:
        ax.annotate(lab[0], (b, pct(e, n)), xytext=(lab[1], lab[2]), textcoords="offset points",
                    fontsize=7, ha="right" if lab[1] < 0 else "left")
ax.set_ylim(20, 106)
ax.text(0.99, 0.04, "the gap to 100% is the uniform/overflow case (see T3a)", transform=ax.transAxes,
        ha="right", fontsize=7, color="#4a4a47")
ax.set_xscale("log")
ax.set_xlabel("persistent state per destination (bytes, log scale)")
ax.set_ylabel("% results correctly rounded")
ax.set_title("binary32, all 14 cases", fontsize=8, loc="left")
save(fig, "fig2_state_vs_accuracy")

# F3: Theorem-3 bound versus observation
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
for ax, (fmt, ss) in zip(axes, sens.items()):
    xs, ys = [], []
    exact = 0
    for r in ss:
        v = float(r["max_rel_to_max_addend"])
        if v > 0:
            xs.append(float(r["simple_bound"])); ys.append(v)
        else:
            exact += 1
    ax.scatter(xs, ys, s=7, color=COLOR["CAC-2x32"], alpha=0.55, linewidth=0)
    lo, hi = min(xs + ys), max(xs + ys)
    ax.plot([lo, hi], [lo, hi], color="#8a8a85", linewidth=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("bound  n x 2^-(K-1)W")
    ax.set_title("%s (%d of %d configs exact)" % (fmt, exact, len(ss)), fontsize=8, loc="left")
axes[0].set_ylabel("observed worst error / largest addend")
save(fig, "fig3_bound_vs_observed")

# F4: training divergence over steps, one panel per kind of variation
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
names = list(dict.fromkeys(r["scheme"] for r in tr))
for ax, v in zip(axes, ("schedule", "devices", "order")):
    for s in sorted(names, key=lambda n: (n == "FLOAT", n.startswith("CAC"))):
        xs = sorted((int(r["step"]) + 1, float(r["frac_params_differ"])) for r in tr if r["scheme"] == s and r["variation"] == v)
        hot = s in ("FLOAT", "SORT-IDX") and xs[-1][1] > 0
        ax.plot([x for x, _ in xs], [y for _, y in xs], color=COLOR.get(s, "#555"),
                linewidth=1.6 if hot or s.startswith("CAC") else 1.0, zorder=3 if s.startswith("CAC") else 2)
        if hot:
            ax.annotate(s, xs[-1], xytext=(-3, -11 if s == "SORT-IDX" else 4), textcoords="offset points", fontsize=7, ha="right")
    ax.set_title("%s varied" % v, fontsize=8, loc="left")
    ax.set_xlabel("SGD step")
    ax.text(0.02, 0.06, "all other schemes (incl. CAC): 0", transform=ax.transAxes, fontsize=6.5, color="#4a4a47")
axes[0].set_ylabel("fraction of parameters differing\nfrom replica 0")
save(fig, "fig4_training_divergence")

# F5: CAS failure vs skew
fig, ax = plt.subplots(figsize=(6.0, 3.0))
for s, comb, ls, anchor, off, ha in (("FLOAT-CAS", "0", "-", 1.0, (-8, 8), "right"),
                                     ("CAC-2x32", "0", "-", 1.0, (8, -13), "left"),
                                     ("CAC-2x32", "1", "--", 1.5, (8, -13), "left")):
    pts = {}
    for r in con:
        if r["scheme"] == s and r["combine"] == comb and r["kind"] == "window" and r["window"] == "64":
            pts[int(r["alpha_x2"]) / 2.0] = pct(int(r["cas_fails"]), int(r["cas_attempts"]))
    xs = sorted(pts)
    ax.plot(xs, [pts[x] for x in xs], ls, color=COLOR[s], marker="o", markersize=4, linewidth=1.4)
    ax.annotate(s + (" + warp combining" if comb == "1" else ""), (anchor, pts[anchor]), xytext=off,
                textcoords="offset points", fontsize=7, ha=ha)
ax.set_xlabel("Zipf exponent of the destination distribution"); ax.set_ylabel("failed CAS %")
ax.set_title("64 warps in flight", fontsize=8, loc="left")
save(fig, "fig5_cas_failures")

print("wrote", os.path.join(OUT, "tables.md"), "and figures in", FIG)
