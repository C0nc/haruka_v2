"""Two-panel metrics figure for the identifiable (data_easy) benchmark.

PANEL A -- per-quadrant recovery, grouped by quadrant, one bar per method, with a
CEILING RULE drawn per group. The ceiling is what a strong probe achieves from
raw expression with the same pooling the methods get, and it differs sharply
between quadrants (1.000 / 0.993 / 0.691 / 0.988 here). Plotting raw values
without those rules invites reading "0.67 is worse than 0.98" as a quality gap
when the first is 97% of its attainable maximum and the second is 99% of a much
higher one.

PANEL B -- assignment margin (diagonal minus off-diagonal), the headline. It is a
DIFFERENT measure on a different scale from panel A, so it gets its own panel
rather than a second y-axis on the first.

Colour: categorical slots taken in fixed order from the data-viz reference
palette, unchanged. Identity is never colour-alone -- every bar is also labelled
on the x-axis of panel B and the legend is always present in panel A.

    python plot_metrics_easy.py --out results_easy_base/metrics_easy.png
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

#: data-viz reference palette, categorical slots 1..6, light mode, fixed order
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d8d7d2"

QUADRANTS = [
    ("background", "gt_arch", "background\n-> cell type"),
    ("spatial", "gt_niche", "spatial\n-> bg domain"),
    ("salient", "gt_sal_intrinsic", "salient\n-> sal state"),
    ("salient_spatial", "gt_sal_domain", "salient_spatial\n-> sal domain"),
]


def load_rows(specs):
    rows = []
    for label, path, method in specs:
        if not os.path.exists(path):
            print(f"[skip] missing {path}")
            continue
        if path.endswith(".jsonl"):
            recs = [json.loads(l) for l in open(path)]
            rec = next((r for r in recs if method is None or r["method"] == method), None)
        else:
            df = pd.read_csv(path)
            sub = df if method is None else df[df.method == method]
            rec = sub.iloc[0].to_dict() if len(sub) else None
        if rec is None:
            print(f"[skip] {method} not in {path}")
            continue
        rows.append((label, rec))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad")
    ap.add_argument("--out", default="results_easy_base/metrics_easy.png")
    args = ap.parse_args()

    import anndata as ad
    ceil = dict(ad.read_h5ad(args.data).uns["sim2x2"].get("quadrant_ceilings", {}))
    if not ceil:      # measured values, in case the dataset predates the writer
        ceil = {"gt_arch": 1.000, "gt_niche": 0.993,
                "gt_sal_intrinsic": 0.691, "gt_sal_domain": 0.988}

    B = "results_easy_base/simv4_g0.8_r0.8_p1_d1_s0"
    specs = [
        ("Haruka (500ep)", "results_easy_ep500/metrics.csv",
         "haruka_both_sal8_timebatch_ep500_fixed_seed0"),
        ("CINEMA-OT", "results_easy_cinema/metrics_raw_primary.jsonl", None),
        ("contrastiveVI", "results_easy_base/metrics_own_gmm.csv",
         "contrastivevi_2x2_seed0"),
        ("CellCharter", "results_easy_base/metrics_own_gmm.csv",
         "cellcharter_2x2_seed0"),
        ("PCA", "results_easy_base/metrics_own_gmm.csv", "pca_2x2_seed0"),
    ]
    rows = load_rows(specs)
    if not rows:
        raise SystemExit("nothing to plot")

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(13.5, 4.6), gridspec_kw=dict(width_ratios=[2.5, 1]))

    # --- panel A: per-quadrant, grouped, with ceiling rules ------------------
    n = len(rows)
    x = np.arange(len(QUADRANTS))
    w = 0.8 / n
    for i, (label, rec) in enumerate(rows):
        vals = [rec.get(f"DEP__{r}__{t}", np.nan) for r, t, _ in QUADRANTS]
        axA.bar(x + (i - (n - 1) / 2) * w, vals, w * 0.88, label=label,
                color=PALETTE[i % len(PALETTE)], linewidth=0, zorder=3)
    for j, (_, t, _) in enumerate(QUADRANTS):
        c = ceil.get(t)
        if c:
            axA.plot([x[j] - 0.42, x[j] + 0.42], [c, c], color=INK,
                     lw=1.6, ls=(0, (4, 2)), zorder=4,
                     label="ceiling" if j == 0 else None)
            axA.text(x[j] + 0.44, c, f"{c:.2f}", va="center", ha="left",
                     fontsize=7.5, color=INK2)
    axA.set_xticks(x)
    axA.set_xticklabels([lab for _, _, lab in QUADRANTS], fontsize=9)
    axA.set_ylabel("dependence", fontsize=10, color=INK2)
    axA.set_ylim(0, 1.12)
    axA.set_title("Per-quadrant recovery vs attainable ceiling",
                  fontsize=11, color=INK, loc="left")
    axA.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, -0.16))
    axA.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    axA.set_axisbelow(True)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axA.spines[s].set_color(GRID)

    # --- panel B: the headline, its own scale --------------------------------
    labels = [l for l, _ in rows]
    marg = [r.get("assignment_margin", np.nan) for _, r in rows]
    order = np.argsort(marg)
    axB.barh(np.arange(len(rows)), [marg[i] for i in order], 0.62,
             color=[PALETTE[i % len(PALETTE)] for i in order],
             linewidth=0, zorder=3)
    for k, i in enumerate(order):
        axB.text(marg[i] + 0.008, k, f"{marg[i]:.3f}", va="center",
                 fontsize=8.5, color=INK)
    axB.set_yticks(np.arange(len(rows)))
    axB.set_yticklabels([labels[i] for i in order], fontsize=9)
    axB.set_xlabel("assignment margin  (diag - off-diag)", fontsize=9.5, color=INK2)
    axB.set_xlim(0, max(marg) * 1.22)
    axB.set_title("Assignment margin", fontsize=11, color=INK, loc="left")
    axB.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    axB.set_axisbelow(True)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axB.spines[s].set_color(GRID)

    fig.suptitle("Identifiable non-linear benchmark  (data_easy: gate, "
                 "cosine -0.035, delta_sp=1.0)", fontsize=12, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=170, facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
