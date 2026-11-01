"""Render the 2x2 benchmark: one representation-by-truth matrix per method.

The output is deliberately not a leaderboard. The claim under test is an
ASSIGNMENT -- each of Haruka's four latents should carry its own quadrant and
decline the other three -- and the only way to see whether that happened is to
look at the matrix. A method can post a strong diagonal purely by putting
everything in every latent, and the off-diagonal is what exposes it.

Blocks printed:

  MATRIX          per method, rows = representation, cols = truth, cells = ARI.
                  The diagonal is marked; read down a column to see which latents
                  are competing for the same factor.
  ASSIGNMENT      diag_mean, offdiag_mean and their difference, by method and
                  effect size. This is the headline.
  QUADRANT        each quadrant's diagonal entry across methods, so it is visible
                  WHICH factor a method fails on, not just that it failed.
  PER-SEED        the same at the largest delta, seed by seed, because the outcome
                  is bimodal and a mean over modes describes no run.

    python report_2x2.py --metrics results_2x2/metrics.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from score_2x2 import ASSIGNMENT, TRUTHS

ORDER = ["pca", "cellcharter", "cinemaot", "contrastivevi", "contrastivevi_cc",
         "haruka_vanilla", "haruka_background", "haruka_both"]
REPS = ["background", "spatial", "salient", "salient_spatial"]
SHORT = {"gt_arch": "arch", "gt_niche": "niche",
         "gt_sal_intrinsic": "sal_intr", "gt_sal_domain": "sal_dom"}


def load(paths) -> pd.DataFrame:
    rows = []
    for p in paths:
        with open(p) as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["method", "data", "score_seed"], keep="last")
    cats = [m for m in ORDER if m in set(df["method"])] + \
           sorted(set(df["method"]) - set(ORDER))
    df["method"] = pd.Categorical(df["method"], categories=cats, ordered=True)
    return df.sort_values(["method", "delta", "sim_seed"])


def matrix_block(df: pd.DataFrame, delta: float) -> str:
    out = [f"\nrepresentation x truth ARI at delta = {delta}"
           f"  (* = the quadrant that representation is responsible for)"]
    sub = df[df["delta"] == delta]
    for method in sub["method"].cat.categories:
        m = sub[sub["method"] == method]
        if not len(m):
            continue
        out.append(f"\n  {method}")
        header = "    " + "".join(f"{SHORT[t]:>12}" for t in TRUTHS)
        out.append(header)
        for r in REPS:
            cells = []
            present = False
            for t in TRUTHS:
                col = f"ARI__{r}__{t}"
                if col not in m or m[col].isna().all():
                    cells.append(f"{'-':>12}")
                    continue
                present = True
                v = m[col].mean()
                mark = "*" if ASSIGNMENT.get(r) == t else " "
                cells.append(f"{v:>11.3f}{mark}")
            if present:
                out.append(f"    {r:<16}" + "".join(cells))
    return "\n".join(out)


def report(df: pd.DataFrame) -> str:
    out = [f"n sim seeds = {df['sim_seed'].nunique()}; "
           f"cells = {int(df['n_cells'].max())}; "
           f"deltas = {sorted(df['delta'].unique())}"]

    out.append("\n" + "=" * 78)
    out.append("ASSIGNMENT -- does each latent own its quadrant AND decline the rest?")
    out.append("=" * 78)
    out.append("\nCAUTION: `diag_mean` averages over however many latents a method HAS, so")
    out.append("the margin is not apples-to-apples across methods with different latent")
    out.append("counts. contrastiveVI has 2 latents and is scored on 2 diagonal cells --")
    out.append("the two EASIEST ones (anatomy, and a cell-intrinsic state) -- while")
    out.append("haruka_both is scored on 4, including the two spatial quadrants that")
    out.append("nothing else even attempts. A method is not better for claiming less.")
    out.append("Read the QUADRANT block below for the like-for-like comparison: it comes")
    out.append("from the same cell of the matrix for every method that fills it.")
    for col, label in (("diag_mean", "diagonal (want HIGH)"),
                       ("offdiag_mean", "off-diagonal (want LOW)"),
                       ("assignment_margin", ">>> MARGIN = diag - offdiag <<<")):
        if col not in df:
            continue
        out.append(f"\n{label}")
        out.append(df.pivot_table(index="method", columns="delta", values=col,
                                  observed=True).round(3).to_string())

    out.append("\n" + "=" * 78)
    out.append("QUADRANT -- which factor does each method actually get?")
    out.append("=" * 78)
    for rep, truth in ASSIGNMENT.items():
        col = f"ARI__{rep}__{truth}"
        if col not in df or df[col].isna().all():
            continue
        out.append(f"\n{rep} -> {SHORT[truth]}")
        out.append(df.pivot_table(index="method", columns="delta", values=col,
                                  observed=True).round(3).to_string())

    out.append("\n" + "=" * 78)
    out.append("LEAKS -- background must not carry condition, animal, or the niche")
    out.append("=" * 78)
    leak = [c for c in ("bg_ARI_condition", "bg_ARI_slice", "niche_in_intrinsic_leak")
            if c in df]
    if leak:
        out.append("\n(niche_in_intrinsic_leak: a cell's own row is independent of its own")
        out.append(" niche by construction, so an INTRINSIC latent recovering gt_niche is")
        out.append(" evidence the spatial branch bled into it, not a success)")
        out.append(df.groupby("method", observed=True)[leak].mean().round(3).to_string())

    dmax = df["delta"].max()
    out.append("\n" + "=" * 78)
    out.append(f"MATRICES at the largest effect size (delta = {dmax})")
    out.append("=" * 78)
    out.append(matrix_block(df, dmax))

    if df["sim_seed"].nunique() > 1 and "assignment_margin" in df:
        out.append("\n" + "=" * 78)
        out.append(f"PER-SEED assignment margin at delta = {dmax}")
        out.append("(the outcome is bimodal by construction of the model, not of the")
        out.append(" benchmark; a mean across modes describes no actual run)")
        out.append("=" * 78)
        piv = df[df["delta"] == dmax].pivot_table(
            index="method", columns="sim_seed", values="assignment_margin",
            observed=True)
        out.append("\n" + piv.round(3).to_string())

    if "runtime_s" in df:
        rt = df.groupby("method", observed=True)["runtime_s"].median().dropna()
        out.append("\nmedian runtime (s): " + ", ".join(f"{m}={v:.0f}" for m, v in rt.items()))
    return "\n".join(out)


def figure(df: pd.DataFrame, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [m for m in df["method"].cat.categories if m in set(df["method"])]
    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(methods)}
    panels = [("assignment_margin", "Assignment margin (diag - offdiag)")] + \
             [(f"ARI__{r}__{t}", f"{r} -> {SHORT[t]}") for r, t in ASSIGNMENT.items()]
    panels = [(c, t) for c, t in panels if c in df and not df[c].isna().all()]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.0),
                             squeeze=False)
    for ax, (col, title) in zip(axes[0], panels):
        for m in methods:
            s = df[df["method"] == m].groupby("delta", observed=True)[col]
            mu, sd = s.mean(), s.std()
            if mu.isna().all():
                continue
            ax.plot(mu.index, mu.values, "o-", color=colors[m], label=m, lw=1.8, ms=5)
            if sd.notna().any():
                ax.fill_between(mu.index, mu - sd.fillna(0), mu + sd.fillna(0),
                                color=colors[m], alpha=0.12, lw=0)
        ax.axhline(0, color="0.4", ls="--", lw=1)
        ax.set_xlabel("effect size  delta")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("ARI")
    axes[0][-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"[report2x2] wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="*", default=None)
    ap.add_argument("--results_dir", default="results_2x2")
    ap.add_argument("--fig", default="results_2x2/benchmark_2x2.png")
    ap.add_argument("--csv", default="results_2x2/metrics_2x2.csv")
    args = ap.parse_args()

    paths = args.metrics or sorted(glob.glob(os.path.join(args.results_dir,
                                                          "metrics_*.jsonl")))
    if not paths:
        raise SystemExit(f"no metrics files under {args.results_dir}")
    df = load(paths)
    df.to_csv(args.csv, index=False)
    text = report(df)
    print(text)
    with open(os.path.join(args.results_dir, "report_2x2.txt"), "w") as fh:
        fh.write(text + "\n")
    try:
        figure(df, args.fig)
    except Exception as e:                                 # pragma: no cover
        print(f"[report2x2] figure skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
