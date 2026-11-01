"""Aggregate `metrics.jsonl` into the tables and the figure.

The headline is deliberately NOT a single leaderboard number. A method here has
to do two things at once -- find the implanted domain in its salient
representation AND keep the tissue architecture in its background representation
-- and the interesting result is where a method trades one for the other. So the
console output is three blocks:

  DETECTION      salient-side recovery of the implant, as a function of effect size
  SPECIFICITY    what the same numbers look like when there is nothing to find
                 (delta = 0) and how much anatomy contaminates the salient side
  BACKGROUND     whether the background representation is the tissue map it claims
                 to be, and whether condition leaked into it

    python report.py --metrics results/metrics.jsonl --fig results/benchmark.png
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

ORDER = ["pca", "cellcharter", "cinemaot", "contrastivevi", "contrastivevi_cc",
         "haruka_vanilla", "haruka_background", "haruka_both", "haruka_salient",
         "haruka_niche"]


def load(path: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows)
    # Later runs of the same cell overwrite earlier ones: the file is append-only
    # so a re-run leaves both, and the fresher one is the one that counts.
    df = df.drop_duplicates(subset=["method", "data", "score_seed"], keep="last")
    df["method"] = pd.Categorical(
        df["method"], categories=[m for m in ORDER if m in set(df["method"])] +
        sorted(set(df["method"]) - set(ORDER)), ordered=True)
    return df.sort_values(["method", "delta"])


def _pivot(df, value, aggfunc="mean"):
    return df.pivot_table(index="method", columns="delta", values=value,
                          aggfunc=aggfunc, observed=True)


def add_null_increments(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract each method's OWN delta=0 score from its scores at every delta.

    THIS IS THE HEADLINE TRANSFORM, and the reason the delta=0 arm exists.

    Raw `sal_AUROC_domain` is not comparable across methods and is not a detection
    rate. On the null build -- where no effect was injected at all -- CellCharter
    scores 0.673, contrastiveVI+CellCharter 0.618 and Haruka 0.602 against the
    counterfactual domain label. None of that is detection. The implanted domain
    is a smooth field defined in a coordinate frame SHARED across slices (it has
    to be, or it would be per-slice noise no method could be asked to find), so
    tissue position alone predicts the label -- and it generalises across animals,
    which is exactly what leave-one-animal-out grouping cannot protect against.
    Any representation carrying positional information, which 3-hop neighbourhood
    aggregation supplies in abundance, clears 0.5 for free.

    So each method's null score is its own floor, and what it detected is the
    increment above that floor. A method with a high floor is not cheating; it is
    simply spatially smooth, and the increment is what remains after that is
    accounted for.
    """
    keys = ["method", "sim_seed", "score_seed", "delta_mode"]
    keys = [k for k in keys if k in df.columns]
    null = df[df["delta"] == 0].set_index(keys)
    metric_cols = [c for c in df.columns
                   if c.startswith(("sal_", "bg_")) and df[c].dtype.kind == "f"]
    out = df.copy()
    for c in metric_cols:
        base = out.set_index(keys).index.map(null[c]) if len(null) else np.nan
        out[f"{c}_over_null"] = out[c].to_numpy() - np.asarray(base, dtype=float)
    return out


def report(df: pd.DataFrame) -> str:
    out = []
    n_seeds = df["sim_seed"].nunique()
    out.append(f"n sim seeds = {n_seeds}; cells = {int(df['n_cells'].max())}; "
               f"deltas = {sorted(df['delta'].unique())}")

    out.append("\n" + "=" * 78)
    out.append("DETECTION -- can the salient representation see the implanted domain?")
    out.append("=" * 78)
    if "sal_AUROC_domain_over_null" in df:
        out.append("\n>>> sal_AUROC_domain OVER ITS OWN delta=0 FLOOR  <<< the headline number")
        out.append("    (raw AUROC is not a detection rate here: the domain is a smooth field in")
        out.append("     a frame shared across slices, so position alone predicts it and any")
        out.append("     spatially smoothed representation clears 0.5 for free. See the floors")
        out.append("     in the SPECIFICITY block.)")
        out.append(_pivot(df[df["delta"] > 0], "sal_AUROC_domain_over_null").round(3).to_string())
    out.append("\nsal_AUROC_domain  (raw; leave-one-animal-out linear probe)")
    out.append(_pivot(df, "sal_AUROC_domain").round(3).to_string())
    out.append("\nsal_ARI_domain  (PCA-50 + full-cov GMM, k=3, vs the true domain labels)")
    out.append(_pivot(df, "sal_ARI_domain").round(3).to_string())

    out.append("\n" + "=" * 78)
    out.append("SPECIFICITY -- what happens when there is nothing to find, and is the")
    out.append("               salient side just a second copy of the anatomy?")
    out.append("=" * 78)
    null = df[df["delta"] == 0]
    if len(null):
        cols = ["sal_AUROC_domain", "sal_ARI_domain", "sal_ARI_anat"]
        out.append("\ndelta = 0 -- THE FLOOR each method must be read against.")
        out.append("sal_AUROC_domain here is what the method scores with NOTHING injected;")
        out.append("it measures spatial smoothness, not false detection.")
        out.append(null.groupby("method", observed=True)[cols].mean().round(3).to_string())
    out.append("\nsal_ARI_anat  (contamination: salient recovering the 7 subregions)")
    out.append(_pivot(df, "sal_ARI_anat").round(3).to_string())

    out.append("\n" + "=" * 78)
    out.append("BACKGROUND -- is it the tissue map, and is it free of the condition?")
    out.append("=" * 78)
    bgcols = {"bg_ARI_anat": "anat[HIGH]",
              "bg_ARI_condition": "cond[~0]",
              "bg_ARI_slice": "animal[low]",
              "bg_AUROC_domain": "domain[~.5]"}
    bg = df.groupby("method", observed=True)[list(bgcols)].mean().round(3)
    bg.columns = list(bgcols.values())
    out.append("\n(background ARI vs: anat=7 subregions, cond=control/disease, "
               "animal=8 slices; domain=probe AUROC)")
    out.append(bg.to_string())

    # The Haruka-v2 README's central methodological point is that this outcome is
    # BIMODAL across seeds -- the background latent either separates architecture
    # from condition or collapses, with little in between -- so a cross-seed mean
    # is a number that describes no actual run. Print the seeds individually at the
    # largest delta and let the reader see the spread.
    if n_seeds > 1:
        dmax = df["delta"].max()
        out.append("\n" + "=" * 78)
        out.append(f"PER-SEED SPREAD at delta = {dmax} (the mean is only meaningful if these agree)")
        out.append("=" * 78)
        for col in ("sal_AUROC_domain", "sal_ARI_domain", "bg_ARI_anat", "bg_ARI_condition"):
            if col not in df:
                continue
            piv = df[df["delta"] == dmax].pivot_table(
                index="method", columns="sim_seed", values=col, observed=True)
            out.append(f"\n{col}")
            out.append(piv.round(3).to_string())

    if "runtime_s" in df:
        rt = df.groupby("method", observed=True)["runtime_s"].median()
        out.append("\nmedian runtime (s): " +
                   ", ".join(f"{m}={v:.0f}" for m, v in rt.dropna().items()))
    return "\n".join(out)


def figure(df: pd.DataFrame, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [m for m in df["method"].cat.categories if m in set(df["method"])]
    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(methods)}

    panels = [
        ("sal_AUROC_domain_over_null", "Detection: salient AUROC above own delta=0 floor", 0.0),
        ("sal_AUROC_domain", "Salient probe AUROC, raw (floor is method-specific)", 0.5),
        ("sal_ARI_domain", "Salient GMM ARI vs implanted domain", 0.0),
        ("bg_ARI_anat", "Background GMM ARI vs anatomy (subregion)", None),
        ("bg_ARI_condition", "Background GMM ARI vs condition (leak; want 0)", 0.0),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.2))
    for ax, (col, title, ref) in zip(axes, panels):
        if col not in df:
            ax.set_visible(False)
            continue
        for m in methods:
            s = df[df["method"] == m].groupby("delta", observed=True)[col]
            mu, sd = s.mean(), s.std()
            ax.plot(mu.index, mu.values, "o-", color=colors[m], label=m, lw=1.8, ms=5)
            if sd.notna().any():
                ax.fill_between(mu.index, mu - sd.fillna(0), mu + sd.fillna(0),
                                color=colors[m], alpha=0.12, lw=0)
        if ref is not None:
            ax.axhline(ref, color="0.4", ls="--", lw=1)
        ax.set_xlabel("effect size  delta")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("score")
    axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"[report] wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/metrics.jsonl")
    ap.add_argument("--fig", default="results/benchmark.png")
    ap.add_argument("--csv", default="results/metrics.csv")
    args = ap.parse_args()

    df = add_null_increments(load(args.metrics))
    df.to_csv(args.csv, index=False)
    text = report(df)
    print(text)
    with open(args.metrics.replace(".jsonl", "_report.txt"), "w") as fh:
        fh.write(text + "\n")
    try:
        figure(df, args.fig)
    except Exception as e:                                # pragma: no cover
        print(f"[report] figure skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
