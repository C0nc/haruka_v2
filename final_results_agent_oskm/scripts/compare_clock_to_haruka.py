#!/usr/bin/env python3
"""Compare the independent frozen Spatial Aging Clock with Haruka responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, average_precision_score, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def heldout_ridge(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    valid = np.isfinite(y)
    x, y, groups = x[valid], y[valid], groups[valid]
    pred = np.full(len(y), np.nan)
    fold = []
    for held in np.unique(groups):
        train, test = groups != held, groups == held
        scaler = StandardScaler().fit(x[train])
        model = Ridge(alpha=100.0, solver="lsqr", tol=1e-4).fit(scaler.transform(x[train]), y[train])
        pred[test] = model.predict(scaler.transform(x[test]))
        fold.append({"held_out_pair": held, "r2": float(r2_score(y[test], pred[test])), "spearman": float(stats.spearmanr(y[test], pred[test]).statistic), "n": int(test.sum())})
    return {"n": int(len(y)), "r2": float(r2_score(y, pred)), "spearman": float(stats.spearmanr(y, pred).statistic), "fold_r2_mean": float(np.mean([v["r2"] for v in fold])), "fold_spearman_mean": float(np.mean([v["spearman"] for v in fold])), "folds": fold}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clock", required=True, type=Path)
    p.add_argument("--scores", required=True, type=Path)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--cc-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    figures = args.out.parent / "figures"
    figures.mkdir(exist_ok=True)

    clock_all = pd.read_csv(args.clock)
    clock = clock_all[clock_all["condition"].astype(str) == "oskm"].copy()
    score = pd.read_csv(args.scores)
    target_position = pd.Index(score["cell_id"].astype(str)).get_indexer(clock["cell_id"].astype(str))
    if (target_position < 0).any() or len(np.unique(target_position)) != len(target_position):
        raise ValueError("Clock subset does not map one-to-one into full-cohort Haruka OSKM rows")
    frame = score.iloc[target_position][["cell_id", "paired_batch", "slice_id", "cell_type", "subregion", "local_score_unscaled", "pair_adjusted_heterogeneity", "responder_main_q95"]].copy().reset_index(drop=True)
    if not np.array_equal(frame["cell_id"].astype(str).to_numpy(), clock["cell_id"].astype(str).to_numpy()):
        raise ValueError("Mapped clock and Haruka rows do not align")
    frame["clock_rejuvenation"] = clock["clock_rejuvenation"].to_numpy(float)
    frame["predicted_age"] = clock["predicted_age"].to_numpy(float)
    frame["clock_pair_adjusted"] = frame["clock_rejuvenation"] - frame.groupby(["paired_batch", "cell_type", "subregion"], observed=True)["clock_rejuvenation"].transform("mean")

    valid = frame[["clock_rejuvenation", "local_score_unscaled"]].notna().all(axis=1)
    raw_spearman = float(stats.spearmanr(frame.loc[valid, "clock_rejuvenation"], frame.loc[valid, "local_score_unscaled"]).statistic)
    adjusted_valid = frame[["clock_pair_adjusted", "pair_adjusted_heterogeneity"]].notna().all(axis=1)
    adjusted_spearman = float(stats.spearmanr(frame.loc[adjusted_valid, "clock_pair_adjusted"], frame.loc[adjusted_valid, "pair_adjusted_heterogeneity"]).statistic)
    pair_corr = []
    for pair, block in frame.groupby("paired_batch", observed=True):
        ok = block[["clock_pair_adjusted", "pair_adjusted_heterogeneity"]].notna().all(axis=1)
        pair_corr.append({"paired_batch": pair, "n": int(ok.sum()), "spearman_adjusted": float(stats.spearmanr(block.loc[ok, "clock_pair_adjusted"], block.loc[ok, "pair_adjusted_heterogeneity"]).statistic)})

    # Clock responder calls are independently thresholded by the matched old-control null.
    control = clock_all[clock_all["condition"].astype(str) == "control"].copy()
    control["clock_rejuvenation"] = control.groupby(
        ["paired_batch", "cell_type", "subregion"], observed=True
    )["predicted_age"].transform("median") - control["predicted_age"]
    thresholds = control.groupby(["paired_batch", "cell_type", "subregion"], observed=True)["clock_rejuvenation"].quantile(0.95).rename("clock_threshold")
    frame = frame.join(thresholds, on=["paired_batch", "cell_type", "subregion"])
    frame["clock_responder_q95"] = frame["clock_rejuvenation"] > frame["clock_threshold"]
    comparable = frame[["clock_responder_q95", "responder_main_q95", "local_score_unscaled", "clock_rejuvenation"]].notna().all(axis=1)
    y = frame.loc[comparable, "clock_responder_q95"].astype(int).to_numpy()
    h = frame.loc[comparable, "responder_main_q95"].astype(int).to_numpy()
    metrics = {
        "n_clock_scored_oskm": int(frame["clock_rejuvenation"].notna().sum()),
        "clock_evaluation_subset": "existing completed deterministic 160k old cohort cache (80k OSKM), mapped by unique cell_id into the 684253-cell full cohort",
        "raw_spearman_clock_vs_haruka": raw_spearman,
        "pair_celltype_subregion_adjusted_spearman": adjusted_spearman,
        "pair_specific_adjusted_correlations": pair_corr,
        "clock_responder_rate": float(y.mean()),
        "haruka_responder_rate_on_clocked_cells": float(h.mean()),
        "responder_jaccard": float(np.sum((y == 1) & (h == 1)) / max(np.sum((y == 1) | (h == 1)), 1)),
        "responder_adjusted_rand_index": float(adjusted_rand_score(y, h)),
        "haruka_score_auroc_for_clock_responder": float(roc_auc_score(y, frame.loc[comparable, "local_score_unscaled"])),
        "haruka_score_auprc_for_clock_responder": float(average_precision_score(y, frame.loc[comparable, "local_score_unscaled"])),
        "interpretation": "Clock concordance is independent endpoint concordance; neither score establishes lineage or causal rejuvenation.",
    }

    source_rows = np.load(args.cc_dir / "source_rows.npy").astype(np.int64)
    reps = {
        "z": np.load(args.model_dir / "background.npy", mmap_mode="r")[source_rows][target_position],
        "s": np.load(args.model_dir / "salient.npy", mmap_mode="r")[source_rows][target_position],
        "z_cc": np.load(args.cc_dir / "spatial.npy", mmap_mode="r")[target_position],
        "s_cc": np.load(args.cc_dir / "salient_spatial.npy", mmap_mode="r")[target_position],
    }
    groups = frame["paired_batch"].astype(str).to_numpy()
    rep_rows = []
    for name, rep in reps.items():
        for endpoint in ["clock_rejuvenation", "clock_pair_adjusted"]:
            result = heldout_ridge(np.asarray(rep, dtype=np.float32), frame[endpoint].to_numpy(float), groups)
            folds = result.pop("folds")
            rep_rows.append({"representation": name, "endpoint": endpoint, **result})
            for item in folds:
                item.update({"representation": name, "endpoint": endpoint})
            pd.DataFrame(folds).to_csv(args.out / f"clock_{endpoint}_{name}_folds.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(args.out / "clock_representation_heldout_metrics.csv", index=False)
    frame.to_csv(args.out / "clock_haruka_cell_comparison.csv.gz", index=False, compression="gzip")
    (args.out / "clock_haruka_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    rng = np.random.default_rng(0)
    rows = np.flatnonzero(valid.to_numpy())
    if len(rows) > 80_000:
        rows = np.sort(rng.choice(rows, 80_000, replace=False))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].hexbin(frame.iloc[rows]["local_score_unscaled"], frame.iloc[rows]["clock_rejuvenation"], gridsize=70, mincnt=1, bins="log", cmap="viridis")
    axes[0].set_xlabel("Haruka-associated matched youngward score"); axes[0].set_ylabel("Clock months younger"); axes[0].set_title(f"Raw concordance, Spearman={raw_spearman:.3f}")
    okrows = np.flatnonzero(adjusted_valid.to_numpy())
    if len(okrows) > 80_000:
        okrows = np.sort(rng.choice(okrows, 80_000, replace=False))
    axes[1].hexbin(frame.iloc[okrows]["pair_adjusted_heterogeneity"], frame.iloc[okrows]["clock_pair_adjusted"], gridsize=70, mincnt=1, bins="log", cmap="magma")
    axes[1].set_xlabel("Haruka pair/anatomy-adjusted response"); axes[1].set_ylabel("Clock pair/anatomy-adjusted shift"); axes[1].set_title(f"Adjusted concordance, Spearman={adjusted_spearman:.3f}")
    fig.savefig(figures / "spatial_clock_vs_haruka_response.png", dpi=220)
    plt.close(fig)

    plot = pd.DataFrame(rep_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    sns.barplot(data=plot, x="representation", y="spearman", hue="endpoint", ax=axes[0])
    sns.barplot(data=plot, x="representation", y="r2", hue="endpoint", ax=axes[1])
    for ax in axes: ax.axhline(0, color="black", lw=0.7); ax.tick_params(axis="x", rotation=25)
    axes[0].set_title("Clock endpoint: leave-pair-out Spearman"); axes[1].set_title("Clock endpoint: leave-pair-out R²")
    fig.savefig(figures / "spatial_clock_representation_comparison.png", dpi=220)
    plt.close(fig)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
