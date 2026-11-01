#!/usr/bin/env python3
"""Pair-aware associations of the published Spatial Aging Clock with Haruka CC."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, normalized_mutual_info_score, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


SEED = 20260913


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def within_rank(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    d = pd.DataFrame({"v": values, "g": groups})
    # Percentile ranks are essential when strata differ in size; raw ranks
    # otherwise share a group-size scale and create a positive pooled artifact.
    return d.groupby("g", observed=True)["v"].rank(method="average", pct=True).to_numpy(float)


def rho(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    ok = np.isfinite(x) & np.isfinite(y)
    out = stats.spearmanr(x[ok], y[ok])
    return float(out.statistic), float(out.pvalue), int(ok.sum())


def block_bootstrap_corr(x: np.ndarray, y: np.ndarray, blocks: np.ndarray,
                         slices: np.ndarray, n_boot: int = 500) -> tuple[float, float]:
    """Bootstrap spatial blocks, preserving each slice and using sufficient moments."""
    ok = np.isfinite(x) & np.isfinite(y)
    d = pd.DataFrame({"x": x[ok], "y": y[ok], "block": blocks[ok], "slice": slices[ok]})
    d["xx"], d["yy"], d["xy"] = d.x * d.x, d.y * d.y, d.x * d.y
    m = d.groupby(["slice", "block"], observed=True).agg(
        n=("x", "size"), sx=("x", "sum"), sy=("y", "sum"),
        sxx=("xx", "sum"), syy=("yy", "sum"), sxy=("xy", "sum")).reset_index()
    by_slice = {s: g[["n", "sx", "sy", "sxx", "syy", "sxy"]].to_numpy(float)
                for s, g in m.groupby("slice", observed=True)}
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n_boot):
        total = np.zeros(6, dtype=float)
        for arr in by_slice.values():
            total += arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)
        n, sx, sy, sxx, syy, sxy = total
        cov = sxy - sx * sy / n
        vx, vy = sxx - sx * sx / n, syy - sy * sy / n
        vals.append(cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan)
    return tuple(np.nanpercentile(vals, [2.5, 97.5]).astype(float))


def make_spatial_blocks(frame: pd.DataFrame, bins: int = 8) -> np.ndarray:
    out = np.empty(len(frame), dtype=object)
    for sample, idx in frame.groupby("slice_id", observed=True).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        n = len(idx)
        xr = stats.rankdata(frame.loc[idx, "center_x"], method="ordinal")
        yr = stats.rankdata(frame.loc[idx, "center_y"], method="ordinal")
        xb = np.minimum((xr * bins / (n + 1)).astype(int), bins - 1)
        yb = np.minimum((yr * bins / (n + 1)).astype(int), bins - 1)
        out[idx] = np.array([f"{sample}:{x}:{y}" for x, y in zip(xb, yb)], dtype=object)
    return out.astype(str)


def pc_associations(frame: pd.DataFrame, pcs: np.ndarray, representation: str,
                    blocks: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, slice_rows = [], []
    slice_group = frame["slice_id"].astype(str).to_numpy()
    anatomy_group = (frame["slice_id"].astype(str) + "|" + frame["cell_type"].astype(str)
                     + "|" + frame["subregion"].astype(str)).to_numpy()
    age = frame["predicted_age"].to_numpy(float)
    age_slice_rank = within_rank(age, slice_group)
    age_anat_rank = within_rank(age, anatomy_group)
    for j in range(10):
        x = pcs[:, j]
        raw_r, raw_p, n = rho(x, age)
        xr = within_rank(x, slice_group)
        wr, wp = stats.pearsonr(xr, age_slice_rank)
        xa = within_rank(x, anatomy_group)
        ar, ap = stats.pearsonr(xa, age_anat_rank)
        lo, hi = block_bootstrap_corr(xr, age_slice_rank, blocks, slice_group)
        directions = []
        for sample, idx in frame.groupby("slice_id", observed=True).groups.items():
            idx = np.asarray(list(idx), dtype=int)
            sr, sp, sn = rho(x[idx], age[idx])
            directions.append(np.sign(sr))
            slice_rows.append({"representation": representation, "pc": j + 1,
                               "slice_id": sample, "n": sn, "spearman": sr, "pvalue": sp})
        rows.append({"representation": representation, "pc": j + 1, "n": n,
                     "raw_spearman": raw_r, "raw_pvalue": raw_p,
                     "within_slice_spearman": float(wr), "within_slice_pvalue": float(wp),
                     "within_slice_block_boot_ci_low": lo, "within_slice_block_boot_ci_high": hi,
                     "within_slice_celltype_subregion_spearman": float(ar),
                     "within_slice_celltype_subregion_pvalue": float(ap),
                     "positive_slices": int(np.sum(np.asarray(directions) > 0)),
                     "negative_slices": int(np.sum(np.asarray(directions) < 0))})
    return pd.DataFrame(rows), pd.DataFrame(slice_rows)


def fitted_pcs(x: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(x[train])
    pca = PCA(n_components=10, svd_solver="randomized", random_state=0).fit(scaler.transform(x[train]))
    return pca.transform(scaler.transform(x[train])), pca.transform(scaler.transform(x[test]))


def prediction_metrics(frame: pd.DataFrame, rep: np.ndarray, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = frame["paired_batch"].astype(str).to_numpy()
    endpoints = ["predicted_age", "predicted_age_slice_ct_subregion_centered"]
    fold_rows, summary = [], []
    preds = {e: np.full(len(frame), np.nan) for e in endpoints}
    for held in sorted(np.unique(groups)):
        train, test = groups != held, groups == held
        xtr, xte = fitted_pcs(rep, train, test)
        for endpoint in endpoints:
            y = frame[endpoint].to_numpy(float)
            model = Ridge(alpha=100.0, solver="lsqr", tol=1e-5).fit(xtr, y[train])
            pr = model.predict(xte)
            preds[endpoint][test] = pr
            fold_rows.append({"representation": name, "split": "leave_pair_out",
                              "held_out_pair": held, "endpoint": endpoint, "n": int(test.sum()),
                              "r2": float(r2_score(y[test], pr)),
                              "spearman": float(stats.spearmanr(y[test], pr).statistic),
                              "mae": float(mean_absolute_error(y[test], pr))})
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(frame))
    test = np.zeros(len(frame), bool); test[order[:len(frame) // 5]] = True
    train = ~test
    xtr, xte = fitted_pcs(rep, train, test)
    for endpoint in endpoints:
        y = frame[endpoint].to_numpy(float)
        model = Ridge(alpha=100.0, solver="lsqr", tol=1e-5).fit(xtr, y[train])
        pr = model.predict(xte)
        fold_rows.append({"representation": name, "split": "random_cell_80_20_leaky",
                          "held_out_pair": "random_cells", "endpoint": endpoint, "n": int(test.sum()),
                          "r2": float(r2_score(y[test], pr)),
                          "spearman": float(stats.spearmanr(y[test], pr).statistic),
                          "mae": float(mean_absolute_error(y[test], pr))})
    fold = pd.DataFrame(fold_rows)
    for endpoint in endpoints:
        y, pr = frame[endpoint].to_numpy(float), preds[endpoint]
        random_row = fold[(fold.representation == name) & (fold.endpoint == endpoint)
                          & (fold.split == "random_cell_80_20_leaky")].iloc[0]
        pair_fold = fold[(fold.representation == name) & (fold.endpoint == endpoint)
                         & (fold.split == "leave_pair_out")]
        summary.append({"representation": name, "endpoint": endpoint, "n": len(y),
                        "held_pair_r2": float(r2_score(y, pr)),
                        "held_pair_spearman": float(stats.spearmanr(y, pr).statistic),
                        "held_pair_mae": float(mean_absolute_error(y, pr)),
                        "mean_fold_r2": float(pair_fold.r2.mean()),
                        "direction_consistent_folds": int((pair_fold.spearman > 0).sum()),
                        "random_cell_r2": float(random_row.r2),
                        "random_cell_spearman": float(random_row.spearman),
                        "r2_leakage_inflation": float(random_row.r2 - r2_score(y, pr))})
    return pd.DataFrame(summary), fold


def dpt_associations(frame: pd.DataFrame, blocks: np.ndarray) -> pd.DataFrame:
    rows = []
    for endpoint in ["dpt_min_pc1", "dpt_max_pc1"]:
        directions = []
        for sample, idx in frame.groupby("slice_id", observed=True).groups.items():
            idx = np.asarray(list(idx), dtype=int)
            r, p, n = rho(frame.loc[idx, endpoint].to_numpy(float), frame.loc[idx, "predicted_age"].to_numpy(float))
            directions.append(np.sign(r))
            rows.append({"endpoint": endpoint, "scope": "slice", "slice_id": sample,
                         "n": n, "spearman": r, "pvalue": p, "ci_low": np.nan, "ci_high": np.nan})
        group = frame["slice_id"].astype(str).to_numpy()
        x = within_rank(frame[endpoint].to_numpy(float), group)
        y = within_rank(frame["predicted_age"].to_numpy(float), group)
        r, p = stats.pearsonr(x, y)
        lo, hi = block_bootstrap_corr(x, y, blocks, group)
        rows.append({"endpoint": endpoint, "scope": "pooled_within_slice", "slice_id": "all",
                     "n": len(frame), "spearman": float(r), "pvalue": float(p), "ci_low": lo,
                     "ci_high": hi, "positive_slices": int(np.sum(np.asarray(directions) > 0)),
                     "negative_slices": int(np.sum(np.asarray(directions) < 0))})
        anat = (frame["slice_id"].astype(str) + "|" + frame["cell_type"].astype(str)
                + "|" + frame["subregion"].astype(str)).to_numpy()
        xa = within_rank(frame[endpoint].to_numpy(float), anat)
        ya = within_rank(frame["predicted_age"].to_numpy(float), anat)
        ar, ap = stats.pearsonr(xa, ya)
        rows.append({"endpoint": endpoint, "scope": "pooled_within_slice_celltype_subregion",
                     "slice_id": "all", "n": len(frame), "spearman": float(ar), "pvalue": float(ap),
                     "ci_low": np.nan, "ci_high": np.nan})
    out = pd.DataFrame(rows)
    out["fdr_bh"] = multipletests(out.pvalue.fillna(1), method="fdr_bh")[1]
    return out


def paired_clock_summary(clock: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = clock.groupby(["paired_batch", "condition", "cell_type"], observed=True).predicted_age.agg(["median", "size"]).reset_index()
    rows = []
    for ct in sorted(clock.cell_type.astype(str).unique()):
        d = g[g.cell_type.astype(str) == ct]
        w = d.pivot(index="paired_batch", columns="condition", values=["median", "size"])
        vals = {}
        for pair in sorted(clock.paired_batch.astype(str).unique()):
            try:
                if w.loc[pair, ("size", "control")] >= 20 and w.loc[pair, ("size", "oskm")] >= 20:
                    vals[pair] = float(w.loc[pair, ("median", "oskm")] - w.loc[pair, ("median", "control")])
            except KeyError:
                pass
        arr = np.asarray(list(vals.values()))
        p = stats.wilcoxon(arr, alternative="two-sided").pvalue if len(arr) >= 2 and np.any(arr != 0) else np.nan
        row = {"cell_type": ct, "n_pairs": len(arr), "median_delta_months": np.median(arr) if len(arr) else np.nan,
               "mean_delta_months": np.mean(arr) if len(arr) else np.nan,
               "younger_pairs": int((arr < 0).sum()), "older_pairs": int((arr > 0).sum()), "wilcoxon_pvalue": p}
        row.update({f"delta_{k}": v for k, v in vals.items()}); rows.append(row)
    by_ct = pd.DataFrame(rows)
    by_ct["fdr_bh"] = multipletests(by_ct.wilcoxon_pvalue.fillna(1), method="fdr_bh")[1]
    strata = clock.groupby(["paired_batch", "condition", "cell_type", "subregion"], observed=True).predicted_age.agg(["median", "size"]).reset_index()
    pair_rows = []
    for pair, d in strata.groupby("paired_batch", observed=True):
        w = d.pivot(index=["cell_type", "subregion"], columns="condition", values=["median", "size"]).dropna()
        good = (w[("size", "control")] >= 20) & (w[("size", "oskm")] >= 20)
        delta = (w.loc[good, ("median", "oskm")] - w.loc[good, ("median", "control")]).to_numpy(float)
        pair_rows.append({"paired_batch": pair, "n_strata": len(delta), "median_stratum_delta_months": float(np.median(delta)),
                          "mean_stratum_delta_months": float(np.mean(delta)), "fraction_strata_younger": float(np.mean(delta < 0))})
    return by_ct, pd.DataFrame(pair_rows)


def plots(frame: pd.DataFrame, all_clock: pd.DataFrame, pcassoc: pd.DataFrame,
          dptassoc: pd.DataFrame, held: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)
    for ax, rep in zip(axes, ["z_cc", "s_cc"]):
        d = pcassoc[pcassoc.representation == rep]
        ax.errorbar(d.pc, d.within_slice_spearman,
                    yerr=[d.within_slice_spearman - d.within_slice_block_boot_ci_low,
                          d.within_slice_block_boot_ci_high - d.within_slice_spearman], fmt="o-", capsize=3)
        sig = d.fdr_bh < .05
        ax.scatter(d.loc[sig, "pc"], d.loc[sig, "within_slice_spearman"], s=85, facecolors="none", edgecolors="red")
        ax.axhline(0, color="black", lw=.8); ax.set_title(rep); ax.set_xlabel("PC (global condition-only basis)")
    axes[0].set_ylabel("Within-slice Spearman with predicted age\n(95% spatial-block bootstrap CI)")
    fig.savefig(figures / "pc_predicted_age_associations.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    rng = np.random.default_rng(SEED)
    for ax, (sample, d) in zip(axes.flat, frame.groupby("slice_id", observed=True)):
        if len(d) > 40000: d = d.iloc[np.sort(rng.choice(len(d), 40000, replace=False))]
        ax.hexbin(d.dpt_min_pc1, d.predicted_age, gridsize=55, mincnt=1, bins="log", cmap="viridis")
        r = dptassoc[(dptassoc.scope == "slice") & (dptassoc.endpoint == "dpt_min_pc1") & (dptassoc.slice_id == sample)].iloc[0]
        ax.set_title(f"{sample}: rho={r.spearman:.3f}"); ax.set_xlabel("s-CC DPT (min-PC1 root)"); ax.set_ylabel("Predicted spatial age")
    fig.savefig(figures / "dpt_predicted_age_by_oskm_slice.png", dpi=220); plt.close(fig)

    sample_idx = np.arange(len(frame))
    if len(sample_idx) > 120000: sample_idx = np.sort(rng.choice(sample_idx, 120000, replace=False))
    d = frame.iloc[sample_idx]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    specs = [("zcc_background_umap1", "zcc_background_umap2", "predicted_age", "z-CC UMAP: predicted age"),
             ("scc_salient_umap1", "scc_salient_umap2", "predicted_age", "s-CC UMAP: predicted age"),
             ("zcc_background_umap1", "zcc_background_umap2", "dpt_min_pc1", "z-CC UMAP: s-CC DPT"),
             ("scc_salient_umap1", "scc_salient_umap2", "dpt_min_pc1", "s-CC UMAP: s-CC DPT")]
    for ax, (x, y, c, title) in zip(axes.flat, specs):
        q = np.nanquantile(d[c], [.01, .99]); im = ax.scatter(d[x], d[y], c=d[c], s=.5, cmap="viridis", rasterized=True, vmin=q[0], vmax=q[1])
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, shrink=.7)
    fig.savefig(figures / "condition_only_umaps_predicted_age_and_dpt.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    for ax, (sample, dd) in zip(axes.flat, all_clock.groupby("slice_id", observed=True)):
        q = np.nanquantile(dd.predicted_age, [.01, .99]); im = ax.scatter(dd.center_x, dd.center_y, c=dd.predicted_age, s=.25, cmap="viridis", rasterized=True, vmin=q[0], vmax=q[1])
        ax.set_title(sample); ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, shrink=.55)
    fig.suptitle("Published Spatial Aging Clock prediction, full old cohort")
    fig.savefig(figures / "spatial_maps_predicted_age_all_slices.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    for ax, (sample, dd) in zip(axes.flat, frame.groupby("slice_id", observed=True)):
        im = ax.scatter(dd.center_x, dd.center_y, c=dd.dpt_min_pc1, s=.25, cmap="magma", rasterized=True)
        ax.set_title(sample); ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, shrink=.6)
    fig.suptitle("Within-slice s-CC diffusion pseudotime (descriptive)")
    fig.savefig(figures / "spatial_maps_scc_dpt_oskm_slices.png", dpi=220); plt.close(fig)

    h = held[held.endpoint == "predicted_age_slice_ct_subregion_centered"].melt(
        id_vars="representation", value_vars=["held_pair_r2", "random_cell_r2"], var_name="split", value_name="r2")
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    sns.barplot(data=h, x="representation", y="r2", hue="split", ax=ax); ax.axhline(0, color="black", lw=.8)
    ax.set_title("Clock-age prediction: leakage-aware vs random-cell split")
    fig.savefig(figures / "held_pair_vs_random_cell_clock_prediction.png", dpi=220); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clock", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--cc-dir", required=True, type=Path)
    p.add_argument("--umap", required=True, type=Path)
    p.add_argument("--dpt", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    figures = args.out / "figures"; tables = args.out / "tables"; figures.mkdir(exist_ok=True); tables.mkdir(exist_ok=True)

    clock = pd.read_csv(args.clock)
    umap = pd.read_csv(args.umap)
    ids = np.load(args.cc_dir / "obs_names.npy", allow_pickle=True).astype(str)
    if not np.array_equal(ids, umap.cell_id.astype(str).to_numpy()): raise ValueError("CC/UMAP row mismatch")
    oskm = clock.set_index("cell_id").loc[ids].reset_index()
    if not oskm.condition.astype(str).eq("oskm").all(): raise ValueError("CC rows are not all OSKM")
    frame = umap.merge(oskm[["cell_id", "predicted_age"]], on="cell_id", validate="one_to_one")
    dpt = pd.read_csv(args.dpt)
    frame = frame.merge(dpt.drop(columns="slice_id"), on="cell_id", validate="one_to_one")

    source = ad.read_h5ad(args.data, backed="r")
    if not np.array_equal(clock.cell_id.astype(str).to_numpy(), source.obs_names.astype(str)): raise ValueError("Full clock/source order mismatch")
    coords = np.asarray(source.obsm["spatial"])
    all_clock = clock.copy(); all_clock["center_x"] = coords[:, 0]; all_clock["center_y"] = coords[:, 1]
    source_rows = umap.source_row.to_numpy(int)
    frame["center_x"] = coords[source_rows, 0]; frame["center_y"] = coords[source_rows, 1]
    source.file.close()
    frame["predicted_age_slice_ct_subregion_centered"] = frame.predicted_age - frame.groupby(
        ["slice_id", "cell_type", "subregion"], observed=True).predicted_age.transform("median")
    blocks = make_spatial_blocks(frame)
    frame["spatial_block_8x8"] = blocks

    reps = {"z_cc": np.load(args.cc_dir / "spatial.npy", mmap_mode="r"),
            "s_cc": np.load(args.cc_dir / "salient_spatial.npy", mmap_mode="r")}
    pc_frames, assoc_frames, slice_frames, held_frames, fold_frames = [], [], [], [], []
    leakage = []
    for name, rep_mmap in reps.items():
        rep = np.asarray(rep_mmap, dtype=np.float32)
        scaler = StandardScaler().fit(rep)
        pca = PCA(n_components=20, svd_solver="randomized", random_state=0)
        pcs = pca.fit_transform(scaler.transform(rep)).astype(np.float32)
        pc_frames.append(pd.DataFrame(pcs[:, :10], columns=[f"{name}_PC{i}" for i in range(1, 11)]))
        assoc, by_slice = pc_associations(frame, pcs, name, blocks); assoc_frames.append(assoc); slice_frames.append(by_slice)
        held, folds = prediction_metrics(frame, rep, name); held_frames.append(held); fold_frames.append(folds)
        clusters = MiniBatchKMeans(n_clusters=4, random_state=0, n_init=20, batch_size=8192).fit_predict(pcs[:, :10])
        nmi = normalized_mutual_info_score(frame.slice_id.astype(str), clusters)
        for j in range(10):
            x = pcs[:, j]; grand = x.mean(); total = np.sum((x - grand) ** 2)
            between = sum(len(idx) * (x[np.asarray(list(idx), int)].mean() - grand) ** 2 for idx in frame.groupby("slice_id", observed=True).groups.values())
            leakage.append({"representation": name, "pc": j + 1, "slice_eta_squared": float(between / total),
                            "kmeans4_slice_nmi_pc1_10": float(nmi), "explained_variance_ratio": float(pca.explained_variance_ratio_[j])})

    pcs_all = pd.concat(pc_frames, axis=1)
    cell_table = pd.concat([frame.reset_index(drop=True), pcs_all], axis=1)
    cell_table.to_csv(tables / "oskm_clock_dpt_pc_scores.csv.gz", index=False, compression="gzip")
    assoc = pd.concat(assoc_frames, ignore_index=True)
    assoc["fdr_bh"] = multipletests(assoc.within_slice_pvalue.fillna(1), method="fdr_bh")[1]
    assoc.to_csv(tables / "pc_predicted_age_associations.csv", index=False)
    pd.concat(slice_frames, ignore_index=True).to_csv(tables / "pc_predicted_age_by_slice.csv", index=False)
    held = pd.concat(held_frames, ignore_index=True); held.to_csv(tables / "held_pair_prediction_metrics.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(tables / "held_pair_prediction_folds.csv", index=False)
    pd.DataFrame(leakage).to_csv(tables / "sample_leakage_pc_metrics.csv", index=False)
    dpt_assoc = dpt_associations(frame, blocks); dpt_assoc.to_csv(tables / "dpt_predicted_age_associations.csv", index=False)
    ct, pairs = paired_clock_summary(clock); ct.to_csv(tables / "clock_oskm_control_celltype_pair_shifts.csv", index=False); pairs.to_csv(tables / "clock_oskm_control_pair_shifts.csv", index=False)
    plots(frame, all_clock, assoc, dpt_assoc, held, figures)

    summary = {"n_full_old_cells": int(len(clock)), "n_oskm_cells": int(len(frame)),
               "clock_coverage_fraction": float(clock.predicted_age.notna().mean()),
               "pc_multiple_testing_family": "20 primary within-slice tests (PC1-PC10 for z-CC and s-CC), Benjamini-Hochberg",
               "bootstrap": "500 replicate stratified 8x8 spatial-block bootstrap within each OSKM slice",
               "held_pair_model": "training-fold StandardScaler + randomized PCA(10, seed0) + Ridge(alpha=100); held paired_batch never enters preprocessing",
               "dpt_interpretation": "static, root-dependent, within-slice descriptive ordering; not lineage or causal time",
               "hashes": {str(p): sha256(p) for p in [args.clock, args.dpt, args.umap,
                          args.cc_dir / "spatial.npy", args.cc_dir / "salient_spatial.npy"]}}
    (args.out / "analysis_metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
