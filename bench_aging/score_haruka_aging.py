"""Evaluate what each Haruka slot captures on MERFISH aging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SLOTS = ["background", "spatial", "salient", "salient_spatial"]
POSITIVE = ["Gfap", "C4b", "H2-K1", "Stat1", "B2m", "Ifi27", "Bst2", "Apoe"]
NEGATIVE = ["Cat", "Gamt"]


def cv_classification(z, y, groups, seed=0):
    y = np.asarray(y).astype(str)
    groups = np.asarray(groups).astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    values = []
    for tr, te in splitter.split(z, y, groups):
        if len(np.unique(y[tr])) < len(np.unique(y)):
            continue
        model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0, class_weight="balanced"))
        model.fit(z[tr], y[tr])
        values.append(balanced_accuracy_score(y[te], model.predict(z[te])))
    return float(np.mean(values)), float(np.std(values)), len(values)


def cv_regression(z, y, groups):
    y = np.asarray(y, dtype=float)
    bins = pd.qcut(y, q=min(5, len(np.unique(y))), duplicates="drop").codes
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    pred = np.full(len(y), np.nan)
    for tr, te in splitter.split(z, bins, groups):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(z[tr], y[tr])
        pred[te] = model.predict(z[te])
    ok = np.isfinite(pred)
    return float(r2_score(y[ok], pred[ok])), float(spearmanr(y[ok], pred[ok]).statistic)


def group_center(z, obs):
    keys = (obs.cell_type.astype(str) + "|" + obs.region_label.astype(str) +
            "|" + obs.technical_batch.astype(str)).to_numpy()
    out = z.copy()
    for key in np.unique(keys):
        m = keys == key
        out[m] -= out[m].mean(0, keepdims=True)
    return out


def marker_scores(adata):
    x = adata.layers["counts"] if "counts" in adata.layers else adata.X
    x = x.toarray() if sp.issparse(x) else np.asarray(x)
    lib = np.maximum(x.sum(1, keepdims=True), 1.0)
    x = np.log1p(x / lib * 1e4)
    index = {g: i for i, g in enumerate(adata.var_names.astype(str))}
    pos = [index[g] for g in POSITIVE if g in index]
    neg = [index[g] for g in NEGATIVE if g in index]
    score = x[:, pos].mean(1) - x[:, neg].mean(1)
    local = np.zeros(len(score), dtype=np.float32)
    mouse = adata.obs.mouse_id.astype(str).to_numpy()
    xy = np.asarray(adata.obsm["spatial"])
    for value in np.unique(mouse):
        idx = np.flatnonzero(mouse == value)
        k = min(9, len(idx))
        nbr = NearestNeighbors(n_neighbors=k).fit(xy[idx]).kneighbors(return_distance=False)
        if k > 1:
            nbr = nbr[:, 1:]
        local[idx] = score[idx[nbr]].mean(1)
    return score.astype(np.float32), local, (score - local).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="bench_aging/data/merfish_aging_100k.h5ad")
    p.add_argument("--emb", required=True)
    p.add_argument("--out", default="bench_aging/results/analysis")
    p.add_argument("--max-probe", type=int, default=40000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(args.data)
    saved_names = np.load(Path(args.emb) / "obs_names.npy", allow_pickle=True).astype(str)
    order = pd.Series(np.arange(a.n_obs), index=a.obs_names.astype(str)).loc[saved_names].to_numpy()
    a = a[order].copy()
    reps = {slot: np.load(Path(args.emb) / f"{slot}.npy").astype(np.float32) for slot in SLOTS}
    if any(len(z) != a.n_obs for z in reps.values()):
        raise ValueError("embedding rows do not match AnnData")

    rng = np.random.default_rng(args.seed)
    per_mouse = max(1, args.max_probe // a.obs.mouse_id.nunique())
    probe_idx = []
    for mouse, frame in a.obs.groupby("mouse_id", observed=True):
        idx = a.obs_names.get_indexer(frame.index)
        probe_idx.extend(rng.choice(idx, min(per_mouse, len(idx)), replace=False))
    probe_idx = np.sort(np.asarray(probe_idx, dtype=int))
    obs = a.obs.iloc[probe_idx].copy()
    mouse = obs.mouse_id.astype(str).to_numpy()
    target = obs.age_group.astype(str).isin(["middle", "late"]).to_numpy()
    score, local_score, intrinsic_score = marker_scores(a)

    probe_rows = []
    labels = {
        "cell_identity": (obs.cell_type.astype(str).to_numpy(), np.ones(len(obs), bool), False),
        "tissue_region": (obs.region_label.astype(str).to_numpy(), np.ones(len(obs), bool), False),
        "control_vs_aged": (obs.condition.astype(str).to_numpy(), np.ones(len(obs), bool), True),
        "age_group_3way": (obs.age_group.astype(str).to_numpy(), np.ones(len(obs), bool), True),
        "target_stage_mid_vs_late": (obs.age_group.astype(str).to_numpy(), target, True),
        "technical_batch": (obs.technical_batch.astype(str).to_numpy(), np.ones(len(obs), bool), False),
    }
    centered = {slot: group_center(z[probe_idx], obs) for slot, z in reps.items()}
    for slot, z_full in reps.items():
        raw = z_full[probe_idx]
        for factor, (y, use, residualized) in labels.items():
            z = centered[slot] if residualized else raw
            mean, sd, folds = cv_classification(z[use], y[use], mouse[use], args.seed)
            probe_rows.append({"slot": slot, "factor": factor, "score": mean,
                               "fold_sd": sd, "n_folds": folds,
                               "covariate_residualized": residualized,
                               "chance": 1.0 / len(np.unique(y[use]))})
        r2, rho = cv_regression(centered[slot], obs.age.astype(float).to_numpy(), mouse)
        probe_rows.append({"slot": slot, "factor": "exact_age_regression", "score": rho,
                           "r2": r2, "fold_sd": np.nan, "n_folds": 5,
                           "covariate_residualized": True, "chance": 0.0})
        for factor, values in [("local_aging_marker_score", local_score[probe_idx]),
                               ("intrinsic_aging_marker_score", intrinsic_score[probe_idx])]:
            r2, rho = cv_regression(centered[slot][target], values[target], mouse[target])
            probe_rows.append({"slot": slot, "factor": factor, "score": rho,
                               "r2": r2, "fold_sd": np.nan, "n_folds": 5,
                               "covariate_residualized": True, "chance": 0.0})
    probes = pd.DataFrame(probe_rows)
    probes.to_csv(out / "cross_mouse_probes.csv", index=False)

    # Unsupervised clustering: no labels enter the fit.
    cluster_rows = []
    cluster_idx = rng.choice(probe_idx, min(30000, len(probe_idx)), replace=False)
    cluster_obs = a.obs.iloc[cluster_idx]
    cluster_labels = {
        "cell_identity": cluster_obs.cell_type.astype(str),
        "tissue_region": cluster_obs.region_label.astype(str),
        "age_group_3way": cluster_obs.age_group.astype(str),
        "control_vs_aged": cluster_obs.condition.astype(str),
    }
    for slot, z in reps.items():
        zz = StandardScaler().fit_transform(z[cluster_idx])
        for factor, y in cluster_labels.items():
            k = len(np.unique(y))
            pred = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit_predict(zz)
            cluster_rows.append({"slot": slot, "factor": factor,
                                 "ari": adjusted_rand_score(y, pred), "n_clusters": k})
    clusters = pd.DataFrame(cluster_rows)
    clusters.to_csv(out / "clustering_ari.csv", index=False)

    # Slot-factor matrix and assignment margins.
    wanted = ["cell_identity", "tissue_region", "target_stage_mid_vs_late",
              "local_aging_marker_score"]
    matrix = probes[probes.factor.isin(wanted)].pivot(index="factor", columns="slot", values="score")
    matrix = matrix.reindex(index=wanted, columns=SLOTS)
    matrix.to_csv(out / "four_slot_factor_matrix.csv")
    expected = dict(zip(wanted, SLOTS))
    margins = []
    for factor, slot in expected.items():
        own = float(matrix.loc[factor, slot])
        other = float(matrix.loc[factor].drop(slot).max())
        margins.append({"factor": factor, "expected_slot": slot, "score": own,
                        "best_other": other, "assignment_margin": own - other})
    pd.DataFrame(margins).to_csv(out / "assignment_margins.csv", index=False)

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax)
    ax.set_title("Cross-mouse information in Haruka's four slots")
    fig.tight_layout(); fig.savefig(out / "four_slot_factor_matrix.png", dpi=220); plt.close(fig)

    # UMAP is descriptive only; probes above are the quantitative test.
    import umap
    vis_idx = rng.choice(probe_idx, min(6000, len(probe_idx)), replace=False)
    vis_obs = a.obs.iloc[vis_idx]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    expected_color = ["cell_type", "region_label", "age_group", "age_group"]
    for col, slot in enumerate(SLOTS):
        emb = umap.UMAP(n_neighbors=30, min_dist=0.25, random_state=args.seed).fit_transform(reps[slot][vis_idx])
        for row, key in enumerate([expected_color[col], "age_group"]):
            values = vis_obs[key].astype(str).to_numpy()
            palette = dict(zip(sorted(set(values)), sns.color_palette("tab20", len(set(values)))))
            colors = [palette[x] for x in values]
            axes[row, col].scatter(emb[:, 0], emb[:, 1], c=colors, s=2, linewidths=0, rasterized=True)
            axes[row, col].set_title(f"{slot}: {key}")
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    fig.tight_layout(); fig.savefig(out / "four_slot_umap.png", dpi=200); plt.close(fig)

    result = {
        "n_cells": int(a.n_obs), "n_probe": int(len(probe_idx)),
        "training_labels": "early control versus pooled middle+late only",
        "middle_late_supervision": False,
        "evaluation": "five-fold stratified group CV by mouse",
        "limitation": "representation fit is transductive; exact age is one-to-one with mouse",
        "assignment_margins": margins,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(matrix.to_string(), flush=True)
    print(pd.DataFrame(margins).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
