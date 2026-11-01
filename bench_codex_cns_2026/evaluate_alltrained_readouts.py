"""Pooled all-donor readouts for an all-cell-trained Haruka protein model."""

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
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from umap import UMAP


SLOTS = {
    "z": "background.npy",
    "s": "salient.npy",
    "z_cc": "spatial.npy",
    "s_cc": "salient_spatial.npy",
}


def align(data: Path, emb: Path) -> tuple[ad.AnnData, dict[str, np.ndarray]]:
    a = ad.read_h5ad(data)
    names = np.load(emb / "obs_names.npy", allow_pickle=True).astype(str)
    order = pd.Index(a.obs_names.astype(str)).get_indexer(names)
    if (order < 0).any():
        raise ValueError("Embedding observations are absent from AnnData")
    a = a[order].copy()
    reps = {k: np.load(emb / f).astype(np.float32) for k, f in SLOTS.items()}
    if any(v.shape[0] != a.n_obs for v in reps.values()):
        raise ValueError("Embedding row count mismatch")
    return a, reps


def binary_lodo(x, y, donor):
    rows, pred = [], np.full(len(y), np.nan, dtype=np.float32)
    for test_donor in sorted(np.unique(donor)):
        train, test = donor != test_donor, donor == test_donor
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        model = make_pipeline(
            StandardScaler(), RidgeClassifier(alpha=10.0, class_weight="balanced")
        )
        model.fit(x[train], y[train])
        score = model.decision_function(x[test]).astype(np.float32)
        pred[test] = score
        rows.append(
            {
                "test_donor": test_donor,
                "n_test": int(test.sum()),
                "prevalence": float(y[test].mean()),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y[test], score >= 0)
                ),
                "roc_auc": float(roc_auc_score(y[test], score)),
                "average_precision": float(average_precision_score(y[test], score)),
            }
        )
    return rows, pred


def categorical_lodo(x, y, donor):
    rows = []
    for test_donor in sorted(np.unique(donor)):
        train, test = donor != test_donor, donor == test_donor
        model = make_pipeline(
            StandardScaler(), RidgeClassifier(alpha=10.0, class_weight="balanced")
        )
        model.fit(x[train], y[train])
        pred = model.predict(x[test])
        rows.append(
            {
                "test_donor": test_donor,
                "n_test": int(test.sum()),
                "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
                "macro_f1": float(f1_score(y[test], pred, average="macro")),
            }
        )
    return rows


def continuous_lodo(x, y, donor):
    rows = []
    for test_donor in sorted(np.unique(donor)):
        train, test = donor != test_donor, donor == test_donor
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(x[train], y[train])
        pred = model.predict(x[test])
        rho = spearmanr(y[test], pred).statistic
        rows.append(
            {
                "test_donor": test_donor,
                "n_test": int(test.sum()),
                "r2": float(r2_score(y[test], pred)),
                "spearman": float(rho),
            }
        )
    return rows


def binary_pooled(x, y, seed):
    rows, pred = [], np.full(len(y), np.nan, dtype=np.float32)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(splitter.split(x, y)):
        model = make_pipeline(
            StandardScaler(), RidgeClassifier(alpha=10.0, class_weight="balanced")
        )
        model.fit(x[train], y[train])
        score = model.decision_function(x[test]).astype(np.float32)
        pred[test] = score
        rows.append({
            "fold": fold, "n_test": int(test.sum()),
            "prevalence": float(y[test].mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y[test], score >= 0)),
            "roc_auc": float(roc_auc_score(y[test], score)),
            "average_precision": float(average_precision_score(y[test], score)),
        })
    return rows, pred


def categorical_pooled(x, y, seed):
    rows = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(splitter.split(x, y)):
        model = make_pipeline(
            StandardScaler(), RidgeClassifier(alpha=10.0, class_weight="balanced")
        )
        model.fit(x[train], y[train])
        pred = model.predict(x[test])
        rows.append({
            "fold": fold, "n_test": int(test.sum()),
            "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
            "macro_f1": float(f1_score(y[test], pred, average="macro")),
        })
    return rows


def continuous_pooled(x, y, seed):
    rows = []
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(splitter.split(x)):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(x[train], y[train])
        pred = model.predict(x[test])
        rows.append({
            "fold": fold, "n_test": int(test.sum()),
            "r2": float(r2_score(y[test], pred)),
            "spearman": float(spearmanr(y[test], pred).statistic),
        })
    return rows


def make_umap(a, reps, mask, out, seed, max_plot):
    idx = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    if len(idx) > max_plot:
        positive = idx[a.obs["plaque_within_5um"].to_numpy()[idx] == 1]
        negative = np.setdiff1d(idx, positive, assume_unique=True)
        keep_pos = rng.choice(positive, min(len(positive), max_plot // 2), replace=False)
        keep_neg = rng.choice(negative, max_plot - len(keep_pos), replace=False)
        idx = np.sort(np.concatenate([keep_pos, keep_neg]))
    distance = a.obs["Distance to nearest Plaque (µm)"].to_numpy(dtype=float)[idx]
    binary = a.obs["plaque_within_5um"].to_numpy(dtype=int)[idx]
    donor = a.obs["donor_id"].astype(str).to_numpy()[idx]
    donor_levels = sorted(np.unique(donor))
    donor_palette = dict(
        zip(donor_levels, sns.color_palette("tab10", len(donor_levels)))
    )

    sns.set_theme(style="white")
    fig, axes = plt.subplots(3, 4, figsize=(16, 10.5))
    coords = []
    for col, (slot, z) in enumerate(reps.items()):
        zz = StandardScaler().fit_transform(z[idx])
        xy = UMAP(
            n_neighbors=30,
            min_dist=0.25,
            metric="cosine",
            random_state=seed + col,
            n_jobs=1,
        ).fit_transform(zz)
        coords.append(
            pd.DataFrame(
                {
                    "obs_name": a.obs_names[idx],
                    "slot": slot,
                    "UMAP1": xy[:, 0],
                    "UMAP2": xy[:, 1],
                    "plaque_within_5um": binary,
                    "plaque_distance_um": distance,
                    "donor_id": a.obs["donor_id"].astype(str).to_numpy()[idx],
                }
            )
        )
        order = np.argsort(binary)
        axes[0, col].scatter(
            xy[order, 0], xy[order, 1], c=np.where(binary[order] == 1, "#d62728", "#d8d8d8"),
            s=np.where(binary[order] == 1, 5.0, 1.8), alpha=0.65, linewidths=0,
            rasterized=True,
        )
        axes[0, col].set_title(f"{slot}: within 5 µm")
        sca = axes[1, col].scatter(
            xy[:, 0], xy[:, 1], c=np.clip(distance, 0, 200), cmap="viridis_r",
            s=2.2, alpha=0.7, linewidths=0, rasterized=True,
        )
        axes[1, col].set_title(f"{slot}: distance (clipped 200 µm)")
        for donor_id in donor_levels:
            take = donor == donor_id
            axes[2, col].scatter(
                xy[take, 0],
                xy[take, 1],
                color=donor_palette[donor_id],
                s=2.5,
                alpha=0.65,
                linewidths=0,
                rasterized=True,
                label=donor_id,
            )
        axes[2, col].set_title(f"{slot}: donor_id")
        if col == 3:
            axes[2, col].legend(
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                frameon=False,
                fontsize=7,
                markerscale=2.5,
            )
        for row in range(3):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    fig.colorbar(sca, ax=axes[1, :].tolist(), label="Distance to nearest plaque (µm)")
    fig.suptitle("AD gray-matter myeloid cells: plaque labels are evaluation-only")
    fig.savefig(out / "plaque_four_slot_umap.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    pd.concat(coords, ignore_index=True).to_csv(out / "plaque_four_slot_umap_coordinates.csv", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--emb", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-plot", type=int, default=16000)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    a, reps = align(args.data, args.emb)
    donor = a.obs["donor_id"].astype(str).to_numpy()
    disease_gm = (
        (a.obs["condition"].astype(str).to_numpy() == "disease")
        & (a.obs["region"].astype(str).to_numpy() == "GM")
        & a.obs["plaque_annotation_available"].to_numpy(dtype=bool)
    )

    fold_rows = []
    prediction_rows = []
    for slot, x in reps.items():
        for endpoint in ("plaque_within_0um", "plaque_within_5um", "plaque_within_30um"):
            y = a.obs[endpoint].to_numpy(dtype=int)[disease_gm]
            rows, pred = binary_pooled(x[disease_gm], y, args.seed)
            for row in rows:
                row.update({"task": endpoint, "slot": slot, "kind": "binary"})
            fold_rows.extend(rows)
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "obs_name": a.obs_names[disease_gm], "slot": slot,
                        "task": endpoint, "truth": y, "prediction": pred,
                        "donor_id": donor[disease_gm],
                    }
                )
            )
        y_cont = a.obs["plaque_proximity_score_30um"].to_numpy(dtype=float)[disease_gm]
        rows = continuous_pooled(x[disease_gm], y_cont, args.seed)
        for row in rows:
            row.update({"task": "plaque_proximity_score_30um", "slot": slot, "kind": "continuous"})
        fold_rows.extend(rows)

        for task, labels in (
            ("paper_initial_leiden", a.obs["paper_initial_leiden"].astype(str).to_numpy()),
            ("region", a.obs["region"].astype(str).to_numpy()),
            ("donor_id", a.obs["donor_id"].astype(str).to_numpy()),
            ("condition", a.obs["condition"].astype(str).to_numpy()),
        ):
            rows = categorical_pooled(x, labels, args.seed)
            for row in rows:
                row.update({"task": task, "slot": slot, "kind": "categorical"})
            fold_rows.extend(rows)

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(args.out / "pooled_readouts_by_fold.csv", index=False)
    numeric = [c for c in ["balanced_accuracy", "roc_auc", "average_precision", "macro_f1", "r2", "spearman"] if c in folds]
    summary = folds.groupby(["task", "slot", "kind"], as_index=False)[numeric].mean()
    summary.to_csv(args.out / "pooled_readouts_summary.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        args.out / "plaque_binary_predictions.csv", index=False
    )
    make_umap(a, reps, disease_gm, args.out, args.seed, args.max_plot)
    meta = {
        "evaluation_scope": "plaque: AD GM cells only; annotation: all primary cells",
        "validation": "pooled stratified five-fold cell CV; no donor holdout",
        "plaque_labels_used_for_training_haruka": False,
        "n_ad_gm": int(disease_gm.sum()),
        "ad_donors": sorted(np.unique(donor[disease_gm]).tolist()),
    }
    (args.out / "evaluation_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
