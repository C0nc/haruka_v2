"""Focused plaque recovery analysis for the all-sections Jill AD Haruka run.

Haruka is fit without plaque annotations. This script uses plaque labels only
after fitting and evaluates linear readout across the two AD tissue sections.
Control cells are excluded so condition identity cannot inflate plaque scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SLOTS = ["background", "spatial", "salient", "salient_spatial"]
PLAQUES = ["plaque_tau", "plaque_abeta", "plaque_any"]
DISPLAY = {
    "background": "Background intrinsic",
    "spatial": "Background spatial",
    "salient": "Salient intrinsic",
    "salient_spatial": "Salient spatial",
    "plaque_tau": "p-tau",
    "plaque_abeta": "Aβ",
    "plaque_any": "Any plaque",
}


def cross_section_scores(z: np.ndarray, y: np.ndarray, graph: np.ndarray) -> tuple[list[dict], np.ndarray]:
    rows: list[dict] = []
    prediction = np.full(len(y), np.nan, dtype=np.float32)
    sections = sorted(np.unique(graph))
    for test_graph in sections:
        train_graph = next(x for x in sections if x != test_graph)
        train = graph == train_graph
        test = graph == test_graph
        model = make_pipeline(
            StandardScaler(),
            RidgeClassifier(alpha=10.0, class_weight="balanced"),
        )
        model.fit(z[train], y[train])
        decision = model.decision_function(z[test]).astype(np.float32)
        prediction[test] = decision
        pred = (decision >= 0).astype(int)
        rows.append({
            "train_section": train_graph,
            "test_section": test_graph,
            "n_test": int(test.sum()),
            "test_prevalence": float(y[test].mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
            "roc_auc": float(roc_auc_score(y[test], decision)),
            "average_precision": float(average_precision_score(y[test], decision)),
        })
    return rows, prediction


def embedding_2d(z: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    try:
        import umap
        model = umap.UMAP(
            n_neighbors=25,
            min_dist=0.30,
            metric="euclidean",
            random_state=seed,
            n_jobs=1,
        )
        return model.fit_transform(z), "UMAP"
    except ImportError:
        return PCA(n_components=2, random_state=seed).fit_transform(z), "PCA"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="bench_real_discovery/data/ad.h5ad")
    p.add_argument(
        "--emb",
        default="bench_real_discovery/results/ad/haruka_all_sections_seed0",
    )
    p.add_argument(
        "--out",
        default="bench_real_discovery/results/ad/haruka_all_sections_seed0/plaque_analysis",
    )
    p.add_argument("--max-plot", type=int, default=12000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    emb = Path(args.emb)
    a = ad.read_h5ad(args.data)
    names = np.load(emb / "obs_names.npy", allow_pickle=True).astype(str)
    order = pd.Series(np.arange(a.n_obs), index=a.obs_names.astype(str)).loc[names].to_numpy()
    a = a[order].copy()
    disease = a.obs.condition.astype(str).to_numpy() == "disease"
    graph = a.obs.graph_id.astype(str).to_numpy()[disease]
    reps = {slot: np.load(emb / f"{slot}.npy").astype(np.float32)[disease] for slot in SLOTS}

    directional: list[dict] = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for slot in SLOTS:
        for plaque in PLAQUES:
            y = a.obs[plaque].to_numpy(dtype=int)[disease]
            rows, pred = cross_section_scores(reps[slot], y, graph)
            for row in rows:
                row.update({"slot": slot, "plaque": plaque})
            directional.extend(rows)
            predictions[(slot, plaque)] = pred
    direction_df = pd.DataFrame(directional)
    direction_df.to_csv(out / "plaque_recovery_by_direction.csv", index=False)
    summary = (
        direction_df.groupby(["slot", "plaque"], sort=False)
        .agg(
            balanced_accuracy=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            roc_auc=("roc_auc", "mean"),
            roc_auc_sd=("roc_auc", "std"),
            average_precision=("average_precision", "mean"),
            plaque_prevalence=("test_prevalence", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out / "plaque_recovery_summary.csv", index=False)

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    for ax, metric, title in zip(
        axes,
        ["balanced_accuracy", "roc_auc", "average_precision"],
        ["Balanced accuracy", "ROC-AUC", "Average precision"],
    ):
        matrix = summary.pivot(index="plaque", columns="slot", values=metric).reindex(
            index=PLAQUES, columns=SLOTS
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=0.5 if metric != "average_precision" else 0,
            vmax=1,
            xticklabels=[DISPLAY[x] for x in SLOTS],
            yticklabels=[DISPLAY[x] for x in PLAQUES],
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Jill AD plaque readout from co-trained, plaque-unsupervised Haruka factors")
    fig.tight_layout()
    fig.savefig(out / "plaque_recovery_heatmaps.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    rng = np.random.default_rng(args.seed)
    plot_idx = np.sort(rng.choice(len(graph), min(args.max_plot, len(graph)), replace=False))
    coords = {}
    method = None
    for slot in SLOTS:
        zz = StandardScaler().fit_transform(reps[slot][plot_idx])
        coords[slot], method = embedding_2d(zz, args.seed)
    fig, axes = plt.subplots(len(PLAQUES), len(SLOTS), figsize=(15.2, 10.5))
    for row, plaque in enumerate(PLAQUES):
        y = a.obs[plaque].to_numpy(dtype=int)[disease][plot_idx]
        for col, slot in enumerate(SLOTS):
            ax = axes[row, col]
            xy = coords[slot]
            neg, pos = y == 0, y == 1
            ax.scatter(xy[neg, 0], xy[neg, 1], s=1.8, c="#d6d6d6", alpha=0.35, rasterized=True)
            ax.scatter(xy[pos, 0], xy[pos, 1], s=5.0, c="#d62728", alpha=0.82, rasterized=True)
            if row == 0:
                ax.set_title(DISPLAY[slot])
            if col == 0:
                ax.set_ylabel(DISPLAY[plaque])
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"Plaque-positive AD cells on each Haruka factor ({method}; plaque labels evaluation-only)")
    fig.tight_layout()
    fig.savefig(out / "plaque_four_slot_umap.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    best = summary.loc[summary.groupby("plaque")["balanced_accuracy"].idxmax()].copy()
    best.to_csv(out / "plaque_best_slots.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print("\nBest slot per plaque endpoint:\n", best.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
