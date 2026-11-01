#!/usr/bin/env python3
"""Plot condition-only Haruka z/s and z-CC/s-CC colored by sample."""

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
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP


def balanced_rows(labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0:
        return np.arange(len(labels))
    if len(labels) <= limit:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    groups = np.unique(labels)
    per_group = max(1, limit // len(groups))
    chosen = []
    for group in groups:
        rows = np.flatnonzero(labels == group)
        chosen.append(rng.choice(rows, min(per_group, len(rows)), replace=False))
    result = np.sort(np.concatenate(chosen))
    if len(result) < limit:
        remainder = np.setdiff1d(np.arange(len(labels)), result, assume_unique=True)
        result = np.sort(np.concatenate([result, rng.choice(remainder, min(limit - len(result), len(remainder)), replace=False)]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cc-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--sample-key", required=True)
    parser.add_argument(
        "--max-plot", type=int, default=40000,
        help="Maximum plotted cells; 0 retains every condition cell.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source = ad.read_h5ad(args.data, backed="r")
    source_rows = np.load(args.cc_dir / "source_rows.npy").astype(int)
    obs = source.obs.iloc[source_rows].copy()
    source.file.close()
    names = np.load(args.cc_dir / "obs_names.npy", allow_pickle=True).astype(str)
    if not np.array_equal(names, obs.index.astype(str).to_numpy()):
        raise ValueError("Condition-only CellCharter rows do not align with source AnnData")
    samples = obs[args.sample_key].astype(str).to_numpy()
    rows = balanced_rows(samples, args.max_plot, args.seed)

    representations = {
        "z intrinsic": np.load(args.cc_dir / "background.npy", mmap_mode="r"),
        "s intrinsic": np.load(args.cc_dir / "salient.npy", mmap_mode="r"),
        "z-CC background": np.load(args.cc_dir / "spatial.npy", mmap_mode="r"),
        "s-CC salient": np.load(args.cc_dir / "salient_spatial.npy", mmap_mode="r"),
    }
    coordinates = pd.DataFrame({
        "cell_id": names[rows],
        "source_row": source_rows[rows],
        "sample": samples[rows],
    })
    diagnostics = []
    fig, axes = plt.subplots(2, 2, figsize=(19, 15))
    palette = dict(zip(sorted(np.unique(samples)), sns.color_palette("husl", len(np.unique(samples)))))
    for ax, (name, matrix) in zip(axes.flat, representations.items()):
        x = np.asarray(matrix[rows], dtype=np.float32)
        x = StandardScaler().fit_transform(x)
        xy = UMAP(
            n_neighbors=30,
            min_dist=0.25,
            metric="euclidean",
            random_state=args.seed,
            n_jobs=1,
        ).fit_transform(x)
        key = name.lower().replace(" ", "_").replace("-", "")
        coordinates[f"{key}_umap1"] = xy[:, 0]
        coordinates[f"{key}_umap2"] = xy[:, 1]
        n_clusters = len(np.unique(samples))
        clusters = MiniBatchKMeans(
            n_clusters=n_clusters,
            n_init=20,
            batch_size=4096,
            random_state=args.seed,
        ).fit_predict(x)
        sample_nmi = normalized_mutual_info_score(samples[rows], clusters)
        diagnostics.append({
            "study": args.study,
            "representation": name,
            "n_plotted": int(len(rows)),
            "n_samples": int(n_clusters),
            "sample_kmeans_nmi": float(sample_nmi),
        })
        for sample in sorted(np.unique(samples)):
            take = samples[rows] == sample
            ax.scatter(
                xy[take, 0], xy[take, 1], s=3, linewidths=0,
                alpha=0.78, color=palette[sample], rasterized=True, label=sample,
            )
        ax.set_title(f"{name} — sample NMI={sample_nmi:.3f}")
        ax.set(xticks=[], yticks=[], xlabel="UMAP 1", ylabel="UMAP 2")
        ax.legend(
            title=args.sample_key, frameon=False, fontsize=6,
            markerscale=3, ncol=max(1, min(4, int(np.ceil(n_clusters / 12)))),
            bbox_to_anchor=(1.01, 1), loc="upper left",
        )
    fig.suptitle(
        f"{args.study}: condition-only composition-gated Haruka — colored by {args.sample_key}",
        fontsize=16,
    )
    fig.tight_layout()
    figure = args.out_dir / "condition_only_z_s_zcc_scc_by_sample.png"
    fig.savefig(figure, dpi=220, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(diagnostics).to_csv(args.out_dir / "sample_separation_metrics.csv", index=False)
    coordinates.to_csv(args.out_dir / "sample_umap_coordinates.csv.gz", index=False, compression="gzip")
    metadata = {
        "study": args.study,
        "data": str(args.data.resolve()),
        "cc_dir": str(args.cc_dir.resolve()),
        "sample_key": args.sample_key,
        "population": "condition cells only",
        "n_condition": int(len(obs)),
        "n_plotted": int(len(rows)),
        "interpretation": "sample NMI diagnoses retained sample geometry; decoder batch conditioning does not guarantee latent invariance",
    }
    (args.out_dir / "sample_umap_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(pd.DataFrame(diagnostics).to_string(index=False), flush=True)
    print(figure, flush=True)


if __name__ == "__main__":
    main()
