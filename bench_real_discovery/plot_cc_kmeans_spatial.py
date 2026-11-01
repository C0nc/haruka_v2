"""Plot condition-only spatial maps of z-CC and s-CC KMeans clusters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--emb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-key", required=True)
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--target-label", default="disease")
    parser.add_argument("--annotation-key", required=True)
    parser.add_argument("--n-clusters", type=int, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    a = ad.read_h5ad(args.data, backed="r")
    names = np.load(args.emb / "obs_names.npy", allow_pickle=True).astype(str)
    pos = pd.Index(a.obs_names.astype(str)).get_indexer(names)
    if (pos < 0).any():
        raise ValueError("Embedding cells are absent from AnnData")
    obs = a.obs.iloc[pos].copy()
    spatial = np.asarray(a.obsm["spatial"])[pos]
    a.file.close()

    target = obs[args.condition_key].astype(str).to_numpy() == args.target_label
    idx = np.flatnonzero(target)
    if not len(idx):
        raise ValueError("No target cells found")
    z_cc = np.load(args.emb / "spatial.npy").astype(np.float32)[idx]
    s_cc = np.load(args.emb / "salient_spatial.npy").astype(np.float32)[idx]
    if z_cc.shape[0] != len(idx) or s_cc.shape[0] != len(idx):
        raise ValueError("Embedding row count mismatch")

    labels = {}
    for slot, rep in (("z_cc", z_cc), ("s_cc", s_cc)):
        scaled = StandardScaler().fit_transform(rep)
        labels[slot] = KMeans(
            n_clusters=args.n_clusters, n_init=20, random_state=args.seed
        ).fit_predict(scaled)

    target_obs = obs.iloc[idx].copy()
    samples = sorted(target_obs[args.sample_key].astype(str).unique())
    palette = sns.color_palette("husl", args.n_clusters)
    sns.set_theme(style="white")
    fig, axes = plt.subplots(
        2, len(samples), figsize=(5.2 * len(samples), 9.0), squeeze=False
    )
    xy = spatial[idx]
    for row, slot in enumerate(("z_cc", "s_cc")):
        cluster = labels[slot]
        prefix = "Z" if slot == "z_cc" else "S"
        for col, sample in enumerate(samples):
            ax = axes[row, col]
            take = target_obs[args.sample_key].astype(str).to_numpy() == sample
            for k in range(args.n_clusters):
                use = take & (cluster == k)
                if use.any():
                    ax.scatter(
                        xy[use, 0], xy[use, 1], s=3.0, color=palette[k],
                        linewidths=0, alpha=0.85, rasterized=True,
                    )
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{sample} — {slot} KMeans")
            if col == 0:
                ax.set_ylabel("Background spatial" if row == 0 else "Salient spatial")
        handles = [Patch(color=palette[k], label=f"{prefix}{k}") for k in range(args.n_clusters)]
        axes[row, -1].legend(
            handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
            frameon=False, fontsize=7, ncol=1,
        )
    fig.suptitle(args.title + " (condition cells only)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 0.94, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    plt.close(fig)

    table = pd.DataFrame({
        "obs_name": target_obs.index.astype(str),
        args.sample_key: target_obs[args.sample_key].astype(str).to_numpy(),
        "x": xy[:, 0], "y": xy[:, 1],
        "z_cc_kmeans": [f"Z{x}" for x in labels["z_cc"]],
        "s_cc_kmeans": [f"S{x}" for x in labels["s_cc"]],
    })
    for key in (args.annotation_key, "region", "tissue_domain", "plaque_any", "plaque_abeta", "plaque_tau"):
        if key in target_obs and key not in table:
            table[key] = target_obs[key].astype(str).to_numpy()
    table.to_csv(args.out.with_name(args.out.stem + "_assignments.csv"), index=False)

    metrics = []
    audit_keys = [args.annotation_key, args.sample_key]
    audit_keys += [k for k in ("region", "tissue_domain", "plaque_any", "plaque_abeta", "plaque_tau") if k in target_obs]
    for slot in ("z_cc", "s_cc"):
        for key in dict.fromkeys(audit_keys):
            truth = target_obs[key].astype(str).to_numpy()
            metrics.append({
                "representation": slot,
                "factor": key,
                "ari": adjusted_rand_score(truth, labels[slot]),
                "nmi": normalized_mutual_info_score(truth, labels[slot]),
                "n_clusters": args.n_clusters,
                "n_cells": len(idx),
            })
    pd.DataFrame(metrics).to_csv(
        args.out.with_name(args.out.stem + "_metrics.csv"), index=False
    )
    meta_path = args.emb / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    print(json.dumps({
        "figure": str(args.out), "n_target": len(idx), "samples": samples,
        "n_clusters": args.n_clusters, "cc_layers": meta.get("n_layers"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
