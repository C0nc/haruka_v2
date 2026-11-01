"""Conservative biological discovery analysis for real SPARC-seq Haruka-MENDER.

The model is label-free beyond NT/target arm membership.  This script uses true
sgRNA identities only after fitting and requires effects to agree across tissue
slices before describing them as reproducible.
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def target_gene(guide: str) -> str:
    match = re.match(r"^sg(.+)_\d+$", str(guide))
    return match.group(1) if match else str(guide).removeprefix("sg")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else np.nan


def cross_slice_cosine(vectors: dict[str, np.ndarray]) -> float:
    values = [cosine(vectors[a], vectors[b]) for a, b in combinations(sorted(vectors), 2)]
    values = [x for x in values if np.isfinite(x)]
    return float(np.mean(values)) if values else np.nan


def log_normalized(a: ad.AnnData) -> np.ndarray:
    x = a.layers["counts"]
    total = np.asarray(x.sum(1)).ravel().clip(min=1)
    if sp.issparse(x):
        out = x.multiply(1e4 / total[:, None]).astype(np.float32)
        out.data = np.log1p(out.data)
        return out.toarray()
    return np.log1p(np.asarray(x, np.float32) * (1e4 / total[:, None]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--emb-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--k-nt", type=int, default=25)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.emb_dir)
    a = ad.read_h5ad(args.data)
    order = np.load(root / "obs_names.npy", allow_pickle=True).astype(str)
    a = a[order].copy()
    obs = a.obs
    anchor = obs.is_anchor.to_numpy(bool)
    guide = obs.perturbation.astype(str).to_numpy()
    slices = obs.slice_id.astype(str).to_numpy()
    control = anchor & (guide == "NT")
    target = anchor & (guide != "NT")

    background = np.load(root / "background.npy").astype(np.float64)
    salient = np.load(root / "salient.npy").astype(np.float64)
    salsp = np.load(root / "salient_spatial.npy").astype(np.float64)
    background = StandardScaler().fit(background[control]).transform(background)
    salient = StandardScaler().fit(salient[control]).transform(salient)
    salsp_target = StandardScaler().fit_transform(salsp[target])
    salsp_scaled = np.zeros_like(salsp)
    salsp_scaled[target] = salsp_target
    for section in np.unique(slices):
        rows = target & (slices == section)
        salsp_scaled[rows] -= salsp_scaled[rows].mean(0, keepdims=True)

    matched = np.full((a.n_obs, args.k_nt), -1, dtype=np.int64)
    delta = np.zeros_like(salient)
    distance = np.full(a.n_obs, np.nan)
    for section in np.unique(slices):
        pool = np.flatnonzero(control & (slices == section))
        query = np.flatnonzero(anchor & (slices == section))
        k = min(args.k_nt + 1, len(pool))
        model = NearestNeighbors(n_neighbors=k).fit(background[pool])
        neighbors = model.kneighbors(background[query], return_distance=False)
        for row, local in zip(query, neighbors):
            choices = pool[local]
            choices = choices[choices != row][:args.k_nt]
            matched[row, :len(choices)] = choices
            center = salient[choices].mean(0)
            delta[row] = salient[row] - center
            distance[row] = np.linalg.norm(delta[row])

    percentile = np.full(a.n_obs, np.nan)
    robust_z = np.full(a.n_obs, np.nan)
    for section in np.unique(slices):
        null = distance[control & (slices == section)]
        med = np.nanmedian(null)
        scale = max(1.4826 * np.nanmedian(np.abs(null - med)), 1e-6)
        rows = anchor & (slices == section)
        robust_z[rows] = (distance[rows] - med) / scale
        percentile[rows] = np.asarray([
            (np.sum(null <= value) + 1) / (len(null) + 1) for value in distance[rows]
        ])

    expression = log_normalized(a)
    genes = np.asarray(a.var_names.astype(str))
    gene_pos = {g: i for i, g in enumerate(genes)}
    summaries = []
    discoveries = {}
    for label in sorted(set(guide[target])):
        rows = np.flatnonzero(target & (guide == label))
        int_centroids, sp_centroids = {}, {}
        expression_deltas = []
        engagement = []
        gene = target_gene(label)
        for section in sorted(set(slices[rows])):
            section_rows = rows[slices[rows] == section]
            int_centroids[section] = delta[section_rows].mean(0)
            sp_centroids[section] = salsp_scaled[section_rows].mean(0)
            for row in section_rows:
                use = matched[row]
                use = use[use >= 0]
                expression_deltas.append(expression[row] - expression[use].mean(0))
            if gene in gene_pos:
                nt = control & (slices == section)
                engagement.append(
                    float(expression[section_rows, gene_pos[gene]].mean()
                          - expression[nt, gene_pos[gene]].mean())
                )
        mean_expression_delta = np.mean(expression_deltas, axis=0)
        up = np.argsort(-mean_expression_delta)[:10]
        down = np.argsort(mean_expression_delta)[:10]
        summaries.append({
            "guide": label,
            "target_gene": gene,
            "n_anchor": len(rows),
            "median_intrinsic_response_z": float(np.nanmedian(robust_z[rows])),
            "responder_fraction": float(np.mean(percentile[rows] >= 0.95)),
            "intrinsic_cross_slice_cosine": cross_slice_cosine(int_centroids),
            "spatial_cross_slice_cosine": cross_slice_cosine(sp_centroids),
            "target_expression_effect_mean": float(np.mean(engagement)) if engagement else np.nan,
            "target_expression_negative_slices": int(np.sum(np.asarray(engagement) < 0)),
        })
        discoveries[label] = {
            "target_gene": gene,
            "target_expression_effect_by_slice": engagement,
            "top_up": [
                {"gene": str(genes[j]), "matched_log_expression_delta": float(mean_expression_delta[j])}
                for j in up
            ],
            "top_down": [
                {"gene": str(genes[j]), "matched_log_expression_delta": float(mean_expression_delta[j])}
                for j in down
            ],
        }

    summary = pd.DataFrame(summaries).sort_values(
        ["intrinsic_cross_slice_cosine", "responder_fraction"], ascending=False
    )
    summary.to_csv(out / "guide_reproducibility.csv", index=False)
    (out / "matched_expression_discoveries.json").write_text(
        json.dumps(discoveries, indent=2) + "\n"
    )
    cell_frame = pd.DataFrame({
        "obs_name": a.obs_names,
        "slice_id": slices,
        "guide": guide,
        "is_anchor": anchor,
        "intrinsic_response_z": robust_z,
        "intrinsic_response_percentile": percentile,
    })
    cell_frame[anchor].to_csv(out / "anchor_response_scores.csv", index=False)

    plot = summary.sort_values("responder_fraction", ascending=False)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fields = [
        ("responder_fraction", "State-matched intrinsic responder fraction"),
        ("intrinsic_cross_slice_cosine", "Intrinsic direction agreement across slices"),
        ("spatial_cross_slice_cosine", "Spatial direction agreement across slices"),
        ("target_expression_effect_mean", "Named-target expression effect"),
    ]
    for ax, (field, title) in zip(axes.ravel(), fields):
        sns.barplot(data=plot, x="guide", y=field, color="#4C78A8", ax=ax)
        ax.axhline(0, color="black", lw=.8)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=55, labelsize=8)
    fig.suptitle("SPARC-seq Haruka-MENDER: response strength and cross-slice replication")
    fig.savefig(out / "spacseq_mender_biology.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
