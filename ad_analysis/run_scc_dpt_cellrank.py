#!/usr/bin/env python3
"""Replicate- and cell-class-aware DPT/CellRank analysis of Haruka s-CC.

The analysis is deliberately performed only on AD/condition cells.  Within each
replicate and each broad cell class, it:

1. builds a KNN graph from Haruka's salient CellCharter representation,
2. roots diffusion pseudotime in a plaque-distant cell,
3. constructs a CellRank 2 PseudotimeKernel,
4. infers GPCCA terminal states and fate probabilities, and
5. annotates (post hoc) which inferred terminal state is most plaque-proximal.

Plaque distance or plaque labels are never used to construct terminal states.
They are used only to orient DPT and to evaluate/annotate inferred fates.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import anndata as ad
import cellrank as cr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score


ROOT = Path("/home/yancui/haruka_v2")
OUT = ROOT / "ad_analysis" / "scc_dpt_cellrank2"

CONFIGS = {
    "starmap": {
        "h5ad": ROOT / "bench_real_discovery/data/ad.h5ad",
        "embedding": ROOT / (
            "bench_real_discovery/results_shared_full_niche_decoderonly_lambda25/ad/"
            "haruka_shared_full_niche_decoderbatch8_batchmode-decoder_only_seed0"
        ),
        "distance": ROOT / "ad_analysis/starmap_plaque_distance_cells.csv.gz",
        "replicate_key": "slice_id",
        "cluster_key": "cell_type",
        "distance_key": "distance_to_abeta_associated_cell_spatial_units",
        "near_key": "plaque_abeta",
        "condition_key": "condition",
        "condition_value": "disease",
        "spatial_scale": "native spatial units",
    },
    "codex": {
        "h5ad": ROOT / (
            "bench_codex_cns_2026/data/processed/"
            "codex_cns_myeloid_primary_no_aml_no_combat.h5ad"
        ),
        "embedding": ROOT / (
            "bench_codex_cns_2026/results_shared_full_niche_decoderonly_lambda25/"
            "codex_cns_myeloid_primary_no_aml_no_combat/"
            "haruka_z_s_cellcharter_cc3_decoderbatch8_lambda25_seed0"
        ),
        "distance": ROOT / "ad_analysis/codex_plaque_distance_cells.csv.gz",
        "replicate_key": "donor_id",
        "cluster_key": "paper_initial_leiden",
        "distance_key": "native_distance_to_any_plaque_um",
        "near_key": "plaque_within_30um",
        "condition_key": "condition",
        "condition_value": "disease",
        "spatial_scale": "micrometres",
    },
}


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
        return np.nan, np.nan
    r = spearmanr(x[ok], y[ok])
    return float(r.statistic), float(r.pvalue)


def transition_toward_plaque(tmat: sparse.spmatrix, distance: np.ndarray) -> np.ndarray:
    coo = tmat.tocoo()
    keep = distance[coo.col] < distance[coo.row]
    return np.bincount(coo.row[keep], weights=coo.data[keep], minlength=len(distance))


def binned_transition(tmat: sparse.spmatrix, pseudotime: np.ndarray, n_bins: int = 5) -> np.ndarray:
    # Rank-based bins avoid empty bins when DPT contains tied values.
    rank = pd.Series(pseudotime).rank(method="average", pct=True).to_numpy()
    bins = np.minimum((rank * n_bins).astype(int), n_bins - 1)
    coo = tmat.tocoo()
    flat = bins[coo.row] * n_bins + bins[coo.col]
    out = np.bincount(flat, weights=coo.data, minlength=n_bins * n_bins).reshape(n_bins, n_bins)
    denom = np.bincount(bins, minlength=n_bins).astype(float)
    out = np.divide(out, denom[:, None], out=np.zeros_like(out), where=denom[:, None] > 0)
    return out


def analyze_group(
    dataset: str,
    replicate: str,
    cluster: str,
    ids: np.ndarray,
    x_scc: np.ndarray,
    distance: np.ndarray,
    near: np.ndarray,
    spatial_xy: np.ndarray,
    out_dir: Path,
    seed: int,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    n = len(ids)
    n_neighbors = min(40, max(15, int(np.sqrt(n))), n - 1)
    obs = pd.DataFrame(index=pd.Index(ids, name="cell_id"))
    obs["plaque_distance"] = distance
    obs["plaque_near"] = near
    a = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32), obs=obs)
    a.obsm["X_scc"] = np.asarray(x_scc, dtype=np.float32)

    # Increase K only if needed to connect the within-replicate, within-class graph.
    for k in sorted(set([n_neighbors, min(60, n - 1), min(100, n - 1)])):
        sc.pp.neighbors(a, n_neighbors=k, use_rep="X_scc", metric="euclidean", random_state=seed)
        n_components = connected_components(a.obsp["connectivities"], directed=False, return_labels=False)
        if n_components == 1:
            n_neighbors = k
            break

    # A plaque-distant root is selected as the s-CC medoid of the farthest decile.
    far_cut = np.nanquantile(distance, 0.90)
    candidates = np.flatnonzero(distance >= far_cut)
    centroid = x_scc[candidates].mean(axis=0)
    root = int(candidates[np.argmin(np.square(x_scc[candidates] - centroid).sum(axis=1))])
    a.uns["iroot"] = root
    n_dcs = min(15, max(3, n - 2))
    sc.tl.diffmap(a, n_comps=n_dcs)
    sc.tl.dpt(a, n_dcs=min(10, n_dcs - 1))
    pseudotime = a.obs["dpt_pseudotime"].to_numpy(dtype=float)
    if not np.isfinite(pseudotime).all():
        finite = np.isfinite(pseudotime)
        pseudotime[~finite] = np.nanmax(pseudotime[finite]) if finite.any() else 0.0
        a.obs["dpt_pseudotime"] = pseudotime

    kernel = cr.kernels.PseudotimeKernel(a, time_key="dpt_pseudotime")
    kernel.compute_transition_matrix(
        threshold_scheme="soft", n_jobs=1, show_progress_bar=False, check_irreducibility=False
    )
    tmat = kernel.transition_matrix.tocsr()

    estimator = cr.estimators.GPCCA(kernel)
    n_terminal_cells = max(10, min(30, n // 8))
    # Very small groups can fail fuzzy discretization with four macrostates.
    n_macrostates = 3 if n < 160 else 4
    macrostate_fallback = False
    try:
        estimator.compute_macrostates(n_states=n_macrostates, n_cells=n_terminal_cells)
    except ValueError:
        # GPCCA's fuzzy discretization can create a state below its minimum
        # membership threshold even in an otherwise connected graph. Retrying
        # with fewer states preserves the CellRank estimator and avoids dropping
        # an entire biological replicate/region.
        macrostate_fallback = True
        estimator = cr.estimators.GPCCA(kernel)
        estimator.compute_macrostates(
            n_states=max(2, n_macrostates - 1),
            n_cells=max(6, min(n_terminal_cells, n // 12)),
        )
    estimator.predict_terminal_states(method="top_n", n_states=2, n_cells=n_terminal_cells)
    fate_solver = "gmres"
    try:
        estimator.compute_fate_probabilities(
            solver="gmres", use_petsc=False, n_jobs=1, show_progress_bar=False, tol=1e-6
        )
    except ValueError:
        # Some non-reversible pseudotime kernels are numerically ill-conditioned
        # for GMRES. CellRank explicitly supports a direct solve for these small
        # per-replicate/per-region problems.
        fate_solver = "direct"
        estimator.compute_fate_probabilities(
            solver="direct", use_petsc=False, n_jobs=1, show_progress_bar=False
        )

    terminal = estimator.terminal_states.astype(object)
    terminal_names = list(map(str, estimator.fate_probabilities.names))
    fate = np.asarray(estimator.fate_probabilities.X, dtype=float)
    terminal_distance = {}
    for terminal_name in terminal_names:
        mask = np.asarray(terminal.astype(str) == terminal_name)
        terminal_distance[terminal_name] = float(np.nanmedian(distance[mask])) if mask.any() else np.nan
    plaque_terminal = min(terminal_names, key=lambda x: terminal_distance.get(x, np.inf))
    plaque_col = terminal_names.index(plaque_terminal)
    plaque_fate = fate[:, plaque_col]

    toward = transition_toward_plaque(tmat, distance)
    dpt_rho, dpt_p = safe_spearman(pseudotime, -distance)
    fate_rho, fate_p = safe_spearman(plaque_fate, -distance)
    transition_rho, transition_p = safe_spearman(toward, -distance)
    near_bool = np.asarray(near, dtype=float) > 0
    valid_near = np.isfinite(np.asarray(near, dtype=float))
    auc_dpt = auc_fate = np.nan
    if valid_near.sum() >= 10 and np.unique(near_bool[valid_near]).size == 2:
        auc_dpt = float(roc_auc_score(near_bool[valid_near], pseudotime[valid_near]))
        auc_fate = float(roc_auc_score(near_bool[valid_near], plaque_fate[valid_near]))

    q20, q80 = np.nanquantile(distance, [0.2, 0.8])
    prox = distance <= q20
    far = distance >= q80
    u_p = np.nan
    if prox.sum() and far.sum():
        u_p = float(mannwhitneyu(plaque_fate[prox], plaque_fate[far], alternative="two-sided").pvalue)

    diff = np.asarray(a.obsm["X_diffmap"])
    cells = pd.DataFrame(
        {
            "dataset": dataset,
            "replicate": replicate,
            "main_cluster": cluster,
            "cell_id": ids,
            "plaque_distance": distance,
            "plaque_near": near,
            "dpt_pseudotime": pseudotime,
            "plaque_fate_probability": plaque_fate,
            "transition_probability_toward_plaque": toward,
            "terminal_state": np.asarray(terminal),
            "plaque_associated_terminal": plaque_terminal,
            "diffmap_1": diff[:, 1] if diff.shape[1] > 1 else diff[:, 0],
            "diffmap_2": diff[:, 2] if diff.shape[1] > 2 else np.zeros(n),
            "spatial_x": spatial_xy[:, 0],
            "spatial_y": spatial_xy[:, 1],
        }
    )
    summary = {
        "dataset": dataset,
        "replicate": replicate,
        "main_cluster": cluster,
        "n_cells": n,
        "n_neighbors": n_neighbors,
        "graph_components": int(n_components),
        "macrostate_fallback": macrostate_fallback,
        "fate_solver": fate_solver,
        "root_cell_id": str(ids[root]),
        "root_plaque_distance": float(distance[root]),
        "dpt_vs_plaque_proximity_spearman": dpt_rho,
        "dpt_vs_plaque_proximity_p": dpt_p,
        "fate_vs_plaque_proximity_spearman": fate_rho,
        "fate_vs_plaque_proximity_p": fate_p,
        "transition_vs_plaque_proximity_spearman": transition_rho,
        "transition_vs_plaque_proximity_p": transition_p,
        "dpt_plaque_near_auc": auc_dpt,
        "fate_plaque_near_auc": auc_fate,
        "proximal_minus_distal_fate": float(np.nanmean(plaque_fate[prox]) - np.nanmean(plaque_fate[far])),
        "proximal_vs_distal_fate_mwu_p": u_p,
        "mean_transition_probability_toward_plaque": float(np.nanmean(toward)),
        "plaque_associated_terminal": plaque_terminal,
        "terminal_state_distances_json": json.dumps(terminal_distance, sort_keys=True),
    }

    bt = binned_transition(tmat, pseudotime)
    trans = pd.DataFrame(bt, columns=[f"to_Q{i + 1}" for i in range(5)])
    trans.insert(0, "from_dpt_quintile", [f"Q{i + 1}" for i in range(5)])
    trans.insert(0, "main_cluster", cluster)
    trans.insert(0, "replicate", replicate)
    trans.insert(0, "dataset", dataset)

    safe_cluster = "".join(c if c.isalnum() or c in "-_" else "_" for c in cluster)
    matrix_dir = out_dir / dataset / "transition_matrices" / replicate
    matrix_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(matrix_dir / f"{safe_cluster}_cellrank_transition.npz", tmat)
    return cells, summary, trans


def load_dataset(dataset: str):
    cfg = CONFIGS[dataset]
    a = ad.read_h5ad(cfg["h5ad"])
    emb_names = np.load(cfg["embedding"] / "obs_names.npy", allow_pickle=True).astype(str)
    x_scc = np.load(cfg["embedding"] / "salient_spatial.npy")
    emb_index = pd.Series(np.arange(len(emb_names)), index=emb_names)
    distance = pd.read_csv(cfg["distance"]).set_index("cell_id")
    # Normalize numeric CODEX IDs to the obs_names string convention.
    distance.index = distance.index.astype(str)
    ids = a.obs_names.astype(str)
    obs = a.obs.copy()
    obs.index = ids
    keep = obs[cfg["condition_key"]].astype(str).eq(cfg["condition_value"])
    ids = ids[keep.to_numpy()]
    ids = np.asarray([x for x in ids if x in emb_index.index and x in distance.index], dtype=str)
    obs = obs.loc[ids]
    x = np.asarray(x_scc[emb_index.loc[ids].to_numpy()], dtype=np.float32)
    d = distance.loc[ids, cfg["distance_key"]].to_numpy(dtype=float)
    near = pd.to_numeric(obs[cfg["near_key"]], errors="coerce").to_numpy(dtype=float)
    xy = np.asarray(a[ids].obsm["spatial"], dtype=float)
    finite = np.isfinite(d) & np.isfinite(x).all(axis=1) & np.isfinite(xy).all(axis=1)
    return cfg, ids[finite], obs.iloc[np.flatnonzero(finite)], x[finite], d[finite], near[finite], xy[finite]


def run_dataset(dataset: str, out_dir: Path, min_cells: int, seed: int):
    cfg, ids, obs, x, distance, near, xy = load_dataset(dataset)
    groups = pd.DataFrame(
        {
            "replicate": obs[cfg["replicate_key"]].astype(str).to_numpy(),
            "main_cluster": obs[cfg["cluster_key"]].astype(str).to_numpy(),
        },
        index=np.arange(len(ids)),
    )
    cell_parts, summaries, transitions, skipped = [], [], [], []
    for (replicate, cluster), loc in groups.groupby(["replicate", "main_cluster"], sort=True).groups.items():
        take = np.asarray(list(loc), dtype=int)
        if len(take) < min_cells:
            skipped.append({"dataset": dataset, "replicate": replicate, "main_cluster": cluster, "n_cells": len(take)})
            continue
        print(f"[{dataset}] {replicate} / {cluster}: n={len(take)}", flush=True)
        try:
            cells, summary, trans = analyze_group(
                dataset, replicate, cluster, ids[take], x[take], distance[take], near[take], xy[take], out_dir, seed
            )
            cell_parts.append(cells)
            summaries.append(summary)
            transitions.append(trans)
        except Exception as exc:
            warnings.warn(f"Failed {dataset}/{replicate}/{cluster}: {exc}")
            skipped.append(
                {"dataset": dataset, "replicate": replicate, "main_cluster": cluster, "n_cells": len(take), "error": repr(exc)}
            )
    return (
        pd.concat(cell_parts, ignore_index=True) if cell_parts else pd.DataFrame(),
        pd.DataFrame(summaries),
        pd.concat(transitions, ignore_index=True) if transitions else pd.DataFrame(),
        pd.DataFrame(skipped),
    )


def significance_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""


def plot_summary(summary: pd.DataFrame, out_dir: Path):
    sns.set_theme(style="whitegrid", context="talk")
    for dataset, sub in summary.groupby("dataset"):
        fig, axes = plt.subplots(1, 2, figsize=(18, max(6, 0.52 * sub.main_cluster.nunique() + 3)))
        for ax, col, title in [
            (axes[0], "dpt_vs_plaque_proximity_spearman", "DPT vs Aβ/plaque proximity"),
            (axes[1], "fate_vs_plaque_proximity_spearman", "Plaque-fate probability vs proximity"),
        ]:
            mat = sub.pivot(index="main_cluster", columns="replicate", values=col)
            sns.heatmap(mat, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", ax=ax, cbar_kws={"label": "Spearman ρ"})
            ax.set_title(title)
            ax.set_xlabel("AD replicate / donor")
            ax.set_ylabel("Main biological cluster")
        fig.suptitle(f"{dataset.upper()}: replicate-resolved Haruka s-CC trajectories", y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_clusterwise_dpt_cellrank_heatmaps.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    metrics = [
        ("dpt_vs_plaque_proximity_spearman", "DPT association"),
        ("fate_vs_plaque_proximity_spearman", "Plaque-fate association"),
        ("fate_plaque_near_auc", "Plaque-near fate AUC"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        sns.boxplot(data=summary, x="dataset", y=col, hue="dataset", ax=ax, showfliers=False, legend=False)
        sns.stripplot(data=summary, x="dataset", y=col, color="black", alpha=0.55, size=4, ax=ax)
        ax.axhline(0.5 if "auc" in col else 0, color="grey", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("")
    fig.suptitle("Aβ/plaque recovery across independently fitted replicate × cell-class trajectories")
    fig.tight_layout()
    fig.savefig(out_dir / "both_ad_datasets_cellrank_summary_boxplots.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_spatial(cells: pd.DataFrame, out_dir: Path):
    for dataset, ds in cells.groupby("dataset"):
        reps = sorted(ds.replicate.unique())
        fig, axes = plt.subplots(len(reps), 3, figsize=(18, 5.3 * len(reps)), squeeze=False)
        for row, rep in enumerate(reps):
            sub = ds[ds.replicate == rep]
            panels = [
                ("plaque_distance", "viridis_r", "Aβ/plaque distance"),
                ("dpt_pseudotime", "magma", "Within-class DPT"),
                ("plaque_fate_probability", "rocket", "CellRank plaque-fate probability"),
            ]
            for ax, (col, cmap, title) in zip(axes[row], panels):
                order = np.argsort(sub[col].to_numpy())
                q = sub.iloc[order]
                sca = ax.scatter(q.spatial_x, q.spatial_y, c=q[col], s=3, cmap=cmap, linewidths=0, rasterized=True)
                ax.set_title(f"{rep}: {title}")
                ax.set_aspect("equal")
                ax.invert_yaxis()
                ax.axis("off")
                fig.colorbar(sca, ax=ax, fraction=0.035, pad=0.01)
        fig.suptitle(f"{dataset.upper()}: spatial plaque-oriented trajectories from Haruka s-CC", y=1.005)
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_replicate_spatial_dpt_fate.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_examples(cells: pd.DataFrame, summary: pd.DataFrame, out_dir: Path):
    for dataset, sds in summary.groupby("dataset"):
        ranked = sds.sort_values("fate_vs_plaque_proximity_spearman", ascending=False).head(4)
        fig, axes = plt.subplots(len(ranked), 3, figsize=(17, 4.7 * len(ranked)), squeeze=False)
        for row, rec in enumerate(ranked.itertuples()):
            sub = cells[
                (cells.dataset == dataset)
                & (cells.replicate == rec.replicate)
                & (cells.main_cluster == str(rec.main_cluster))
            ]
            for ax, col, cmap, title in [
                (axes[row, 0], "plaque_distance", "viridis_r", "Aβ/plaque distance"),
                (axes[row, 1], "dpt_pseudotime", "magma", "DPT"),
                (axes[row, 2], "plaque_fate_probability", "rocket", "Plaque-fate probability"),
            ]:
                p = ax.scatter(sub.diffmap_1, sub.diffmap_2, c=sub[col], cmap=cmap, s=9, linewidths=0, rasterized=True)
                ax.set_title(f"{rec.replicate} · {rec.main_cluster}: {title}")
                ax.set_xlabel("Diffusion component 1")
                ax.set_ylabel("Diffusion component 2")
                fig.colorbar(p, ax=ax, fraction=0.035, pad=0.01)
        fig.suptitle(f"{dataset.upper()}: strongest independently inferred plaque-associated s-CC trajectories", y=1.005)
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_top_cluster_diffusion_trajectories.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def write_readme(summary: pd.DataFrame, skipped: pd.DataFrame, out_dir: Path):
    lines = [
        "# Haruka s-CC DPT + CellRank 2 AD analysis",
        "",
        "Disease cells were analyzed separately for every replicate/donor and broad biological cluster.",
        "The KNN graph uses only `salient_spatial.npy` (Haruka salient CellCharter, s-CC).",
        "DPT is rooted in the s-CC medoid of the plaque-distant decile. CellRank's PseudotimeKernel",
        "then directs transitions along increasing DPT. GPCCA terminal states are inferred without",
        "plaque labels; the terminal state with the smallest median plaque distance is annotated post hoc",
        "as the plaque-associated fate.",
        "",
        "These are cross-sectional state-transition potentials, not lineage-traced temporal fates.",
        "",
        f"Successful replicate × cluster analyses: {len(summary)}",
        f"Skipped/failed groups: {len(skipped)}",
        "",
        "## Files",
        "",
        "- `cellrank_cluster_summary.csv`: per-analysis effect sizes, AUCs, and tests.",
        "- `cellrank_cell_scores.csv.gz`: per-cell DPT, plaque-fate probability, and transition score.",
        "- `cellrank_dpt_quintile_transition_probabilities.csv`: coarse directed transition matrices.",
        "- `*/transition_matrices/*.npz`: full sparse CellRank transition matrices.",
        "- PNG figures: cluster heatmaps, cross-dataset boxplots, spatial maps, and diffusion examples.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(CONFIGS), default=sorted(CONFIGS))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--starmap-embedding", type=Path)
    parser.add_argument("--codex-embedding", type=Path)
    parser.add_argument("--distance-root", type=Path)
    parser.add_argument("--min-cells", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.starmap_embedding is not None:
        CONFIGS["starmap"]["embedding"] = args.starmap_embedding
    if args.codex_embedding is not None:
        CONFIGS["codex"]["embedding"] = args.codex_embedding
    if args.distance_root is not None:
        CONFIGS["starmap"]["distance"] = args.distance_root / "starmap_plaque_distance_cells.csv.gz"
        CONFIGS["codex"]["distance"] = args.distance_root / "codex_plaque_distance_cells.csv.gz"
    args.out.mkdir(parents=True, exist_ok=True)
    all_cells, all_summary, all_transitions, all_skipped = [], [], [], []
    for dataset in args.datasets:
        cells, summary, transitions, skipped = run_dataset(dataset, args.out, args.min_cells, args.seed)
        all_cells.append(cells)
        all_summary.append(summary)
        all_transitions.append(transitions)
        all_skipped.append(skipped)
    cells = pd.concat(all_cells, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True)
    transitions = pd.concat(all_transitions, ignore_index=True)
    skipped = pd.concat(all_skipped, ignore_index=True)
    cells.to_csv(args.out / "cellrank_cell_scores.csv.gz", index=False)
    summary.to_csv(args.out / "cellrank_cluster_summary.csv", index=False)
    transitions.to_csv(args.out / "cellrank_dpt_quintile_transition_probabilities.csv", index=False)
    skipped.to_csv(args.out / "skipped_groups.csv", index=False)
    plot_summary(summary, args.out)
    plot_spatial(cells, args.out)
    plot_examples(cells, summary, args.out)
    write_readme(summary, skipped, args.out)
    print(f"Completed {len(summary)} trajectories; outputs: {args.out}", flush=True)


if __name__ == "__main__":
    main()
