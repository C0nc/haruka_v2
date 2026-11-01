#!/usr/bin/env python3
"""Infer expression-derived CNV burden and tumor s-CC pseudotime in KRAS Xenium.

The analysis is intentionally section-aware. infercnvpy is supplied with normal
AT1 and AT2 cells from every section as bounded reference categories. CNV burden
is the per-cell mean absolute inferred CNV signal (rather than the package's
cluster-level score). The treatment trajectory is fit only to treated tumor
cells in s-CC space; its root is the s-CC cluster closest in z-CC space to the
control-tumor centroid. DPT is descriptive and is not claimed as lineage.
"""

from __future__ import annotations

import argparse
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anndata as ad
import h5py
import infercnvpy as cnv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from infercnvpy.tl._infercnv import _infercnv_chunk
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    mean_absolute_error,
    normalized_mutual_info_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


TUMOR_STATES = ["Tumor_EMT", "Tumor_epithelial", "Tumor_epithelial_Itgb6"]
REFERENCE_STATES = ["AT1", "AT2"]
AUTOSOMES = [f"chr{i}" for i in range(1, 20)]


def decode(values):
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in values], dtype=object)


def categorical(handle: h5py.File, name: str) -> np.ndarray:
    group = handle["obs"][name]
    categories = decode(group["categories"][:])
    codes = group["codes"][:]
    out = np.full(len(codes), "NA", dtype=object)
    valid = codes >= 0
    out[valid] = categories[codes[valid]]
    return out


def read_metadata(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        obs = pd.DataFrame({
            key: categorical(handle, key)
            for key in ["celltype", "condition_label", "slice_id"]
        })
        names = decode(handle["obs"]["_index"][:])
        genes = decode(handle["var"]["_index"][:])
        spatial = np.asarray(handle["obsm"]["spatial"][:], dtype=np.float32)
    obs.index = names
    return obs, genes, spatial, names


def parse_gtf_gene_positions(path: Path) -> dict[str, tuple[str, int, int]]:
    positions: dict[str, tuple[str, int, int]] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            chromosome = fields[0]
            if chromosome not in {str(i) for i in range(1, 20)} | {"X", "Y"}:
                continue
            attrs = {}
            for item in fields[8].split(";"):
                item = item.strip()
                if not item or " " not in item:
                    continue
                key, value = item.split(" ", 1)
                attrs[key] = value.strip().strip('"')
            gene = attrs.get("gene_name")
            if gene:
                positions[gene] = (f"chr{chromosome}", int(fields[3]), int(fields[4]))
    return positions


def choose_rows(obs: pd.DataFrame, references_per_group: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tumor = obs["celltype"].isin(TUMOR_STATES).to_numpy()
    selected = np.flatnonzero(tumor).tolist()
    reference_rows = []
    for section in sorted(obs["slice_id"].unique()):
        for state in REFERENCE_STATES:
            candidates = np.flatnonzero(
                (obs["slice_id"].to_numpy() == section)
                & (obs["celltype"].to_numpy() == state)
            )
            if len(candidates):
                take = rng.choice(candidates, size=min(references_per_group, len(candidates)), replace=False)
                reference_rows.extend(take.tolist())
    selected.extend(reference_rows)
    return np.asarray(sorted(set(selected)), dtype=np.int64), np.asarray(reference_rows, dtype=np.int64)


def extract_csr_rows(path: Path, rows: np.ndarray, n_genes: int, chunk_cells: int = 25000) -> sparse.csr_matrix:
    with h5py.File(path, "r") as handle:
        n_cells = int(handle["X"].attrs["shape"][0])
        indptr = handle["X"]["indptr"][:]
        selected_mask = np.zeros(n_cells, dtype=bool)
        selected_mask[rows] = True
        parts = []
        data_ds = handle["X"]["data"]
        indices_ds = handle["X"]["indices"]
        for left in range(0, n_cells, chunk_cells):
            right = min(left + chunk_cells, n_cells)
            chosen = np.flatnonzero(selected_mask[left:right])
            if not len(chosen):
                continue
            start, stop = int(indptr[left]), int(indptr[right])
            local_ptr = indptr[left:right + 1].astype(np.int64) - start
            block = sparse.csr_matrix(
                (data_ds[start:stop], indices_ds[start:stop], local_ptr),
                shape=(right - left, n_genes),
            )
            parts.append(block[chosen])
    return sparse.vstack(parts, format="csr", dtype=np.float32)


def infercnv_threaded(
    infer: ad.AnnData,
    reference_categories: list[str],
    window_size: int,
    step: int,
    dynamic_threshold: float,
    chunk_size: int,
    n_jobs: int,
) -> None:
    """Run infercnvpy's numerical kernel without its fragile process pool.

    The upstream function sends sparse chunks through ProcessPoolExecutor. On
    this HPC host the workers can fail to start while the parent waits on a
    futex indefinitely. Threads avoid serialization and preserve the exact
    infercnvpy per-chunk calculation; results are stacked in input-row order.
    """
    expression = infer.X.tocsr() if sparse.issparse(infer.X) else np.asarray(infer.X)
    reference = np.vstack([
        np.asarray(expression[infer.obs["reference_group"].to_numpy() == category].mean(axis=0)).ravel()
        for category in reference_categories
    ]).astype(np.float32, copy=False)
    var = infer.var.loc[:, ["chromosome", "start", "end"]]
    starts = list(range(0, infer.n_obs, chunk_size))

    def run_chunk(start: int):
        return _infercnv_chunk(
            expression[start : start + chunk_size],
            var,
            reference,
            3,
            window_size,
            step,
            dynamic_threshold,
            False,
        )

    chunks = []
    chromosome_positions = None
    with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as executor:
        for completed, (positions, values, _) in enumerate(executor.map(run_chunk, starts), start=1):
            if chromosome_positions is None:
                chromosome_positions = positions
            chunks.append(values)
            if completed == 1 or completed % 10 == 0 or completed == len(starts):
                print(f"inferCNV chunks: {completed}/{len(starts)}", flush=True)
    infer.obsm["X_cnv"] = sparse.vstack(chunks, format="csr")
    infer.uns["cnv"] = {"chr_pos": chromosome_positions}


def embedding_pcs(values: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    values = StandardScaler().fit_transform(np.asarray(values, dtype=np.float32))
    return PCA(n_components=min(n_components, values.shape[1]), random_state=seed).fit_transform(values).astype(np.float32)


def grouped_embedding_benchmark(
    representation: np.ndarray,
    burden: np.ndarray,
    cnv_cluster: np.ndarray,
    groups: np.ndarray,
    name: str,
    seed: int,
) -> pd.DataFrame:
    rows = []
    logo = LeaveOneGroupOut()
    high = np.zeros(len(burden), dtype=np.int8)
    for section in np.unique(groups):
        take = groups == section
        high[take] = burden[take] >= np.quantile(burden[take], 0.75)
    for train, test in logo.split(representation, groups=groups):
        model = HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.06, max_leaf_nodes=24,
            l2_regularization=1.0, random_state=seed,
        )
        model.fit(representation[train], burden[train])
        pred = model.predict(representation[test])
        rho = spearmanr(burden[test], pred).statistic
        classifier = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced", random_state=seed)
        classifier.fit(representation[train], high[train])
        prob = classifier.predict_proba(representation[test])[:, 1]
        auc = roc_auc_score(high[test], prob) if len(np.unique(high[test])) == 2 else np.nan
        kmeans = KMeans(n_clusters=min(8, len(np.unique(cnv_cluster))), n_init=20, random_state=seed)
        pred_cluster = kmeans.fit_predict(representation[train])
        centers = kmeans.cluster_centers_
        test_cluster = ((representation[test, None, :] - centers[None, :, :]) ** 2).sum(2).argmin(1)
        rows.append({
            "representation": name,
            "held_out_slice": str(groups[test][0]),
            "n_test": len(test),
            "burden_r2": r2_score(burden[test], pred),
            "burden_spearman": rho,
            "burden_mae": mean_absolute_error(burden[test], pred),
            "high_burden_auc": auc,
            "cnv_cluster_nmi": normalized_mutual_info_score(cnv_cluster[test], test_cluster),
            "cnv_cluster_ari": adjusted_rand_score(cnv_cluster[test], test_cluster),
        })
    return pd.DataFrame(rows)


def run_pseudotime(
    s_pc: np.ndarray,
    z_pc_all_tumor: np.ndarray,
    all_tumor_rows: np.ndarray,
    treated_rows: np.ndarray,
    obs: pd.DataFrame,
    neighbors: int,
    resolution: float,
    seed: int,
) -> ad.AnnData:
    treated_pos = pd.Index(all_tumor_rows).get_indexer(treated_rows)
    control_pos = np.flatnonzero(obs.iloc[all_tumor_rows]["condition_label"].to_numpy() == "Control")
    control_centroid = z_pc_all_tumor[control_pos].mean(0)
    z_treated = z_pc_all_tumor[treated_pos]
    control_distance = np.linalg.norm(z_treated - control_centroid, axis=1)

    data = ad.AnnData(s_pc)
    data.obs = obs.iloc[treated_rows].reset_index(drop=False).rename(columns={"index": "cell_id"})
    data.obs["z_control_distance"] = control_distance
    sc.pp.neighbors(data, n_neighbors=neighbors, use_rep="X", random_state=seed)
    sc.tl.leiden(
        data, resolution=resolution, key_added="scc_cluster", random_state=seed,
        flavor="igraph", n_iterations=2, directed=False,
    )
    cluster_distance = data.obs.groupby("scc_cluster", observed=True)["z_control_distance"].median()
    root_cluster = str(cluster_distance.idxmin())
    in_root = np.flatnonzero(data.obs["scc_cluster"].astype(str).to_numpy() == root_cluster)
    root = int(in_root[np.argmin(control_distance[in_root])])
    data.uns["iroot"] = root
    data.uns["root_cluster"] = root_cluster
    sc.tl.diffmap(data)
    sc.tl.dpt(data)
    sc.tl.umap(data, min_dist=0.3, random_state=seed)
    dpt = data.obs["dpt_pseudotime"].to_numpy()
    data.obs["pseudotime_bin"] = pd.cut(
        dpt, bins=np.linspace(0, 1, 7), include_lowest=True,
        labels=[f"Q{i}" for i in range(1, 7)],
    )
    data.obs["terminal"] = dpt >= np.quantile(dpt[np.isfinite(dpt)], 0.90)
    return data


def save_umap(data: ad.AnnData, out: Path) -> None:
    xy = data.obsm["X_umap"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    panels = [
        ("dpt_pseudotime", "viridis", "Salient-CC DPT"),
        ("celltype", "tab10", "Tumor annotation"),
        ("slice_id", "tab10", "Section"),
        ("cnv_burden", "magma", "Whole-genome inferred CNV burden"),
        ("cnv_cluster", "tab10", "Inferred CNV cluster"),
        ("scc_cluster", "tab20", "Salient-CC state"),
    ]
    for ax, (key, cmap, title) in zip(axes.flat, panels):
        values = data.obs[key]
        if pd.api.types.is_numeric_dtype(values):
            plot = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=cmap, s=2, linewidths=0, rasterized=True)
            fig.colorbar(plot, ax=ax, fraction=0.046, pad=0.02)
        else:
            cats = sorted(values.astype(str).unique())
            colors = sns.color_palette(cmap if cmap != "tab10" else "tab10", len(cats))
            for cat, color in zip(cats, colors):
                take = values.astype(str).to_numpy() == cat
                ax.scatter(xy[take, 0], xy[take, 1], s=2, color=color, linewidths=0, rasterized=True, label=cat)
            ax.legend(frameon=False, fontsize=7, markerscale=4, ncol=2)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Treated KRAS tumor trajectory in Haruka salient-CC space", fontsize=16)
    fig.tight_layout()
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_spatial(data: ad.AnnData, spatial: np.ndarray, out: Path) -> None:
    sections = sorted(data.obs["slice_id"].astype(str).unique())
    fig, axes = plt.subplots(len(sections), 2, figsize=(12, 4.3 * len(sections)), squeeze=False)
    for i, section in enumerate(sections):
        take = data.obs["slice_id"].astype(str).to_numpy() == section
        rows = data.obs.loc[take, "source_row"].to_numpy(dtype=int)
        xy = spatial[rows]
        dpt = data.obs.loc[take, "dpt_pseudotime"].to_numpy()
        burden = data.obs.loc[take, "cnv_burden"].to_numpy()
        p0 = axes[i, 0].scatter(xy[:, 0], xy[:, 1], c=dpt, cmap="viridis", vmin=0, vmax=1, s=5, linewidths=0, rasterized=True)
        lo, hi = np.quantile(burden, [0.02, 0.98])
        p1 = axes[i, 1].scatter(xy[:, 0], xy[:, 1], c=burden, cmap="magma", vmin=lo, vmax=hi, s=5, linewidths=0, rasterized=True)
        axes[i, 0].set_title(f"{section}: tumor s-CC pseudotime")
        axes[i, 1].set_title(f"{section}: inferred CNV burden")
        for ax in axes[i]:
            ax.set_aspect("equal"); ax.axis("off")
        fig.colorbar(p0, ax=axes[i, 0], fraction=0.035, pad=0.01)
        fig.colorbar(p1, ax=axes[i, 1], fraction=0.035, pad=0.01)
    fig.tight_layout()
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_composition(data: ad.AnnData, out: Path) -> None:
    composition = pd.crosstab(data.obs["pseudotime_bin"], data.obs["celltype"], normalize="index")
    ax = composition.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="Set2")
    ax.set_ylabel("Tumor-state proportion")
    ax.set_xlabel("Salient-CC pseudotime bin")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad"))
    parser.add_argument("--cc-dir", type=Path, default=Path("/home/yancui/haruka_v2/bench_kras_xenium/results_cellcharter/haruka_z_s_cc3"))
    parser.add_argument("--gtf", type=Path, default=Path("/home/yancui/haruka_v2/bench_kras_xenium/resources/Mus_musculus.GRCm39.116.gtf.gz"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/yancui/haruka_v2/bench_kras_xenium/analysis/tumor_infercnv_scc_pseudotime"))
    parser.add_argument("--references-per-group", type=int, default=1000)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--cnv-chunk-size", type=int, default=2000)
    parser.add_argument("--cnv-jobs", type=int, default=4)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--resolution", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    figure_dir = args.out_dir / "figures"
    table_dir = args.out_dir / "tables"
    matrix_dir = args.out_dir / "matrices"
    for directory in [figure_dir, table_dir, matrix_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    obs, genes, spatial, names = read_metadata(args.data)
    cc_names = np.load(args.cc_dir / "obs_names.npy", allow_pickle=True).astype(str)
    if not np.array_equal(names.astype(str), cc_names):
        raise ValueError("Haruka CellCharter rows do not align with the source AnnData")
    selected_rows, reference_rows = choose_rows(obs, args.references_per_group, args.seed)
    selected_obs = obs.iloc[selected_rows].copy()
    selected_obs["source_row"] = selected_rows
    is_reference = np.isin(selected_rows, reference_rows)
    selected_obs["infercnv_role"] = np.where(is_reference, "reference", "tumor")
    selected_obs["reference_group"] = np.where(
        is_reference,
        selected_obs["slice_id"].astype(str) + "__" + selected_obs["celltype"].astype(str),
        "tumor",
    )

    positions = parse_gtf_gene_positions(args.gtf)
    gene_mask = np.asarray([gene in positions and positions[gene][0] in AUTOSOMES for gene in genes])
    kept_genes = genes[gene_mask]
    print(f"Gene-position coverage: {gene_mask.sum()}/{len(genes)}", flush=True)
    x = extract_csr_rows(args.data, selected_rows, len(genes))[:, gene_mask]
    infer = ad.AnnData(x, obs=selected_obs, var=pd.DataFrame(index=kept_genes))
    infer.var["chromosome"] = [positions[g][0] for g in kept_genes]
    infer.var["start"] = [positions[g][1] for g in kept_genes]
    infer.var["end"] = [positions[g][2] for g in kept_genes]
    reference_categories = sorted(infer.obs.loc[infer.obs["infercnv_role"] == "reference", "reference_group"].unique())
    infercnv_threaded(
        infer,
        reference_categories,
        window_size=args.window_size,
        step=args.step,
        dynamic_threshold=1.5,
        chunk_size=args.cnv_chunk_size,
        n_jobs=args.cnv_jobs,
    )
    cnv_matrix = infer.obsm["X_cnv"]
    if not sparse.issparse(cnv_matrix):
        cnv_matrix = sparse.csr_matrix(cnv_matrix)
    burden = np.asarray(abs(cnv_matrix).mean(1)).ravel()
    infer.obs["cnv_burden"] = burden
    tumor_take = infer.obs["infercnv_role"].to_numpy() == "tumor"
    tumor_cnv = cnv_matrix[tumor_take]
    n_svd = min(20, tumor_cnv.shape[1] - 1)
    cnv_pc = TruncatedSVD(n_components=n_svd, random_state=args.seed).fit_transform(tumor_cnv)
    cnv_cluster = KMeans(n_clusters=8, n_init=20, random_state=args.seed).fit_predict(cnv_pc).astype(str)
    infer.obs.loc[tumor_take, "cnv_cluster"] = cnv_cluster
    sparse.save_npz(matrix_dir / "tumor_infercnv_profile.npz", tumor_cnv)

    # infercnvpy summary heatmap before releasing the reference cells.
    infer.obs["heatmap_group"] = np.where(
        infer.obs["infercnv_role"].to_numpy() == "reference",
        "Normal_AT1_AT2_reference",
        infer.obs["celltype"].astype(str),
    )
    cnv.pl.chromosome_heatmap_summary(
        infer, groupby="heatmap_group", use_rep="cnv", figsize=(16, 5), show=False,
    )
    plt.savefig(figure_dir / "infercnv_tumor_state_summary.png", dpi=240, bbox_inches="tight")
    plt.close("all")

    all_tumor_rows = infer.obs.loc[tumor_take, "source_row"].to_numpy(dtype=int)
    tumor_obs = obs.iloc[all_tumor_rows].copy()
    tumor_obs["source_row"] = all_tumor_rows
    tumor_obs["cnv_burden"] = burden[tumor_take]
    tumor_obs["cnv_cluster"] = cnv_cluster
    tumor_obs.index.name = "cell_id"
    tumor_obs.to_csv(table_dir / "tumor_cell_infercnv_assignments.csv.gz", compression="gzip")

    z_cc = np.load(args.cc_dir / "spatial.npy", mmap_mode="r")
    s_cc = np.load(args.cc_dir / "salient_spatial.npy", mmap_mode="r")
    z_pc_all = embedding_pcs(z_cc[all_tumor_rows], 30, args.seed)
    treated_mask = tumor_obs["condition_label"].to_numpy() == "KRASi_treated"
    treated_rows = all_tumor_rows[treated_mask]
    s_pc = embedding_pcs(s_cc[treated_rows], 30, args.seed)
    z_pc_treated = z_pc_all[treated_mask]
    treated_burden = tumor_obs.loc[treated_mask, "cnv_burden"].to_numpy()
    treated_cnv_cluster = tumor_obs.loc[treated_mask, "cnv_cluster"].to_numpy()
    treated_sections = tumor_obs.loc[treated_mask, "slice_id"].to_numpy()

    benchmark = pd.concat([
        grouped_embedding_benchmark(z_pc_treated, treated_burden, treated_cnv_cluster, treated_sections, "z_cc", args.seed),
        grouped_embedding_benchmark(s_pc, treated_burden, treated_cnv_cluster, treated_sections, "s_cc", args.seed),
    ], ignore_index=True)
    benchmark.to_csv(table_dir / "zcc_scc_infercnv_leave_slice_out.csv", index=False)
    benchmark.groupby("representation").agg(
        n_slices=("held_out_slice", "nunique"),
        burden_spearman_mean=("burden_spearman", "mean"),
        burden_r2_mean=("burden_r2", "mean"),
        high_burden_auc_mean=("high_burden_auc", "mean"),
        cnv_cluster_nmi_mean=("cnv_cluster_nmi", "mean"),
        cnv_cluster_ari_mean=("cnv_cluster_ari", "mean"),
    ).to_csv(table_dir / "zcc_scc_infercnv_summary.csv")

    trajectory = run_pseudotime(
        s_pc, z_pc_all, all_tumor_rows, treated_rows, obs,
        args.neighbors, args.resolution, args.seed,
    )
    source_to_tumor_pos = pd.Series(np.arange(len(all_tumor_rows)), index=all_tumor_rows)
    positions_in_tumor = source_to_tumor_pos.loc[treated_rows].to_numpy()
    trajectory.obs["source_row"] = treated_rows
    trajectory.obs["cnv_burden"] = tumor_obs.iloc[positions_in_tumor]["cnv_burden"].to_numpy()
    trajectory.obs["cnv_cluster"] = tumor_obs.iloc[positions_in_tumor]["cnv_cluster"].to_numpy()
    trajectory.obs.to_csv(table_dir / "treated_tumor_scc_pseudotime_assignments.csv.gz", index=False, compression="gzip")

    by_section = []
    for section, frame in trajectory.obs.groupby("slice_id"):
        rho = spearmanr(frame["dpt_pseudotime"], frame["cnv_burden"]).statistic
        itgb6 = (frame["celltype"].astype(str) == "Tumor_epithelial_Itgb6").astype(int)
        auc = roc_auc_score(itgb6, frame["dpt_pseudotime"]) if itgb6.nunique() == 2 else np.nan
        by_section.append({
            "slice_id": section, "n_cells": len(frame),
            "dpt_cnv_burden_spearman": rho,
            "dpt_itgb6_auc": auc,
            "terminal_fraction": frame["terminal"].mean(),
        })
    section_metrics = pd.DataFrame(by_section)
    section_metrics.to_csv(table_dir / "pseudotime_section_metrics.csv", index=False)
    trajectory.obs.groupby(["slice_id", "celltype"], observed=True).agg(
        n_cells=("dpt_pseudotime", "size"),
        median_pseudotime=("dpt_pseudotime", "median"),
        median_cnv_burden=("cnv_burden", "median"),
        terminal_fraction=("terminal", "mean"),
    ).reset_index().to_csv(table_dir / "pseudotime_by_section_tumor_state.csv", index=False)
    pd.crosstab(
        trajectory.obs["pseudotime_bin"], trajectory.obs["celltype"], normalize="index",
    ).to_csv(table_dir / "tumor_state_fraction_by_pseudotime_bin.csv")

    save_umap(trajectory, figure_dir / "tumor_scc_umap_pseudotime_cnv.png")
    save_spatial(trajectory, spatial, figure_dir / "treated_tumor_spatial_pseudotime_cnv.png")
    save_composition(trajectory, figure_dir / "tumor_state_composition_over_scc_pseudotime.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=benchmark, x="representation", y="burden_spearman", ax=ax, color="#bfdbfe")
    sns.stripplot(data=benchmark, x="representation", y="burden_spearman", hue="held_out_slice", ax=ax, size=7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Held-out-section Spearman: predicted vs inferred CNV burden")
    ax.set_xlabel("")
    ax.legend(title="Held-out section", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(figure_dir / "zcc_scc_cnv_burden_benchmark.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "data": str(args.data), "cc_dir": str(args.cc_dir), "gtf": str(args.gtf),
        "n_genes_input": int(len(genes)), "n_genes_with_autosomal_position": int(gene_mask.sum()),
        "n_tumor": int(tumor_take.sum()), "n_reference": int((~tumor_take).sum()),
        "n_treated_tumor": int(len(treated_rows)), "reference_categories": reference_categories,
        "infercnv": {
            "window_size": args.window_size,
            "step": args.step,
            "dynamic_threshold": 1.5,
            "chunk_size": args.cnv_chunk_size,
            "n_jobs": args.cnv_jobs,
            "executor": "threaded infercnvpy numerical kernel",
        },
        "cnv_burden": "per-cell mean absolute infercnvpy signal across all inferred autosomal windows",
        "pseudotime": "DPT on treated tumor s-CC; root cluster minimizes median z-CC distance to control tumor centroid",
        "root_cluster": trajectory.uns["root_cluster"],
        "interpretation_limit": "cross-sectional expression-derived ordering; neither DNA-validated CNV nor lineage",
    }
    (args.out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("\nEmbedding benchmark summary:\n", pd.read_csv(table_dir / "zcc_scc_infercnv_summary.csv").to_string(index=False), flush=True)
    print("\nPseudotime section metrics:\n", section_metrics.to_string(index=False), flush=True)
    print(f"Outputs: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
