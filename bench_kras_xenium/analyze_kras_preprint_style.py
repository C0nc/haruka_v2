#!/usr/bin/env python3
"""Preprint-style biological analysis of new Haruka on KRASi Xenium lung.

The Haruka fit is Control versus MRTX1133-treated.  All unsupervised response
clusters are learned only within treated sections.  Cell type and compartment
are used as post-fit annotations, with the important caveat that annotated
cell-type neighbourhood composition is one of Haruka's reconstruction targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    balanced_accuracy_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


TUMOR_TYPES = ["Tumor_epithelial", "Tumor_epithelial_Itgb6", "Tumor_EMT"]
PROGRAMS = {
    "MAPK_AP1": ["Dusp1", "Dusp4", "Egr1", "Fosl1", "Fosl2", "Spry1", "Spry2", "Etv4", "Etv5"],
    "EMT": ["Vim", "Fn1", "Sparc", "Spp1", "Tagln", "Acta2", "Tgfbr1", "Tgfbr2"],
    "epithelial_Itgb6": ["Itgb6", "Epcam", "Krt2", "Krt20", "Sox9", "Muc1", "Klf5", "Yap1", "Tead1"],
    "cycling": ["Mki67", "Top2a", "Cdk1", "Ccnb1", "Pcna", "Mcm2"],
    "interferon": ["Stat1", "Irf1", "Isg15", "Ifit1", "Ifit3", "Cxcl10"],
    "antigen_presentation": ["Tap1", "Ciita", "Psmb9", "H2-DMb2"],
    "hypoxia_stress": ["Hif1a", "Vegfa", "Slc2a1", "Ldha", "Hmox1", "Ndrg1"],
    "myeloid_suppression": ["Arg1", "Trem2", "Mrc1", "Lgals9", "Ccl2", "Ccl7", "Cxcl2", "Tnf"],
}


def balanced_rows(obs: pd.DataFrame, mask: np.ndarray, per_slice: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    slices = obs["slice_id"].astype(str).to_numpy()
    selected: list[np.ndarray] = []
    for value in sorted(np.unique(slices[mask])):
        rows = np.flatnonzero(mask & (slices == value))
        if len(rows) > per_slice:
            rows = rng.choice(rows, per_slice, replace=False)
        selected.append(np.sort(rows))
    return np.concatenate(selected)


def scatter_cat(ax, xy: np.ndarray, values: np.ndarray, title: str, size: float = 1.0) -> None:
    values = np.asarray(values).astype(str)
    cats = sorted(np.unique(values))
    colors = sns.color_palette("husl", len(cats))
    for cat, color in zip(cats, colors):
        take = values == cat
        ax.scatter(xy[take, 0], xy[take, 1], s=size, color=color, label=cat,
                   linewidths=0, rasterized=True)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


def cluster_all(rep: np.ndarray, fit_rows: np.ndarray, target: np.ndarray,
                n_clusters: int, seed: int) -> tuple[np.ndarray, StandardScaler, MiniBatchKMeans]:
    scaler = StandardScaler().fit(np.asarray(rep[fit_rows]))
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=20,
                            batch_size=8192).fit(scaler.transform(np.asarray(rep[fit_rows])))
    labels = np.full(len(target), -1, dtype=np.int16)
    for left in range(0, len(target), 100_000):
        rows = np.arange(left, min(left + 100_000, len(target)))
        rows = rows[target[rows]]
        if len(rows):
            labels[rows] = model.predict(scaler.transform(np.asarray(rep[rows])))
    return labels, scaler, model


def cluster_metrics(rep: np.ndarray, obs: pd.DataFrame, rows: np.ndarray,
                    representation: str, seed: int) -> list[dict]:
    scaled = StandardScaler().fit_transform(np.asarray(rep[rows]))
    results: list[dict] = []
    for key in ["celltype", "compartment", "slice_id"]:
        truth = obs.iloc[rows][key].astype(str).to_numpy()
        k = len(np.unique(truth))
        pred = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10,
                               batch_size=8192).fit_predict(scaled)
        results.append({"representation": representation, "label": key,
                        "n_clusters": k, "ari": adjusted_rand_score(truth, pred),
                        "nmi": normalized_mutual_info_score(truth, pred),
                        "evaluation_subset": "balanced treated-cell sample"})
    return results


def enrichment_matrix(labels: np.ndarray, annotation: np.ndarray, n_clusters: int) -> pd.DataFrame:
    table = pd.crosstab(pd.Series(labels, name="cluster"), pd.Series(annotation, name="label"))
    table = table.reindex(range(n_clusters), fill_value=0)
    observed = table.div(table.sum(1).replace(0, np.nan), axis=0)
    expected = table.sum(0) / table.to_numpy().sum()
    return np.log2((observed + 1e-4).div(expected + 1e-4, axis=1))


def heldout_slice_classifier(rep: np.ndarray, obs: pd.DataFrame, rows: np.ndarray,
                             representation: str) -> tuple[dict, np.ndarray]:
    block = obs.iloc[rows]
    y = (block["celltype"].astype(str) == "Tumor_epithelial_Itgb6").astype(int).to_numpy()
    groups = block["slice_id"].astype(str).to_numpy()
    predictions = np.full(len(rows), np.nan)
    for group in np.unique(groups):
        train = groups != group
        test = ~train
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        scaler = StandardScaler().fit(np.asarray(rep[rows[train]]))
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                                   random_state=0).fit(scaler.transform(rep[rows[train]]), y[train])
        predictions[test] = model.predict_proba(scaler.transform(rep[rows[test]]))[:, 1]
    valid = np.isfinite(predictions)
    return ({"representation": representation, "endpoint": "Itgb6+ residual tumor vs canonical tumor",
             "n_cells": int(valid.sum()), "n_sections": int(pd.Series(groups[valid]).nunique()),
             "roc_auc": roc_auc_score(y[valid], predictions[valid]),
             "average_precision": average_precision_score(y[valid], predictions[valid]),
             "balanced_accuracy": balanced_accuracy_score(y[valid], predictions[valid] >= 0.5),
             "validation": "leave-one-treated-section-out",
             "caveat": "celltype-informed niche reconstruction makes this an internal biological readout, not an independent validation"}, predictions)


def program_scores(data: ad.AnnData, rows: np.ndarray) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    genes = pd.Index(data.var_names.astype(str))
    present = {name: [g for g in members if g in genes] for name, members in PROGRAMS.items()}
    chosen = sorted(set(g for members in present.values() for g in members))
    if not chosen:
        return pd.DataFrame(index=np.arange(len(rows))), present
    positions = genes.get_indexer(chosen)
    scores = {name: np.zeros(len(rows), dtype=np.float32) for name in present}
    gene_to_col = {g: j for j, g in enumerate(chosen)}
    for left in range(0, len(rows), 40_000):
        rr = rows[left:left + 40_000]
        block = data[rr].to_memory()
        full = block.layers["counts"] if "counts" in block.layers else block.X
        matrix = full[:, positions]
        if sparse.issparse(full):
            full = full.tocsr().astype(np.float32)
            matrix = full[:, positions]
            library = np.asarray(full.sum(1)).ravel().astype(np.float32)
            matrix = matrix.multiply((10_000.0 / np.maximum(library, 1.0))[:, None]).tocsr()
            matrix.data = np.log1p(matrix.data)
            matrix = matrix.toarray()
        else:
            full = np.asarray(full, dtype=np.float32)
            matrix = full[:, positions]
            library = full.sum(1).astype(np.float32)
            matrix = np.log1p(matrix * (10_000.0 / np.maximum(library, 1.0))[:, None])
        for name, members in present.items():
            cols = [gene_to_col[g] for g in members]
            if cols:
                scores[name][left:left + len(rr)] = matrix[:, cols].mean(1)
    return pd.DataFrame(scores), present


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cc-dir", type=Path, required=True)
    ap.add_argument("--haruka-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--clusters", type=int, default=12)
    ap.add_argument("--plot-per-slice", type=int, default=3000)
    ap.add_argument("--fit-per-slice", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    fig_dir = args.out_dir / "figures"; table_dir = args.out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True); table_dir.mkdir(parents=True, exist_ok=True)

    data = ad.read_h5ad(args.data, backed="r")
    obs_full = data.obs.copy()
    names = np.load(args.cc_dir / "obs_names.npy", allow_pickle=True).astype(str)
    source_rows_path = args.cc_dir / "source_rows.npy"
    source_rows = (
        np.load(source_rows_path).astype(int)
        if source_rows_path.exists()
        else np.arange(len(obs_full), dtype=int)
    )
    obs = obs_full.iloc[source_rows].copy()
    if not np.array_equal(names, obs.index.astype(str).to_numpy()):
        raise ValueError("CellCharter rows do not align with AnnData")
    z_cc = np.load(args.cc_dir / "spatial.npy", mmap_mode="r")
    s_cc = np.load(args.cc_dir / "salient_spatial.npy", mmap_mode="r")
    treated = obs["condition_label"].astype(str).to_numpy() == "KRASi_treated"
    plot_rows = balanced_rows(obs, treated, args.plot_per_slice, args.seed)
    fit_rows = balanced_rows(obs, treated, args.fit_per_slice, args.seed + 1)

    z_labels, z_scaler, _ = cluster_all(z_cc, fit_rows, treated, args.clusters, args.seed)
    s_labels, s_scaler, _ = cluster_all(s_cc, fit_rows, treated, args.clusters, args.seed)
    z_umap = umap.UMAP(n_neighbors=30, min_dist=0.25, random_state=args.seed).fit_transform(
        z_scaler.transform(np.asarray(z_cc[plot_rows])))
    s_umap = umap.UMAP(n_neighbors=30, min_dist=0.25, random_state=args.seed).fit_transform(
        s_scaler.transform(np.asarray(s_cc[plot_rows])))

    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    scatter_cat(axes[0, 0], z_umap, obs.iloc[plot_rows]["celltype"], "Background z-CC — cell type")
    scatter_cat(axes[0, 1], z_umap, obs.iloc[plot_rows]["compartment"], "Background z-CC — compartment")
    scatter_cat(axes[0, 2], z_umap, obs.iloc[plot_rows]["slice_id"], "Background z-CC — section")
    scatter_cat(axes[1, 0], s_umap, s_labels[plot_rows].astype(str), "Salient s-CC — response cluster")
    scatter_cat(axes[1, 1], s_umap, obs.iloc[plot_rows]["celltype"], "Salient s-CC — cell type")
    scatter_cat(axes[1, 2], s_umap, obs.iloc[plot_rows]["slice_id"], "Salient s-CC — section")
    for ax in axes.flat:
        ax.legend(markerscale=5, fontsize=5.5, frameon=False, ncol=2, loc="best")
    fig.suptitle("MRTX1133-treated lung only: new Haruka background and salient niches", fontsize=16)
    fig.tight_layout(); fig.savefig(fig_dir / "treated_only_zcc_scc_umap.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    coordinates = np.asarray(data.obsm["spatial"])[source_rows]
    treated_slices = sorted(obs.loc[treated, "slice_id"].astype(str).unique())
    fig, axes = plt.subplots(len(treated_slices), 2, figsize=(14, 5 * len(treated_slices)), squeeze=False)
    slice_values = obs["slice_id"].astype(str).to_numpy()
    for i, sid in enumerate(treated_slices):
        rows = np.flatnonzero(treated & (slice_values == sid))
        axes[i, 0].scatter(coordinates[rows, 0], coordinates[rows, 1], c=z_labels[rows], cmap="tab20",
                           s=0.8, linewidths=0, rasterized=True)
        axes[i, 1].scatter(coordinates[rows, 0], coordinates[rows, 1], c=s_labels[rows], cmap="tab20",
                           s=0.8, linewidths=0, rasterized=True)
        axes[i, 0].set_title(f"{sid}: background z-CC domains")
        axes[i, 1].set_title(f"{sid}: salient s-CC response domains")
        for ax in axes[i]: ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); fig.savefig(fig_dir / "treated_spatial_zcc_scc_clusters.png", dpi=240, bbox_inches="tight"); plt.close(fig)

    metrics = cluster_metrics(z_cc, obs, plot_rows, "z_cc", args.seed)
    metrics += cluster_metrics(s_cc, obs, plot_rows, "s_cc", args.seed)
    pd.DataFrame(metrics).to_csv(table_dir / "zcc_scc_clustering_metrics.csv", index=False)

    assignments = obs[["slice_id", "condition_label", "celltype", "compartment"]].copy()
    assignments["z_cc_cluster"] = z_labels; assignments["s_cc_cluster"] = s_labels
    assignments.to_csv(table_dir / "cell_cluster_assignments.csv.gz", compression="gzip")
    for rep_name, labels in [("z_cc", z_labels), ("s_cc", s_labels)]:
        take = treated & (labels >= 0)
        for annotation in ["celltype", "compartment"]:
            enr = enrichment_matrix(labels[take], obs.loc[take, annotation].astype(str).to_numpy(), args.clusters)
            enr.to_csv(table_dir / f"{rep_name}_{annotation}_log2_enrichment.csv")
            fig, ax = plt.subplots(figsize=(max(9, 0.32 * enr.shape[1]), 7))
            sns.heatmap(enr, cmap="vlag", center=0, ax=ax, cbar_kws={"label": "log2 observed / expected"})
            ax.set_title(f"{rep_name} cluster enrichment — {annotation} (treated)")
            fig.tight_layout(); fig.savefig(fig_dir / f"{rep_name}_{annotation}_enrichment.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    cluster_fraction = (assignments.loc[treated].groupby(["slice_id", "s_cc_cluster"], observed=True)
                        .size().rename("n").reset_index())
    cluster_fraction["fraction"] = cluster_fraction["n"] / cluster_fraction.groupby("slice_id")["n"].transform("sum")
    cluster_fraction.to_csv(table_dir / "s_cc_cluster_fraction_by_section.csv", index=False)

    composition_shift_full = np.load(args.haruka_dir / "salient_composition_shift.npy", mmap_mode="r")
    composition_shift = composition_shift_full[source_rows]
    categories = json.loads((args.haruka_dir / "meta.json").read_text())["cell_type_categories"]
    shift_means = []
    for cluster_id in range(args.clusters):
        rows = np.flatnonzero(s_labels == cluster_id)
        shift_means.append(np.asarray(composition_shift[rows], dtype=np.float64).mean(0))
    shift_means = np.vstack(shift_means)
    shift_frame = pd.DataFrame(shift_means, columns=categories)
    shift_frame.insert(0, "s_cc_cluster", np.arange(args.clusters))
    shift_frame.to_csv(table_dir / "s_cc_decoded_composition_shift.csv", index=False)
    top = np.argsort(np.max(np.abs(shift_means), axis=0))[-24:]
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(shift_means[:, top], cmap="vlag", center=0, xticklabels=np.asarray(categories)[top],
                yticklabels=[f"S{i}" for i in range(args.clusters)], ax=ax,
                cbar_kws={"label": "D(z,s) - D(z,0)"})
    ax.set_title("Decoded salient neighborhood-composition shifts")
    fig.tight_layout(); fig.savefig(fig_dir / "salient_decoded_composition_shift.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    tumor = treated & obs["celltype"].astype(str).isin(["Tumor_epithelial", "Tumor_epithelial_Itgb6"]).to_numpy()
    tumor_rows = balanced_rows(obs, tumor, min(args.fit_per_slice, 12000), args.seed + 2)
    resistance_metrics = []
    predictions = {}
    for name, rep in [("z_cc", z_cc), ("s_cc", s_cc)]:
        result, prediction = heldout_slice_classifier(rep, obs, tumor_rows, name)
        resistance_metrics.append(result); predictions[name] = prediction
    pd.DataFrame(resistance_metrics).to_csv(table_dir / "residual_tumor_state_metrics.csv", index=False)

    response_norm = np.linalg.norm(np.asarray(s_cc[plot_rows]), axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    scatter_cat(axes[0], s_umap, obs.iloc[plot_rows]["celltype"], "s-CC tumor/microenvironment states")
    pts = axes[1].scatter(s_umap[:, 0], s_umap[:, 1], c=response_norm, cmap="magma", s=1.2,
                          linewidths=0, rasterized=True)
    axes[1].set_title("Salient response magnitude"); axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.colorbar(pts, ax=axes[1], fraction=0.045, label="||s-CC||2")
    metric_df = pd.DataFrame(resistance_metrics)
    axes[2].bar(metric_df["representation"], metric_df["roc_auc"], color=["#4c72b0", "#dd8452"])
    axes[2].axhline(0.5, color="black", ls="--", lw=1); axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Leave-one-section-out ROC AUC")
    axes[2].set_title("Itgb6+ residual tumor-state separation")
    axes[0].legend(markerscale=5, fontsize=5, ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(fig_dir / "residual_tumor_response_summary.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    score_rows = balanced_rows(obs, treated, min(args.fit_per_slice, 8000), args.seed + 3)
    scores, present_programs = program_scores(data, source_rows[score_rows])
    scores["s_cc_cluster"] = s_labels[score_rows]
    scores["slice_id"] = obs.iloc[score_rows]["slice_id"].astype(str).to_numpy()
    scores["celltype"] = obs.iloc[score_rows]["celltype"].astype(str).to_numpy()
    scores.to_csv(table_dir / "sampled_response_program_scores.csv.gz", index=False, compression="gzip")
    program_cols = list(PROGRAMS)
    program_cluster = scores.groupby("s_cc_cluster", observed=True)[program_cols].mean()
    program_z = program_cluster.sub(program_cluster.mean(0), axis=1).div(program_cluster.std(0).replace(0, 1), axis=1)
    program_cluster.to_csv(table_dir / "s_cc_cluster_program_means.csv")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(program_z, cmap="vlag", center=0, ax=ax, cbar_kws={"label": "program-wise z-score"})
    ax.set_title("KRASi salient-cluster gene programs")
    fig.tight_layout(); fig.savefig(fig_dir / "s_cc_cluster_program_heatmap.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    # Preprint-style section-level summary. Interpretation is descriptive because
    # arm is inseparable from Xenium run and only 3 control/4 treated sections exist.
    section_records = []
    for sid in sorted(obs["slice_id"].astype(str).unique()):
        rows = np.flatnonzero(slice_values == sid)
        section_records.append({"slice_id": sid, "condition": str(obs.iloc[rows]["condition_label"].iloc[0]),
                                "n_cells": len(rows), "mean_z_cc_norm": float(np.linalg.norm(z_cc[rows], axis=1).mean()),
                                "mean_s_cc_norm": float(np.linalg.norm(s_cc[rows], axis=1).mean()),
                                "mean_decoded_shift_norm": float(np.linalg.norm(composition_shift[rows], axis=1).mean())})
    section_frame = pd.DataFrame(section_records)
    section_frame.to_csv(table_dir / "section_level_response_summary.csv", index=False)
    section_vectors = []
    for sid in section_frame["slice_id"]:
        rows = np.flatnonzero(slice_values == sid)
        section_vectors.append(np.r_[np.asarray(z_cc[rows]).mean(0), np.asarray(s_cc[rows]).mean(0)])
    section_xy = PCA(n_components=2, random_state=args.seed).fit_transform(StandardScaler().fit_transform(np.vstack(section_vectors)))
    fig, ax = plt.subplots(figsize=(7, 6))
    for condition, color in [("Control", "#4c72b0"), ("KRASi_treated", "#c44e52")]:
        take = section_frame["condition"] == condition
        ax.scatter(section_xy[take, 0], section_xy[take, 1], s=100, label=condition, color=color)
    for i, sid in enumerate(section_frame["slice_id"]): ax.text(section_xy[i, 0], section_xy[i, 1], sid, fontsize=9)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Section-level Haruka niche summary (descriptive)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(fig_dir / "section_level_pca.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    metadata = {
        "training_contrast": "Control vs MRTX1133 (KRAS G12D inhibitor)",
        "all_sections_cotrained": True,
        "holdout_training": False,
        "cellcharter": "three slice-bounded hop blocks; own-cell block removed",
        "response_clustering": "K-means on treated s-CC only",
        "background_clustering": "K-means on treated z-CC only",
        "batch_correction": "decoder-only slice_id injection with separate expression/composition/niche projections",
        "critical_design_caveat": "one run per arm; section-level observations are not independent biological replicates",
        "annotation_caveat": "celltype-informed niche reconstruction; annotation enrichments are interpretation, not independent validation",
        "present_program_genes": present_programs,
        "n_cells": len(obs), "n_control": int((~treated).sum()), "n_treated": int(treated.sum()),
        "condition_only_cellcharter": bool(source_rows_path.exists()),
        "source_n_cells": len(obs_full),
    }
    (args.out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))
    pd.DataFrame({"row": plot_rows, "z_umap1": z_umap[:, 0], "z_umap2": z_umap[:, 1],
                  "s_umap1": s_umap[:, 0], "s_umap2": s_umap[:, 1]}).to_csv(
        table_dir / "treated_umap_coordinates.csv.gz", index=False, compression="gzip")
    print(json.dumps({"metadata": metadata, "resistance_metrics": resistance_metrics,
                      "figures": sorted(str(p) for p in fig_dir.glob("*.png"))}, indent=2))


if __name__ == "__main__":
    main()
