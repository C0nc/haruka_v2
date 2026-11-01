#!/usr/bin/env python3
"""Reproduce the Haruka preprint lung/SLAT analyses on KRAS Xenium.

The manuscript notebooks use (i) background/salient domain enrichment,
(ii) slice-level PCA of salient-domain frequencies, and (iii) SLAT alignment
on the background CellCharter representation followed by a high-confidence
source-state -> target-state Sankey and source-cell differential expression.

This implementation preserves that logic but evaluates every Control x
MRTX1133-treated section pair.  Because the sections are not longitudinally
paired and condition is confounded with Xenium run, all transitions are
reported as cross-sectional associations, never lineage or causal fates.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import warnings
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
from scipy import sparse
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scSLAT.model import Cal_Spatial_Net, load_anndatas, run_SLAT, spatial_match


TUMOR_TYPES = ["Tumor_epithelial", "Tumor_epithelial_Itgb6", "Tumor_EMT"]
REFERENCE_FATE = "Tumor_epithelial"


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    good = np.isfinite(p)
    if not good.any():
        return out
    vals = p[good]
    order = np.argsort(vals)
    ranked = vals[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1].clip(0, 1)
    restored = np.empty_like(q)
    restored[order] = q
    out[good] = restored
    return out


def sample_slice(obs: pd.DataFrame, sid: str, n: int, seed: int) -> np.ndarray:
    rows = np.flatnonzero(obs["slice_id"].astype(str).to_numpy() == sid)
    if len(rows) <= n:
        return rows
    return np.sort(np.random.default_rng(seed).choice(rows, n, replace=False))


def enrichment(labels: np.ndarray, annotation: np.ndarray) -> pd.DataFrame:
    table = pd.crosstab(pd.Series(labels, name="cluster"), pd.Series(annotation, name="label"))
    expected = np.outer(table.sum(1), table.sum(0)) / max(table.to_numpy().sum(), 1)
    score = np.log2((table.to_numpy() + 1.0) / (expected + 1.0))
    return pd.DataFrame(score, index=table.index, columns=table.columns)


def fit_global_clusters(z_cc: np.ndarray, s_cc: np.ndarray, obs: pd.DataFrame,
                        n_clusters: int, fit_per_slice: int, seed: int) -> pd.DataFrame:
    """Fit paper-style domains jointly and assign every control/treated cell."""
    slices = sorted(obs["slice_id"].astype(str).unique())
    fit_rows = np.concatenate([
        sample_slice(obs, sid, fit_per_slice, seed + i * 211)
        for i, sid in enumerate(slices)
    ])
    result = pd.DataFrame(index=obs.index.astype(str))
    for offset, (key, rep) in enumerate([("z_cc_cluster", z_cc), ("s_cc_cluster", s_cc)]):
        scaler = StandardScaler().fit(np.asarray(rep[fit_rows]))
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed + offset,
                                n_init=20, batch_size=8192).fit(
                                    scaler.transform(np.asarray(rep[fit_rows])))
        labels = np.empty(len(obs), dtype=np.int16)
        for left in range(0, len(obs), 100_000):
            right = min(left + 100_000, len(obs))
            labels[left:right] = model.predict(
                scaler.transform(np.asarray(rep[left:right])))
        result[key] = labels
    result["slice_id"] = obs["slice_id"].astype(str).to_numpy()
    result["condition_label"] = obs["condition_label"].astype(str).to_numpy()
    return result


def plot_enrichment(assignments: pd.DataFrame, obs: pd.DataFrame, treated: np.ndarray,
                    fig_dir: Path, table_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(22, 12))
    for row, (rep, cluster_key) in enumerate(
        [("background", "z_cc_cluster"), ("salient", "s_cc_cluster")]
    ):
        for col, annotation in enumerate(["celltype", "compartment"]):
            mat = enrichment(
                assignments.loc[treated, cluster_key].to_numpy(),
                obs.loc[treated, annotation].astype(str).to_numpy(),
            )
            mat.to_csv(table_dir / f"{rep}_{annotation}_log2_enrichment.csv")
            sns.heatmap(mat, center=0, cmap="vlag", ax=axes[row, col],
                        cbar_kws={"label": "log2 observed/expected"})
            axes[row, col].set_title(f"{rep.capitalize()} domains: {annotation} enrichment")
            axes[row, col].set_xlabel(annotation)
            axes[row, col].set_ylabel("domain")
    fig.suptitle("Paper-style domain enrichment (treated sections)", fontsize=16)
    fig.tight_layout()
    fig.savefig(fig_dir / "domain_enrichment_background_salient.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def orient_pc(pc: np.ndarray, condition: np.ndarray) -> np.ndarray:
    treated = condition == "KRASi_treated"
    if pc[treated].mean() < pc[~treated].mean():
        return -pc
    return pc


def section_pca(z_cc: np.ndarray, s_cc: np.ndarray, assignments: pd.DataFrame,
                obs: pd.DataFrame, fig_dir: Path, table_dir: Path) -> None:
    slice_ids = sorted(obs["slice_id"].astype(str).unique())
    condition = np.array([
        obs.loc[obs["slice_id"].astype(str) == sid, "condition_label"].astype(str).iloc[0]
        for sid in slice_ids
    ])
    slice_values = obs["slice_id"].astype(str).to_numpy()
    records: list[dict] = []
    panels: list[tuple[str, np.ndarray]] = []

    for rep, cluster_key, embedding in [
        ("z_cluster_frequency", "z_cc_cluster", z_cc),
        ("s_cluster_frequency", "s_cc_cluster", s_cc),
    ]:
        freq = (assignments.assign(slice_id=slice_values)
                .groupby(["slice_id", cluster_key], observed=True).size()
                .unstack(fill_value=0).reindex(slice_ids).astype(float))
        freq = freq.div(freq.sum(1), axis=0).to_numpy()
        panels.append((rep, freq))

    for rep, embedding in [("z_cc_mean", z_cc), ("s_cc_mean", s_cc)]:
        means = np.vstack([
            np.asarray(embedding[np.flatnonzero(slice_values == sid)]).mean(0)
            for sid in slice_ids
        ])
        panels.append((rep, means))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = {"Control": "#4C78A8", "KRASi_treated": "#E45756"}
    for ax, (name, matrix) in zip(axes.flat, panels):
        scaled = StandardScaler().fit_transform(matrix)
        coords = PCA(n_components=2, random_state=0).fit_transform(scaled)
        coords[:, 0] = orient_pc(coords[:, 0], condition)
        control_pc = coords[condition == "Control", 0]
        treated_pc = coords[condition == "KRASi_treated", 0]
        stat, pval = mannwhitneyu(control_pc, treated_pc, alternative="two-sided", method="exact")
        pooled = np.sqrt((control_pc.var(ddof=1) + treated_pc.var(ddof=1)) / 2)
        d = (treated_pc.mean() - control_pc.mean()) / pooled if pooled > 0 else np.nan
        for sid, cond, xy in zip(slice_ids, condition, coords):
            ax.scatter(*xy, s=100, color=colors[cond], edgecolor="black", linewidth=0.6)
            ax.annotate(sid, xy, xytext=(4, 4), textcoords="offset points", fontsize=9)
            records.append({"representation": name, "slice_id": sid, "condition": cond,
                            "PC1": xy[0], "PC2": xy[1], "mannwhitney_p": pval,
                            "cohens_d": d})
        ax.set_title(f"{name.replace('_', ' ')}\nPC1 treatment d={d:.2f}, exact p={pval:.3g}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=v,
                          markeredgecolor="black", markersize=9, label=k)
               for k, v in colors.items()]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Section-level PCA following the Haruka lung notebook", y=1.01, fontsize=16)
    fig.tight_layout()
    fig.savefig(fig_dir / "section_level_paper_pca.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(records).to_csv(table_dir / "section_level_pca_coordinates_metrics.csv", index=False)


def build_pair_adata(features: np.ndarray, coords: np.ndarray, names: np.ndarray) -> ad.AnnData:
    out = ad.AnnData(X=np.asarray(features, dtype=np.float32))
    out.obs_names = pd.Index(names.astype(str))
    out.obsm["spatial"] = np.asarray(coords, dtype=np.float32)
    Cal_Spatial_Net(out, k_cutoff=20, model="KNN", verbose=False)
    return out


def run_all_slat_pairs(obs: pd.DataFrame, coords: np.ndarray, z_cc: np.ndarray,
                       assignments: pd.DataFrame, pair_dir: Path, per_slice: int,
                       seed: int, epochs: int, hidden_size: int) -> pd.DataFrame:
    slice_values = obs["slice_id"].astype(str).to_numpy()
    condition = obs["condition_label"].astype(str).to_numpy()
    controls = sorted(np.unique(slice_values[condition == "Control"]))
    treated = sorted(np.unique(slice_values[condition == "KRASi_treated"]))
    sampled = {sid: sample_slice(obs, sid, per_slice, seed + i * 103)
               for i, sid in enumerate(controls + treated)}
    np.savez_compressed(pair_dir / "sampled_rows_by_section.npz", **sampled)
    all_records: list[pd.DataFrame] = []

    for pair_i, (control_sid, treated_sid) in enumerate(
        [(c, t) for c in controls for t in treated]
    ):
        cache = pair_dir / f"{control_sid}__{treated_sid}.csv.gz"
        if cache.exists():
            print(f"[resume] {cache.name}", flush=True)
            frame = pd.read_csv(cache)
            source_idx = frame.source_row.to_numpy(dtype=int)
            target_idx = frame.target_row.to_numpy(dtype=int)
            frame["source_z_cluster"] = assignments.iloc[source_idx]["z_cc_cluster"].to_numpy()
            frame["target_z_cluster"] = assignments.iloc[target_idx]["z_cc_cluster"].to_numpy()
            frame["source_s_cluster"] = assignments.iloc[source_idx]["s_cc_cluster"].to_numpy()
            frame["target_s_cluster"] = assignments.iloc[target_idx]["s_cc_cluster"].to_numpy()
            frame.to_csv(cache, index=False, compression="gzip")
            all_records.append(frame)
            continue
        source_rows = sampled[control_sid]
        target_rows = sampled[treated_sid]
        random.seed(seed + pair_i)
        np.random.seed(seed + pair_i)
        torch.manual_seed(seed + pair_i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + pair_i)

        source = build_pair_adata(z_cc[source_rows], coords[source_rows], obs.index[source_rows].to_numpy())
        target = build_pair_adata(z_cc[target_rows], coords[target_rows], obs.index[target_rows].to_numpy())
        edges, features = load_anndatas([source, target], feature="raw", check_order=False)
        emb_source, emb_target, runtime = run_SLAT(
            features, edges, epochs=epochs, LGCN_layer=5,
            hidden_size=hidden_size,
        )
        best, _, similarities = spatial_match(
            [emb_source, emb_target], reorder=False, top_n=20,
            smooth=True, smooth_range=20, scale_coord=True,
            adatas=[source, target], verbose=False,
        )
        matched_source = source_rows[best]
        frame = pd.DataFrame({
            "control_slice": control_sid,
            "treated_slice": treated_sid,
            "source_row": matched_source,
            "target_row": target_rows,
            "source_cell_id": obs.index[matched_source].astype(str).to_numpy(),
            "target_cell_id": obs.index[target_rows].astype(str).to_numpy(),
            "similarity": similarities[:, 0],
            "source_celltype": obs.iloc[matched_source]["celltype"].astype(str).to_numpy(),
            "target_celltype": obs.iloc[target_rows]["celltype"].astype(str).to_numpy(),
            "source_z_cluster": assignments.iloc[matched_source]["z_cc_cluster"].to_numpy(),
            "target_z_cluster": assignments.iloc[target_rows]["z_cc_cluster"].to_numpy(),
            "source_s_cluster": assignments.iloc[matched_source]["s_cc_cluster"].to_numpy(),
            "target_s_cluster": assignments.iloc[target_rows]["s_cc_cluster"].to_numpy(),
            "slat_runtime_s": runtime,
        })
        frame.to_csv(cache, index=False, compression="gzip")
        all_records.append(frame)
        print(f"[pair] {control_sid}->{treated_sid}: n={len(frame)}, "
              f"score>=0.8={(frame.similarity >= 0.8).sum()}, runtime={runtime:.1f}s", flush=True)
        del source, target, edges, features, emb_source, emb_target, best, similarities
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.concat(all_records, ignore_index=True)


def normalized_transition(frame: pd.DataFrame, source_key: str, target_key: str,
                          source_order: list[str] | None = None,
                          target_order: list[str] | None = None) -> pd.DataFrame:
    table = pd.crosstab(frame[source_key], frame[target_key]).astype(float)
    if source_order is not None:
        table = table.reindex(source_order, fill_value=0)
    if target_order is not None:
        table = table.reindex(columns=target_order, fill_value=0)
    return table.div(table.sum(1).replace(0, np.nan), axis=0).fillna(0)


def transition_analysis(matches: pd.DataFrame, obs: pd.DataFrame, coords: np.ndarray,
                        threshold: float, fig_dir: Path, table_dir: Path) -> pd.DataFrame:
    sensitivity = []
    for cutoff in [0.70, 0.75, 0.80, 0.85, 0.90]:
        keep = matches.similarity >= cutoff
        tumor = keep & matches.source_celltype.isin(TUMOR_TYPES) & matches.target_celltype.isin(TUMOR_TYPES)
        sensitivity.append({"threshold": cutoff, "n_all": int(keep.sum()),
                            "n_tumor_to_tumor": int(tumor.sum()),
                            "fraction_all": float(keep.mean())})
    pd.DataFrame(sensitivity).to_csv(table_dir / "slat_threshold_sensitivity.csv", index=False)

    high = matches[matches.similarity >= threshold].copy()
    tumor = high[high.source_celltype.isin(TUMOR_TYPES) & high.target_celltype.isin(TUMOR_TYPES)].copy()
    trans = normalized_transition(tumor, "source_celltype", "target_celltype", TUMOR_TYPES, TUMOR_TYPES)
    trans.to_csv(table_dir / "slat_tumor_transition_matrix.csv")
    s_trans = normalized_transition(high, "source_s_cluster", "target_s_cluster")
    s_trans.to_csv(table_dir / "slat_salient_cluster_transition_matrix.csv")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(trans, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=axes[0])
    axes[0].set_title(f"SLAT tumor-state mapping (cosine >= {threshold})")
    axes[0].set_xlabel("treated target state")
    axes[0].set_ylabel("matched control state")
    sns.heatmap(s_trans, cmap="mako", ax=axes[1], cbar_kws={"label": "row fraction"})
    axes[1].set_title("Salient-domain transition after background-SLAT alignment")
    axes[1].set_xlabel("treated s-CC cluster")
    axes[1].set_ylabel("matched control s-CC cluster")
    fig.tight_layout()
    fig.savefig(fig_dir / "slat_transition_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    counts = (tumor.groupby(["source_celltype", "target_celltype"], observed=True).size()
              .rename("n").reset_index())
    try:
        import plotly.graph_objects as go
        source_names = list(counts.source_celltype.unique())
        target_names = list(counts.target_celltype.unique())
        src_map = {x: i for i, x in enumerate(source_names)}
        tgt_map = {x: i + len(source_names) for i, x in enumerate(target_names)}
        fig = go.Figure(go.Sankey(
            node={"label": [f"Control: {x}" for x in source_names] +
                           [f"Treated: {x}" for x in target_names], "pad": 15, "thickness": 18},
            link={"source": [src_map[x] for x in counts.source_celltype],
                  "target": [tgt_map[x] for x in counts.target_celltype],
                  "value": counts.n.tolist()},
        ))
        fig.update_layout(title=f"Paper-style SLAT tumor-state flow (score >= {threshold})", height=650)
        fig.write_html(fig_dir / "slat_tumor_transition_sankey.html", include_plotlyjs="cdn")
    except Exception as exc:
        warnings.warn(f"Could not write Sankey HTML: {exc}")

    # Aggregate target outcomes onto sampled source cells, reproducing the
    # manuscript's spatial view of initial states associated with later states.
    canonical_source = tumor[tumor.source_celltype == REFERENCE_FATE]
    agg = (canonical_source.assign(is_itgb6=(canonical_source.target_celltype == "Tumor_epithelial_Itgb6").astype(float))
           .groupby(["control_slice", "source_row"], observed=True)
           .agg(itgb6_score=("is_itgb6", "mean"), n_matches=("is_itgb6", "size"))
           .reset_index())
    agg.to_csv(table_dir / "control_initial_state_spatial_scores.csv", index=False)
    control_slices = sorted(agg.control_slice.unique())
    fig, axes = plt.subplots(1, len(control_slices), figsize=(6 * len(control_slices), 5), squeeze=False)
    for ax, sid in zip(axes.flat, control_slices):
        block = agg[agg.control_slice == sid]
        rows = block.source_row.to_numpy(dtype=int)
        sca = ax.scatter(coords[rows, 0], coords[rows, 1], c=block.itgb6_score,
                         cmap="viridis", vmin=0, vmax=1, s=5, linewidths=0, rasterized=True)
        ax.set_title(f"{sid}: initial canonical tumor\nassociation with treated Itgb6 state")
        ax.set_aspect("equal")
        ax.axis("off")
    fig.colorbar(sca, ax=axes.ravel().tolist(), fraction=0.02, label="Itgb6 target fraction")
    fig.savefig(fig_dir / "slat_control_spatial_initial_state.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return tumor


def load_expression(data: ad.AnnData, rows: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(rows.astype(int), return_inverse=True)
    matrix = data[unique].X
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32)
    # X in this object is already library-normalized/log1p.  Avoid applying the
    # manuscript notebook's normalization a second time.
    return matrix[inverse]


def source_state_de(data: ad.AnnData, tumor_matches: pd.DataFrame, threshold: float,
                    fig_dir: Path, table_dir: Path) -> dict:
    selected = tumor_matches[
        (tumor_matches.similarity >= threshold) &
        (tumor_matches.source_celltype == REFERENCE_FATE) &
        (tumor_matches.target_celltype.isin(TUMOR_TYPES))
    ].copy()
    if selected.empty:
        raise RuntimeError("No high-confidence canonical-source tumor mappings available for DE")
    expression = load_expression(data, selected.source_row.to_numpy())
    var_names = data.var_names.astype(str).to_numpy()
    de_summaries: dict[str, dict] = {}

    for fate in ["Tumor_epithelial_Itgb6", "Tumor_EMT"]:
        keep = selected.target_celltype.isin([REFERENCE_FATE, fate]).to_numpy()
        groups = selected.loc[keep, "target_celltype"].astype(str).to_numpy()
        if (groups == fate).sum() < 20 or (groups == REFERENCE_FATE).sum() < 20:
            continue
        de_adata = ad.AnnData(X=expression[keep], obs=pd.DataFrame({"target_fate": groups}),
                              var=pd.DataFrame(index=var_names))
        sc.tl.rank_genes_groups(de_adata, groupby="target_fate", groups=[fate],
                                reference=REFERENCE_FATE, method="wilcoxon",
                                use_raw=False, pts=True)
        result = sc.get.rank_genes_groups_df(de_adata, group=fate)
        result.to_csv(table_dir / f"cell_level_DE_initial_{fate}_vs_canonical.csv", index=False)

        # Conservative section-aware contrasts: average over target sections
        # within each of the three independent control sections, then signed-rank.
        section_diffs = []
        for control_sid in sorted(selected.control_slice.unique()):
            block = selected.control_slice.to_numpy() == control_sid
            hi = block & (selected.target_celltype.to_numpy() == fate)
            lo = block & (selected.target_celltype.to_numpy() == REFERENCE_FATE)
            if hi.sum() and lo.sum():
                section_diffs.append(expression[hi].mean(0) - expression[lo].mean(0))
        diffs = np.vstack(section_diffs)
        pvals = np.ones(len(var_names))
        for j in range(len(var_names)):
            if np.any(diffs[:, j] != 0):
                try:
                    pvals[j] = wilcoxon(diffs[:, j], alternative="two-sided").pvalue
                except ValueError:
                    pvals[j] = 1.0
        section_table = pd.DataFrame({
            "gene": var_names,
            "mean_log_expression_difference": diffs.mean(0),
            "median_log_expression_difference": np.median(diffs, axis=0),
            "positive_control_section_fraction": (diffs > 0).mean(0),
            "pvalue": pvals,
            "fdr": bh_fdr(pvals),
            "n_control_sections": diffs.shape[0],
        }).sort_values(["positive_control_section_fraction", "mean_log_expression_difference"],
                       ascending=[False, False])
        section_table.to_csv(table_dir / f"section_aware_initial_{fate}_vs_canonical.csv", index=False)

        x = result.logfoldchanges.to_numpy()
        y = -np.log10(result.pvals_adj.clip(lower=1e-300).to_numpy())
        sig_up = (result.pvals_adj < 0.05) & (result.logfoldchanges >= 0.75)
        sig_down = (result.pvals_adj < 0.05) & (result.logfoldchanges <= -0.75)
        fig, ax = plt.subplots(figsize=(9, 7))
        color = np.where(sig_up, "#D62728", np.where(sig_down, "#1F77B4", "#BDBDBD"))
        ax.scatter(x, y, c=color, s=7, alpha=0.65, linewidths=0, rasterized=True)
        top = result.assign(abs_fc=result.logfoldchanges.abs()).query("pvals_adj < 0.05").nlargest(12, "abs_fc")
        for _, row in top.iterrows():
            ax.annotate(row["names"], (row.logfoldchanges, -np.log10(max(row.pvals_adj, 1e-300))), fontsize=8)
        ax.axvline(0.75, color="black", ls="--", lw=0.8)
        ax.axvline(-0.75, color="black", ls="--", lw=0.8)
        ax.axhline(-np.log10(0.05), color="black", ls="--", lw=0.8)
        ax.set_xlabel("log2 fold change in matched control source cells")
        ax.set_ylabel("-log10 adjusted p-value (cell-level, descriptive)")
        ax.set_title(f"Initial canonical tumor cells associated with\n{fate} vs canonical treated outcome")
        fig.tight_layout()
        fig.savefig(fig_dir / f"volcano_initial_{fate}_vs_canonical.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        de_summaries[fate] = {
            "n_fate_matches": int((groups == fate).sum()),
            "n_reference_matches": int((groups == REFERENCE_FATE).sum()),
            "n_cell_level_up": int(sig_up.sum()),
            "n_cell_level_down": int(sig_down.sum()),
            "top_section_consistent_genes": section_table.head(20).gene.tolist(),
            "best_section_aware_fdr": float(section_table.fdr.min()),
        }

        # Match the notebook's Enrichr step but use the correct organism.
        try:
            import gseapy as gp
            for direction, gene_mask in [("up", sig_up), ("down", sig_down)]:
                genes = result.loc[gene_mask, "names"].astype(str).head(300).tolist()
                if len(genes) < 5:
                    continue
                enr = gp.enrichr(gene_list=genes, organism="Mouse",
                                 gene_sets=["GO_Biological_Process_2025"],
                                 background=var_names.tolist(), outdir=None, no_plot=True)
                enr.res2d.to_csv(table_dir / f"GO_{fate}_{direction}.csv", index=False)
                show = enr.res2d.query("`Adjusted P-value` < 0.1").head(15).copy()
                if len(show):
                    fig, ax = plt.subplots(figsize=(9, max(4, 0.38 * len(show))))
                    show = show.iloc[::-1]
                    ax.scatter(-np.log10(show["Adjusted P-value"].clip(lower=1e-300)),
                               range(len(show)), s=25 + 120 * show["Overlap"].str.split("/").str[0].astype(float) /
                               show["Overlap"].str.split("/").str[1].astype(float),
                               c=show["Combined Score"], cmap="viridis")
                    ax.set_yticks(range(len(show)), labels=show.Term)
                    ax.set_xlabel("-log10 adjusted p-value")
                    ax.set_title(f"GO enrichment: {fate}, {direction}")
                    fig.tight_layout()
                    fig.savefig(fig_dir / f"GO_{fate}_{direction}.png", dpi=220, bbox_inches="tight")
                    plt.close(fig)
        except Exception as exc:
            warnings.warn(f"Enrichr failed for {fate}: {exc}")
        del de_adata
        gc.collect()
    return de_summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cc-dir", type=Path, required=True)
    parser.add_argument("--assignment-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-slice", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--confidence", type=float, default=0.8)
    parser.add_argument("--clusters", type=int, default=12)
    parser.add_argument("--cluster-fit-per-slice", type=int, default=15000)
    args = parser.parse_args()
    fig_dir = args.out_dir / "figures"
    table_dir = args.out_dir / "tables"
    pair_dir = args.out_dir / "slat_pairs"
    for path in [fig_dir, table_dir, pair_dir]:
        path.mkdir(parents=True, exist_ok=True)

    data = ad.read_h5ad(args.data, backed="r")
    obs = data.obs.copy()
    names = np.load(args.cc_dir / "obs_names.npy", allow_pickle=True).astype(str)
    if not np.array_equal(names, obs.index.astype(str).to_numpy()):
        raise ValueError("CellCharter arrays do not align with AnnData rows")
    z_cc = np.load(args.cc_dir / "spatial.npy", mmap_mode="r")
    s_cc = np.load(args.cc_dir / "salient_spatial.npy", mmap_mode="r")
    coords = np.asarray(data.obsm["spatial"])
    assignments = fit_global_clusters(
        z_cc, s_cc, obs, args.clusters, args.cluster_fit_per_slice, args.seed
    )
    assignments.to_csv(table_dir / "global_cell_cluster_assignments.csv.gz",
                       compression="gzip")
    treated = obs["condition_label"].astype(str).to_numpy() == "KRASi_treated"

    plot_enrichment(assignments, obs, treated, fig_dir, table_dir)
    section_pca(z_cc, s_cc, assignments, obs, fig_dir, table_dir)
    matches = run_all_slat_pairs(obs, coords, z_cc, assignments, pair_dir,
                                 args.per_slice, args.seed, args.epochs, args.hidden_size)
    matches.to_csv(table_dir / "all_control_treated_slat_matches.csv.gz",
                   index=False, compression="gzip")
    tumor = transition_analysis(matches, obs, coords, args.confidence, fig_dir, table_dir)
    de_summary = source_state_de(data, tumor, args.confidence, fig_dir, table_dir)

    meta = {
        "analysis": "Haruka preprint lung + SLAT workflow adapted to KRAS Xenium",
        "paper_notebooks": ["lung_repo.ipynb", "slat_anaylsis_repo.ipynb"],
        "source_representation": "background z-CC (3 slice-bounded CellCharter hops, own block removed)",
        "target_response_representation": "salient s-CC clusters and annotated tumor states",
        "domain_clustering": (
            f"MiniBatchKMeans k={args.clusters}, fit on up to "
            f"{args.cluster_fit_per_slice} cells from every section; all cells assigned"
        ),
        "section_pairs": "all 3 Control x all 4 MRTX1133-treated = 12",
        "cells_sampled_per_section": args.per_slice,
        "slat": {"version": "0.2.1", "spatial_knn": 20, "LGCN_layers": 5,
                 "epochs": args.epochs, "hidden_size": args.hidden_size,
                 "top_n": 20, "spatial_smoothing": True,
                 "confidence_threshold": args.confidence},
        "n_total_matches": int(len(matches)),
        "n_high_confidence": int((matches.similarity >= args.confidence).sum()),
        "n_high_confidence_tumor_to_tumor": int(len(tumor)),
        "de_summary": de_summary,
        "inference": "cell-level Wilcoxon reproduces notebook; section-aware signed-rank is primary",
        "critical_limitation": (
            "Control and treated sections are not longitudinally paired; condition is perfectly "
            "confounded with Xenium run/cassette. SLAT mappings are cross-sectional associations, "
            "not observed lineage, fate, or causal transitions."
        ),
    }
    (args.out_dir / "reproduction_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
