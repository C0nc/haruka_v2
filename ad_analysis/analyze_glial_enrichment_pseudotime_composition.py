#!/usr/bin/env python3
"""Functional enrichment and subtype shifts along shared glial s-CC DPT."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import anndata as ad
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, hypergeom


ROOT = Path("/home/yancui/haruka_v2")
RESULTS = ROOT / "ad_analysis/scc_dpt_cellrank2"
GLIA = RESULTS / "glia_terminal_analysis"
OUT = GLIA / "enrichment_and_composition"

CONFIG = {
    "starmap": {
        "h5ad": ROOT / "bench_real_discovery/data/ad.h5ad",
        "embedding": ROOT / (
            "bench_real_discovery/results_shared_full_niche_decoderonly_lambda25/ad/"
            "haruka_shared_full_niche_decoderbatch8_batchmode-decoder_only_seed0"
        ),
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
    },
}

STAR_LINEAGES = ["Micro", "Astro", "Oligo"]
Q_LABELS = ["Q1 early", "Q2", "Q3", "Q4", "Q5 late"]


def shared_dpt(dataset: str, scores: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    cfg = CONFIG[dataset]
    names = np.load(cfg["embedding"] / "obs_names.npy", allow_pickle=True).astype(str)
    embedding = np.load(cfg["embedding"] / "salient_spatial.npy", mmap_mode="r")
    emb_index = pd.Series(np.arange(len(names)), index=names)
    a_source = ad.read_h5ad(cfg["h5ad"], backed="r")
    obs = a_source.obs.copy()
    obs.index = obs.index.astype(str)
    frames = []
    for replicate, group in scores.groupby("replicate", sort=True):
        ids = group.cell_id.astype(str).to_numpy()
        valid = np.asarray([x in emb_index.index and x in obs.index for x in ids])
        group = group.iloc[np.flatnonzero(valid)].copy().reset_index(drop=True)
        ids = group.cell_id.astype(str).to_numpy()
        x = np.asarray(embedding[emb_index.loc[ids].to_numpy()], dtype=np.float32)
        a = ad.AnnData(X=np.zeros((len(ids), 1), dtype=np.float32), obs=pd.DataFrame(index=ids))
        a.obsm["X_scc"] = x
        n_components = 2
        used_k = None
        for k in [40, 60, 100]:
            k = min(k, len(ids) - 1)
            sc.pp.neighbors(a, n_neighbors=k, use_rep="X_scc", metric="euclidean", random_state=seed)
            n_components = connected_components(a.obsp["connectivities"], directed=False, return_labels=False)
            used_k = k
            if n_components == 1:
                break
        distance = group.plaque_distance.to_numpy(dtype=float)
        far = np.flatnonzero(distance >= np.nanquantile(distance, 0.90))
        center = x[far].mean(axis=0)
        root = int(far[np.argmin(np.square(x[far] - center).sum(axis=1))])
        a.uns["iroot"] = root
        sc.tl.diffmap(a, n_comps=15)
        sc.tl.dpt(a, n_dcs=10)
        pt = a.obs.dpt_pseudotime.to_numpy(dtype=float)
        if not np.isfinite(pt).all():
            finite = np.isfinite(pt)
            pt[~finite] = np.nanmax(pt[finite]) if finite.any() else 1.0
        group["shared_scc_dpt"] = pt
        group["shared_dpt_root_cell"] = ids[root]
        group["shared_dpt_neighbors"] = used_k
        group["shared_dpt_graph_components"] = n_components
        if dataset == "starmap":
            # Read the original fine-grained annotation directly. Although
            # `cell_subtype` is currently a copy, this guarantees that ct_top
            # cannot accidentally be used for composition analysis.
            group["ct_sub"] = obs.loc[ids, "ct_sub"].astype(str).to_numpy()
        else:
            group["codex_state"] = obs.loc[ids, "paper_initial_leiden"].astype(str).to_numpy()
        frames.append(group)
        print(f"{dataset} {replicate}: n={len(group)}, k={used_k}, components={n_components}", flush=True)
    a_source.file.close()
    return pd.concat(frames, ignore_index=True)


def add_quantile_bin(frame: pd.DataFrame, group_keys: list[str], dpt_key: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["pseudotime_bin"] = frame.groupby(group_keys, sort=False)[dpt_key].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=Q_LABELS)
    )
    return frame


def composition_table(frame: pd.DataFrame, panel_keys: list[str], label_key: str) -> pd.DataFrame:
    count = (
        frame.groupby(panel_keys + ["pseudotime_bin", label_key], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    count["proportion"] = count.n_cells / count.groupby(panel_keys + ["pseudotime_bin"], observed=True).n_cells.transform("sum")
    return count


def composition_stats(table: pd.DataFrame, panel_keys: list[str], label_key: str) -> pd.DataFrame:
    rows = []
    for keys, group in table.groupby(panel_keys, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        contingency = group.pivot(index="pseudotime_bin", columns=label_key, values="n_cells").fillna(0)
        chi2, p, _, _ = chi2_contingency(contingency)
        n = contingency.to_numpy().sum()
        denom = max(min(contingency.shape) - 1, 1)
        cramers_v = np.sqrt(chi2 / (n * denom))
        prop = group.pivot(index="pseudotime_bin", columns=label_key, values="proportion").fillna(0)
        prop = prop.reindex(Q_LABELS, fill_value=0)
        js = float(jensenshannon(prop.iloc[0], prop.iloc[-1], base=2) ** 2)
        record = dict(zip(panel_keys, keys))
        record.update(
            {
                "label_key": label_key,
                "n_cells": int(n),
                "n_categories": contingency.shape[1],
                "chi2_p_value": float(p),
                "cramers_v": float(cramers_v),
                "early_late_js_divergence": js,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def stacked_panels(
    table: pd.DataFrame,
    row_key: str | None,
    col_key: str,
    label_key: str,
    title: str,
    path: Path,
):
    rows = [None] if row_key is None else list(dict.fromkeys(table[row_key].astype(str)))
    cols = sorted(table[col_key].astype(str).unique())
    categories = sorted(table[label_key].astype(str).unique())
    palette = dict(zip(categories, sns.color_palette("husl", len(categories))))
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(7.2 * len(cols), 5.2 * len(rows)), squeeze=False)
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            sub = table[table[col_key].astype(str) == col]
            if row is not None:
                sub = sub[sub[row_key].astype(str) == row]
            matrix = sub.pivot(index="pseudotime_bin", columns=label_key, values="proportion").fillna(0)
            matrix = matrix.reindex(Q_LABELS, fill_value=0).reindex(columns=categories, fill_value=0)
            bottom = np.zeros(len(matrix))
            x = np.arange(len(matrix))
            for category in categories:
                values = matrix[category].to_numpy()
                axes[i, j].bar(x, values, bottom=bottom, color=palette[category], width=0.82, label=category)
                bottom += values
            panel = col if row is None else f"{row} · {col}"
            axes[i, j].set_title(panel)
            axes[i, j].set_xticks(x, Q_LABELS, rotation=30, ha="right")
            axes[i, j].set_ylim(0, 1)
            axes[i, j].set_ylabel("Cell proportion" if j == 0 else "")
            axes[i, j].set_xlabel("Shared salient-CC pseudotime" if i == len(rows) - 1 else "")
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[x]) for x in categories]
    fig.legend(handles, categories, title=label_key, loc="center left", bbox_to_anchor=(1.002, 0.5), frameon=False)
    fig.suptitle(title, y=1.005)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def run_enrichment(out: Path) -> pd.DataFrame:
    meta = pd.read_csv(GLIA / "starmap_glial_initial_molecular_terminal_associations.csv")
    results = []
    libraries = ["GO_Biological_Process_2025", "Reactome_Pathways_2024"]
    background = set(meta.feature.astype(str).str.upper().unique())
    meta["feature_upper"] = meta.feature.astype(str).str.upper()
    # Fetch only public definitions; enrichment of private lists is local.
    gene_sets = {library: gp.get_library(name=library, organism="Mouse") for library in libraries}
    for lineage in STAR_LINEAGES:
        frame = meta[(meta.lineage == lineage) & meta.same_direction_both_replicates.astype(bool)].copy()
        for fate, ascending in [("Plaque-proximal terminal", False), ("Alternative terminal", True)]:
            genes = set(frame.sort_values("meta_spearman_rho", ascending=ascending).feature_upper.head(150)) & background
            local_rows = []
            for library, terms in gene_sets.items():
                for term, members_raw in terms.items():
                    members = set(members_raw) & background
                    overlap = genes & members
                    if len(overlap) < 2 or not members:
                        continue
                    p_value = float(hypergeom.sf(len(overlap) - 1, len(background), len(members), len(genes)))
                    enrichment_ratio = (len(overlap) / len(genes)) / (len(members) / len(background))
                    local_rows.append({
                        "lineage": lineage, "fate": fate, "Gene_set": library, "Term": term,
                        "Overlap": f"{len(overlap)}/{len(genes)}", "P-value": p_value,
                        "Odds Ratio": enrichment_ratio,
                        "Combined Score": -np.log10(max(p_value, 1e-300)) * enrichment_ratio,
                        "Genes": ";".join(sorted(overlap)), "query_genes_json": json.dumps(sorted(genes)),
                    })
            enr = pd.DataFrame(local_rows)
            if not enr.empty:
                order = np.argsort(enr["P-value"].to_numpy())
                ranked = enr["P-value"].to_numpy()[order]
                q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
                q = np.minimum.accumulate(q[::-1])[::-1]
                restored = np.empty_like(q)
                restored[order] = np.minimum(q, 1.0)
                enr["Adjusted P-value"] = restored
                results.append(enr)
    all_results = pd.concat(results, ignore_index=True)
    all_results.to_csv(out / "starmap_glial_terminal_functional_enrichment.csv", index=False)

    plot_rows = []
    for (lineage, fate), group in all_results.groupby(["lineage", "fate"], sort=False):
        sig = group[group["Adjusted P-value"] < 0.10].copy()
        if sig.empty:
            sig = group.nsmallest(5, "Adjusted P-value").copy()
        else:
            sig = sig.sort_values(["Adjusted P-value", "Combined Score"], ascending=[True, False]).head(6)
        sig["minus_log10_q"] = -np.log10(sig["Adjusted P-value"].clip(lower=1e-300))
        plot_rows.append(sig)
    plot = pd.concat(plot_rows, ignore_index=True)
    plot["panel"] = plot.lineage + " | " + plot.fate.str.replace(" terminal", "", regex=False)
    panels = list(dict.fromkeys(plot.panel))
    fig, axes = plt.subplots(2, 3, figsize=(22, 12), squeeze=False)
    for ax, panel in zip(axes.ravel(), panels):
        sub = plot[plot.panel == panel].sort_values("minus_log10_q")
        labels = sub.Term.str.replace(r" \(GO:\d+\)$", "", regex=True).str.slice(0, 62)
        ax.scatter(sub.minus_log10_q, np.arange(len(sub)), s=np.clip(sub["Combined Score"], 20, 250), c=sub.minus_log10_q, cmap="viridis")
        ax.set_yticks(np.arange(len(sub)), labels)
        ax.set_xlabel("−log10 adjusted P")
        ax.set_title(panel)
    fig.suptitle("STARmap glial early-signature functional enrichment")
    fig.tight_layout()
    fig.savefig(out / "starmap_glial_terminal_functional_enrichment.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return all_results


def main():
    global RESULTS, GLIA, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=RESULTS)
    parser.add_argument("--starmap-embedding", type=Path)
    parser.add_argument("--codex-embedding", type=Path)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    RESULTS = args.root
    GLIA = RESULTS / "glia_terminal_analysis"
    if args.starmap_embedding is not None:
        CONFIG["starmap"]["embedding"] = args.starmap_embedding
    if args.codex_embedding is not None:
        CONFIG["codex"]["embedding"] = args.codex_embedding
    if args.out == OUT:
        args.out = GLIA / "enrichment_and_composition"
    OUT = args.out
    args.out.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(RESULTS / "cellrank_cell_scores.csv.gz", dtype={"cell_id": str, "main_cluster": str})

    star_scores = scores[(scores.dataset == "starmap") & scores.main_cluster.isin(STAR_LINEAGES)].copy()
    codex_scores = scores[scores.dataset == "codex"].copy()
    star_shared = shared_dpt("starmap", star_scores)
    codex_shared = shared_dpt("codex", codex_scores)
    shared = pd.concat([star_shared, codex_shared], ignore_index=True)
    shared.to_csv(args.out / "shared_salientcc_dpt_cells.csv.gz", index=False)

    # Shared glial DPT: proportions use the fine-grained ct_sub labels.
    star_shared = add_quantile_bin(star_shared, ["replicate"], "shared_scc_dpt")
    star_sub = composition_table(star_shared, ["replicate"], "ct_sub")
    stacked_panels(
        star_sub, None, "replicate", "ct_sub",
        "STARmap AD: ct_sub proportions along shared salient-CC DPT",
        args.out / "starmap_shared_scc_dpt_cell_subtype_stacked.png",
    )

    # Existing within-lineage DPT remains valid for subtype changes inside each lineage.
    star_within = add_quantile_bin(star_scores.merge(
        star_shared[["cell_id", "ct_sub"]], on="cell_id", how="left"
    ), ["replicate", "main_cluster"], "dpt_pseudotime")
    within_sub = composition_table(star_within, ["main_cluster", "replicate"], "ct_sub")
    stacked_panels(
        within_sub, "main_cluster", "replicate", "ct_sub",
        "STARmap AD: ct_sub proportions along within-lineage salient-CC DPT",
        args.out / "starmap_within_lineage_dpt_cell_subtype_stacked.png",
    )

    codex_shared = add_quantile_bin(codex_shared, ["replicate"], "shared_scc_dpt")
    codex_state = composition_table(codex_shared, ["replicate"], "codex_state")
    stacked_panels(
        codex_state, None, "replicate", "codex_state",
        "CODEX AD: published myeloid-state proportions along shared salient-CC DPT",
        args.out / "codex_shared_scc_dpt_myeloid_state_stacked.png",
    )

    tables = {
        "starmap_shared_ct_sub": (star_sub, ["replicate"], "ct_sub"),
        "starmap_within_lineage_ct_sub": (within_sub, ["main_cluster", "replicate"], "ct_sub"),
        "codex_shared_state": (codex_state, ["replicate"], "codex_state"),
    }
    stats = []
    for name, (table, panel_keys, label_key) in tables.items():
        table.to_csv(args.out / f"{name}_composition.csv", index=False)
        temp = composition_stats(table, panel_keys, label_key)
        temp.insert(0, "analysis", name)
        stats.append(temp)
    pd.concat(stats, ignore_index=True).to_csv(args.out / "pseudotime_composition_shift_statistics.csv", index=False)

    run_enrichment(args.out)
    print(f"Saved enrichment and composition analysis to {args.out}", flush=True)


if __name__ == "__main__":
    main()
