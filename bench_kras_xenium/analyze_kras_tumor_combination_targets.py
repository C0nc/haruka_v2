#!/usr/bin/env python3
"""Compare actionable tumor-side genes in treated Itgb6+ versus other tumor cells."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


DATA = Path("/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad")
OUT = Path("/home/yancui/haruka_v2/bench_kras_xenium/analysis/fig6_macrophage_paga/combination_targets")
GENES = [
    "Cd47", "Cd274", "Csf1", "Csf2", "Ccl2", "Ccl7", "Cxcl2", "Cxcl12",
    "Tgfb1", "Tgfb2", "Tgfb3", "Gas6", "Il10", "Vegfa", "Mif", "Spp1",
    "Itgb6", "Krt8", "Krt18", "Krt19", "Epcam", "Msl3", "Lgals9",
]


def decode(values):
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in values], dtype=object)


def categorical(handle, name):
    group = handle["obs"][name]
    categories = decode(group["categories"][:])
    codes = group["codes"][:]
    out = np.full(len(codes), "NA", dtype=object)
    valid = codes >= 0
    out[valid] = categories[codes[valid]]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(DATA, "r") as handle:
        celltype = categorical(handle, "celltype")
        condition = categorical(handle, "condition_label")
        section = categorical(handle, "slice_id")
        genes = decode(handle["var"]["_index"][:])
        indptr = handle["X"]["indptr"][:]
        n_cells, n_genes = map(int, handle["X"].attrs["shape"])

    gene_lookup = {gene.lower(): i for i, gene in enumerate(genes)}
    kept = [(gene, gene_lookup[gene.lower()]) for gene in GENES if gene.lower() in gene_lookup]
    query_names = np.asarray([x[0] for x in kept], dtype=object)
    query_idx = np.asarray([x[1] for x in kept], dtype=int)
    tumor_types = np.asarray(sorted(set(celltype[np.char.startswith(celltype.astype(str), "Tumor_")])), dtype=object)
    treated_sections = np.asarray(sorted(set(section[condition == "KRASi_treated"])), dtype=object)
    type_pos = {value: i for i, value in enumerate(tumor_types)}
    section_pos = {value: i for i, value in enumerate(treated_sections)}
    full_type = np.asarray([type_pos.get(value, -1) for value in celltype], dtype=np.int16)
    full_section = np.asarray([section_pos.get(value, -1) for value in section], dtype=np.int8)
    selected = (condition == "KRASi_treated") & (full_type >= 0) & (full_section >= 0)

    shape = (len(tumor_types), len(treated_sections), len(query_idx))
    sums = np.zeros(shape, dtype=np.float64)
    detects = np.zeros(shape, dtype=np.int64)
    counts = np.zeros(shape[:2], dtype=np.int64)
    chunk = 25000
    with h5py.File(DATA, "r") as handle:
        values_ds = handle["X"]["data"]
        indices_ds = handle["X"]["indices"]
        for left in range(0, n_cells, chunk):
            right = min(left + chunk, n_cells)
            chosen = np.flatnonzero(selected[left:right])
            if not len(chosen):
                continue
            start, stop = int(indptr[left]), int(indptr[right])
            local_ptr = indptr[left:right + 1].astype(np.int64) - start
            block = sparse.csr_matrix(
                (values_ds[start:stop], indices_ds[start:stop], local_ptr),
                shape=(right - left, n_genes),
            )[chosen][:, query_idx]
            block_type = full_type[left:right][chosen]
            block_section = full_section[left:right][chosen]
            for tp in np.unique(block_type):
                for sp in np.unique(block_section[block_type == tp]):
                    take = (block_type == tp) & (block_section == sp)
                    part = block[take]
                    sums[tp, sp] += np.asarray(part.sum(0)).ravel()
                    detects[tp, sp] += np.asarray((part != 0).sum(0)).ravel()
                    counts[tp, sp] += int(take.sum())

    rows = []
    for tp, type_name in enumerate(tumor_types):
        for sp, section_name in enumerate(treated_sections):
            n = int(counts[tp, sp])
            for gi, gene in enumerate(query_names):
                rows.append({
                    "celltype": type_name,
                    "slice_id": section_name,
                    "gene": gene,
                    "n_cells": n,
                    "mean_logexpr": sums[tp, sp, gi] / max(n, 1),
                    "pct_detected": detects[tp, sp, gi] / max(n, 1),
                })
    long = pd.DataFrame(rows)
    long.to_csv(OUT / "treated_tumor_candidate_expression_by_celltype_section.csv", index=False)

    # Section-stratified Itgb6+ tumor contrast against all other tumor states.
    itgb6_name = "Tumor_epithelial_Itgb6"
    itgb6_pos = type_pos[itgb6_name]
    contrasts = []
    for sp, section_name in enumerate(treated_sections):
        pos_n = int(counts[itgb6_pos, sp])
        neg_n = int(counts[:, sp].sum() - pos_n)
        if pos_n < 20 or neg_n < 20:
            continue
        pos_mean = sums[itgb6_pos, sp] / pos_n
        neg_mean = (sums[:, sp].sum(0) - sums[itgb6_pos, sp]) / neg_n
        pos_pct = detects[itgb6_pos, sp] / pos_n
        neg_pct = (detects[:, sp].sum(0) - detects[itgb6_pos, sp]) / neg_n
        for gi, gene in enumerate(query_names):
            contrasts.append({
                "slice_id": section_name, "gene": gene,
                "n_itgb6": pos_n, "n_other_tumor": neg_n,
                "itgb6_mean": pos_mean[gi], "other_mean": neg_mean[gi],
                "mean_delta": pos_mean[gi] - neg_mean[gi],
                "itgb6_pct": pos_pct[gi], "other_pct": neg_pct[gi],
                "pct_delta": pos_pct[gi] - neg_pct[gi],
            })
    contrast = pd.DataFrame(contrasts)
    contrast.to_csv(OUT / "treated_itgb6_vs_other_tumor_candidate_expression_by_section.csv", index=False)
    summary = contrast.groupby("gene", as_index=False).agg(
        n_sections=("slice_id", "nunique"),
        median_mean_delta=("mean_delta", "median"),
        mean_mean_delta=("mean_delta", "mean"),
        positive_sections=("mean_delta", lambda x: int((x > 0).sum())),
        median_itgb6_mean=("itgb6_mean", "median"),
        median_itgb6_pct=("itgb6_pct", "median"),
        median_other_pct=("other_pct", "median"),
    ).sort_values(["positive_sections", "median_mean_delta"], ascending=False)
    summary.to_csv(OUT / "treated_itgb6_vs_other_tumor_candidate_expression_summary.csv", index=False)
    print("Tumor types:", tumor_types.tolist())
    print("Counts:\n", pd.DataFrame(counts, index=tumor_types, columns=treated_sections))
    print("\nItgb6-positive tumor contrast:\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
