#!/usr/bin/env python3
"""Derive section-robust macrophage s-CC markers for KRAS combination hypotheses."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


MAC_STATES = {
    "Monocyte", "moMacrophage", "Macrophage_interstitial",
    "Macrophage_alveolar", "Macrophage_alveolar_prolif",
}


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad"))
    parser.add_argument("--assignments", type=Path, default=Path("/home/yancui/haruka_v2/bench_kras_xenium/analysis/fig6_macrophage_paga/tables/macrophage_paga_assignments.csv.gz"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/yancui/haruka_v2/bench_kras_xenium/analysis/fig6_macrophage_paga/combination_targets"))
    parser.add_argument("--chunk-cells", type=int, default=25000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    assignment = pd.read_csv(args.assignments)
    with h5py.File(args.data, "r") as handle:
        celltype = categorical(handle, "celltype")
        condition = categorical(handle, "condition_label")
        slice_id = categorical(handle, "slice_id")
        genes = decode(handle["var"]["_index"][:])
        indptr = handle["X"]["indptr"][:]
        n_cells, n_genes = map(int, handle["X"].attrs["shape"])

    macro_rows = np.flatnonzero(np.isin(celltype, list(MAC_STATES)))
    treated_macro = assignment["condition_label"].astype(str).to_numpy() == "KRASi_treated"
    if "source_row" in assignment:
        selected_rows = assignment.loc[treated_macro, "source_row"].astype(int).to_numpy()
        if not np.all(np.isin(selected_rows, macro_rows)):
            raise ValueError("Explicit macrophage source rows do not align to the source data")
    else:
        if len(macro_rows) != len(assignment):
            raise ValueError("Macrophage assignment rows do not align")
        selected_rows = macro_rows[treated_macro]
    cluster_key = "s_cc_cluster" if "s_cc_cluster" in assignment else "s_cc_paga_cluster"
    selected_cluster = assignment.loc[treated_macro, cluster_key].astype(int).to_numpy()
    selected_slice = assignment.loc[treated_macro, "slice_id"].astype(str).to_numpy()
    clusters = np.asarray(sorted(np.unique(selected_cluster)), dtype=int)
    slices = np.asarray(sorted(np.unique(selected_slice)), dtype=object)
    cluster_to_pos = {value: i for i, value in enumerate(clusters)}
    slice_to_pos = {value: i for i, value in enumerate(slices)}

    full_cluster = np.full(n_cells, -1, dtype=np.int16)
    full_slice = np.full(n_cells, -1, dtype=np.int8)
    for row, cluster, section in zip(selected_rows, selected_cluster, selected_slice):
        full_cluster[row] = cluster_to_pos[cluster]
        full_slice[row] = slice_to_pos[section]

    sums = np.zeros((len(clusters), len(slices), n_genes), dtype=np.float64)
    detects = np.zeros_like(sums)
    counts = np.zeros((len(clusters), len(slices)), dtype=np.int64)
    with h5py.File(args.data, "r") as handle:
        values_ds = handle["X"]["data"]
        indices_ds = handle["X"]["indices"]
        for left in range(0, n_cells, args.chunk_cells):
            right = min(left + args.chunk_cells, n_cells)
            chosen = np.flatnonzero(full_cluster[left:right] >= 0)
            if not len(chosen):
                continue
            start, stop = int(indptr[left]), int(indptr[right])
            local_ptr = indptr[left:right + 1].astype(np.int64) - start
            block = sparse.csr_matrix(
                (values_ds[start:stop], indices_ds[start:stop], local_ptr),
                shape=(right - left, n_genes),
            )[chosen]
            block_cluster = full_cluster[left:right][chosen]
            block_slice = full_slice[left:right][chosen]
            for cluster_pos in np.unique(block_cluster):
                for slice_pos in np.unique(block_slice[block_cluster == cluster_pos]):
                    take = (block_cluster == cluster_pos) & (block_slice == slice_pos)
                    part = block[take]
                    sums[cluster_pos, slice_pos] += np.asarray(part.sum(0)).ravel()
                    detects[cluster_pos, slice_pos] += np.asarray((part != 0).sum(0)).ravel()
                    counts[cluster_pos, slice_pos] += int(take.sum())

    total_sum = sums.sum((0, 1))
    total_detect = detects.sum((0, 1))
    total_n = int(counts.sum())
    marker_tables = []
    marker_full_tables = []
    robust_tables = []
    robust_full_tables = []
    for cluster_pos, cluster in enumerate(clusters):
        cluster_sum = sums[cluster_pos].sum(0)
        cluster_detect = detects[cluster_pos].sum(0)
        cluster_n = int(counts[cluster_pos].sum())
        other_n = total_n - cluster_n
        delta = cluster_sum / max(cluster_n, 1) - (total_sum - cluster_sum) / max(other_n, 1)
        pct = cluster_detect / max(cluster_n, 1)
        pct_other = (total_detect - cluster_detect) / max(other_n, 1)
        frame = pd.DataFrame({
            "cluster": int(cluster), "gene": genes, "n_cluster": cluster_n,
            "mean_logexpr_delta": delta, "pct_cluster": pct,
            "pct_other": pct_other, "pct_delta": pct - pct_other,
        })
        frame["marker_score"] = frame["mean_logexpr_delta"] * np.maximum(frame["pct_delta"], 0)
        marker_full_tables.append(frame)
        marker_tables.append(frame.sort_values(["marker_score", "mean_logexpr_delta"], ascending=False).head(250))

        slice_delta = []
        for slice_pos, section in enumerate(slices):
            cn = int(counts[cluster_pos, slice_pos])
            sn = int(counts[:, slice_pos].sum())
            on = sn - cn
            if cn >= 20 and on >= 20:
                d = sums[cluster_pos, slice_pos] / cn - (sums[:, slice_pos].sum(0) - sums[cluster_pos, slice_pos]) / on
                slice_delta.append(d)
        if slice_delta:
            arr = np.vstack(slice_delta)
            robust = pd.DataFrame({
                "cluster": int(cluster), "gene": genes, "n_sections": arr.shape[0],
                "median_section_delta": np.median(arr, axis=0),
                "mean_section_delta": np.mean(arr, axis=0),
                "min_section_delta": np.min(arr, axis=0),
                "positive_sections": (arr > 0).sum(0),
            })
            robust_full_tables.append(robust)
            robust_tables.append(robust.sort_values(
                ["positive_sections", "median_section_delta"], ascending=False
            ).head(500))

    markers = pd.concat(marker_tables, ignore_index=True)
    markers_full = pd.concat(marker_full_tables, ignore_index=True)
    robust = pd.concat(robust_tables, ignore_index=True)
    robust_full = pd.concat(robust_full_tables, ignore_index=True)
    markers.to_csv(args.out_dir / "scc_macrophage_cluster_markers.csv", index=False)
    robust.to_csv(args.out_dir / "scc_macrophage_section_robust_markers.csv", index=False)
    pd.DataFrame(counts, index=clusters, columns=slices).rename_axis("cluster").to_csv(
        args.out_dir / "cluster_cell_counts_by_section.csv"
    )

    candidates = [
        "Cd47", "Sirpa", "Tgfb1", "Tgfbr1", "Tgfbr2", "Trem2", "Csf1r",
        "Ccr2", "Arg1", "Mrc1", "Lgals9", "Cxcl2", "Ccl2", "Ccl7", "Tnf",
        "Axl", "Mertk", "Vegfa", "Hif1a", "Stat3", "Jak1", "Jak2", "Socs3",
        "Nfkb1", "Nfkb2", "Cd274", "Il10", "Nlrp3", "Marco", "Fcgr1",
        "Adgre1", "Itgam", "Csf1", "Csf2", "Syk", "Pik3cg", "Pik3cd",
    ]
    candidate_table = robust_full[robust_full.gene.isin(candidates)].merge(
        markers_full[["cluster", "gene", "mean_logexpr_delta", "pct_cluster", "pct_other", "marker_score"]],
        on=["cluster", "gene"], how="left",
    )
    candidate_table.to_csv(args.out_dir / "actionable_candidate_expression_by_scc_cluster.csv", index=False)
    print("treated macrophages", total_n)
    print("cluster counts by section\n", pd.DataFrame(counts, index=clusters, columns=slices))
    for cluster in [2, 3]:
        print(f"\ncluster {cluster} robust markers")
        print(robust[robust.cluster == cluster].head(30).to_string(index=False))
        print(f"\ncluster {cluster} candidates")
        print(candidate_table[candidate_table.cluster == cluster].sort_values("median_section_delta", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
