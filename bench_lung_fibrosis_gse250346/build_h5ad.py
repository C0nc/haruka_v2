#!/usr/bin/env python3
"""Build a compact AnnData object from the corrected GSE250346 Seurat export."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--contrast",
        choices=("less-vs-more", "unaffected-vs-fibrosis"),
        default="less-vs-more",
    )
    args = parser.parse_args()

    root = args.export_dir
    n_genes, n_cells, n_nonzero = np.loadtxt(
        root / "counts_dims.tsv", dtype=np.int64
    ).reshape(-1)
    indices = np.fromfile(root / "counts_i.int32.bin", dtype=np.int32)
    indptr = np.fromfile(root / "counts_p.int32.bin", dtype=np.int32)
    values = np.fromfile(root / "counts_x.float32.bin", dtype=np.float32)
    if len(indices) != n_nonzero or len(values) != n_nonzero:
        raise ValueError("Sparse count payload does not match counts_dims.tsv")
    if len(indptr) != n_cells + 1:
        raise ValueError("Sparse column pointer does not match the exported cell count")

    gene_by_cell = sp.csc_matrix(
        (values, indices, indptr), shape=(n_genes, n_cells), dtype=np.float32
    )
    counts = gene_by_cell.T.tocsr()
    obs = pd.read_csv(root / "obs.csv.gz", low_memory=False)
    with gzip.open(root / "var_names.txt.gz", "rt") as handle:
        var_names = [line.rstrip("\n") for line in handle]
    if len(obs) != n_cells or len(var_names) != n_genes:
        raise ValueError("Observation or feature metadata is inconsistent with counts")

    obs.index = pd.Index(obs.pop("obs_name").astype(str), name=None)
    if not obs.index.is_unique:
        raise ValueError("GSE250346 cell identifiers are not unique")
    categorical = (
        "condition", "severity_group", "sample_affect", "sample_type",
        "slice_id", "sample", "donor", "patient", "batch", "run", "tma",
        "cell_type", "final_CT", "lineage", "final_lineage", "CNiche",
        "TNiche",
    )
    for column in categorical:
        if column not in obs:
            continue
        obs[column] = obs[column].fillna("Unknown").astype("category")

    var = pd.DataFrame(index=pd.Index(var_names, name=None))
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.layers["counts"] = counts
    adata.obsm["spatial"] = obs[["x_centroid", "y_centroid"]].to_numpy(
        dtype=np.float32
    )
    if args.contrast == "unaffected-vs-fibrosis":
        adata.uns["contrast_definition"] = {
            "reference": "Unaffected lung cores",
            "condition": (
                "All pulmonary-fibrosis cores: Less Affected and More Affected"
            ),
            "severity_role": (
                "Less/More Affected withheld from Haruka training and used "
                "only for within-condition evaluation"
            ),
        }
    else:
        adata.uns["contrast_definition"] = {
            "reference": "Less Affected pulmonary-fibrosis cores (<75% pathology)",
            "condition": "More Affected pulmonary-fibrosis cores (>=75% pathology)",
            "unaffected_role": "excluded from model fit; reserved as external reference",
        }
    adata.uns["batch_design"] = {
        "decoder_batch_key": "batch (GEO run)",
        "spatial_graph_boundary": "slice_id (tissue core)",
        "donor_key": "donor (patient)",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.out, compression="gzip")
    print(adata)
    print(obs.groupby(["condition", "batch"], observed=True).size())


if __name__ == "__main__":
    main()
