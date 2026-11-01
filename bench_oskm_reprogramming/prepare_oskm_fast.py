#!/usr/bin/env python
"""Sequential-HDF5 entry point for :mod:`prepare_oskm` on the 3.2 GB atlas."""

import numpy as np
from scipy import sparse
import anndata as ad

import prepare_oskm


def read_selected_rows(matrix, selected, chunk_rows=10_000):
    selected = np.sort(np.asarray(selected, dtype=np.int64))
    output = np.empty((len(selected), matrix.shape[1]), dtype=np.float32)
    for left in range(0, matrix.shape[0], chunk_rows):
        right = min(left + chunk_rows, matrix.shape[0])
        out_left = np.searchsorted(selected, left, side="left")
        out_right = np.searchsorted(selected, right, side="left")
        if out_left == out_right:
            continue
        block = np.asarray(matrix[left:right, :], dtype=np.float32)
        output[out_left:out_right] = block[selected[out_left:out_right] - left]
    return output


def fast_materialize(source, selected):
    selected = np.sort(np.asarray(selected, dtype=np.int64))
    counts = sparse.csr_matrix(read_selected_rows(source.X, selected))
    data = ad.AnnData(
        X=counts.copy(),
        obs=source.obs.iloc[selected].copy(),
        var=source.var.copy(),
    )
    data.layers["counts"] = counts.copy()
    data.obsm["spatial"] = np.asarray(source.obsm["spatial"])[selected].astype(
        np.float32
    )
    return data


if __name__ == "__main__":
    prepare_oskm.materialize = fast_materialize
    prepare_oskm.main()
