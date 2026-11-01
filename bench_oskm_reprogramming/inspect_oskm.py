#!/usr/bin/env python
"""Inspect the Zenodo partial-reprogramming MERFISH AnnData metadata."""

from pathlib import Path

import anndata as ad
import numpy as np


DATA = Path(__file__).parent / "data" / "reprogramming.h5ad"


def main() -> None:
    adata = ad.read_h5ad(DATA, backed="r")
    obs = adata.obs
    print("shape", adata.shape, flush=True)
    for column in (
        "cohort",
        "age",
        "mouse_id",
        "slide_id",
        "celltype",
        "region",
        "subregion",
    ):
        print(f"\n{column}", flush=True)
        print(obs[column].value_counts(dropna=False).to_string(), flush=True)

    print("\nmouse mapping", flush=True)
    print(
        obs.groupby(
            ["cohort", "mouse_id", "slide_id", "age"], observed=True
        ).size().to_string(),
        flush=True,
    )
    print("\nslide mapping", flush=True)
    print(
        obs.groupby(["cohort", "slide_id", "age"], observed=True)
        .size()
        .to_string(),
        flush=True,
    )
    print("\nvar names", flush=True)
    print(list(adata.var_names), flush=True)

    indices = np.linspace(0, adata.n_obs - 1, 1000, dtype=int)
    values = np.asarray(adata.X[indices, :])
    print(
        "\nX sampled min/max/mean",
        np.nanmin(values),
        np.nanmax(values),
        np.nanmean(values),
        flush=True,
    )
    print(
        "fraction integer",
        np.mean(np.isclose(values, np.rint(values))),
        "fraction zero",
        np.mean(values == 0),
        "nan",
        np.isnan(values).sum(),
        flush=True,
    )
    library_size = values.sum(axis=1)
    print(
        "sample library size min/median/max",
        np.min(library_size),
        np.median(library_size),
        np.max(library_size),
        flush=True,
    )
    print(
        "spatial",
        adata.obsm["spatial"].shape,
        np.asarray(adata.obsm["spatial"][:5]),
        flush=True,
    )


if __name__ == "__main__":
    main()
