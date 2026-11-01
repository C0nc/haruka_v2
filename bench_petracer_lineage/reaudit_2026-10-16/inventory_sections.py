#!/usr/bin/env python3
"""Per-section inventory of tumour / non-tumour regions in the PEtracer h5td files (read-only)."""
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

SRC = Path("/home/yancui/Jill/fourteen/petracer_data")
OUT = Path("/home/yancui/haruka_v2/bench_petracer_lineage/reaudit_2026-10-16")
rows, tables = [], []
for mouse in ("M1", "M2", "M3"):
    with h5py.File(SRC / f"{mouse}_tumor_tracing.h5td", "r") as h:
        obs = ad.io.read_elem(h["obs"])
        print(mouse, list(h.keys()), list(h["obsm"].keys()), list(h["layers"].keys()) if "layers" in h else None)
    if mouse == "M1":
        for c in ("tumor", "within_tumor", "tumor_boundary_dist", "lung_boundary_dist", "cell_neighborhood", "type", "cell_subtype"):
            v = obs[c]
            print(c, v.dtype, v.nunique(), v.value_counts(dropna=False).head(12).to_dict() if v.nunique() < 40 else v.describe().to_dict())
    obs["within_tumor"] = obs["within_tumor"].astype(str)
    for sec, o in obs.groupby("sample", observed=True):
        wt = o["within_tumor"].isin(["True", "1", "1.0"])
        d = pd.to_numeric(o["tumor_boundary_dist"], errors="coerce")
        mal = o["cell_subtype"].astype(str).eq("Malignant")
        row = dict(section=sec, n=len(o), n_within_tumor=int(wt.sum()), n_outside=int((~wt).sum()),
                   n_tumors=int(o.loc[wt, "tumor"].astype(str).replace("nan", np.nan).dropna().nunique()),
                   malignant_frac_inside=float(mal[wt].mean()) if wt.any() else np.nan,
                   malignant_frac_outside=float(mal[~wt].mean()),
                   dist_min=float(d.min()), dist_median_in=float(d[wt].median()) if wt.any() else np.nan,
                   dist_median_out=float(d[~wt].median()), dist_nan=int(d.isna().sum()))
        ct = pd.crosstab(o["cell_subtype"].astype(str), np.where(wt, "tumour", "nontumour"))
        nonmal = ct.drop(index="Malignant", errors="ignore")
        both = nonmal[(nonmal.get("tumour", 0) >= 200) & (nonmal.get("nontumour", 0) >= 200)]
        row["n_nonmal_types_both_ge200"] = len(both)
        row["types_both_ge200"] = ";".join(both.index)
        row["n_nonmal_in_both_types_tumour"] = int(both.get("tumour", pd.Series(dtype=int)).sum())
        row["n_nonmal_in_both_types_nontumour"] = int(both.get("nontumour", pd.Series(dtype=int)).sum())
        rows.append(row)
        ct = ct.reset_index().assign(section=sec)
        tables.append(ct)
        print(row, flush=True)
pd.DataFrame(rows).to_csv(OUT / "section_region_inventory.csv", index=False)
pd.concat(tables).to_csv(OUT / "section_celltype_by_region.csv", index=False)
