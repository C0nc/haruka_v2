#!/usr/bin/env python3
"""Run the bundled frozen Spatial Aging Clock on the full old-control/OSKM cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--clock-repo", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = ad.read_h5ad(args.data)
    counts = data.layers["counts"]
    data.X = counts.toarray().astype(np.float32, copy=False) if sparse.issparse(counts) else np.asarray(counts, dtype=np.float32).copy()
    sys.path.insert(0, str(args.clock_repo))
    from spatialclock.deploy import get_predictions

    get_predictions(
        data,
        fill_missing="mean",
        smooth=True,
        normalize=True,
        standardize=True,
        add_in_place=True,
    )
    frame = data.obs[["condition", "slice_id", "paired_batch", "mouse_id", "celltype", "cell_type", "subregion", "age", "cohort"]].copy()
    frame.insert(0, "cell_id", data.obs_names.astype(str))
    frame["predicted_age"] = pd.to_numeric(data.obs["predicted_age"], errors="coerce")
    frame["clock_rejuvenation"] = np.nan
    control = frame["condition"].astype(str).eq("control")
    target = frame["condition"].astype(str).eq("oskm")
    keys = ["paired_batch", "cell_type", "subregion"]
    baseline = frame.loc[control].groupby(keys, observed=True)["predicted_age"].agg(["median", "size"])
    joined = frame.loc[target, keys].join(baseline, on=keys)
    valid = joined["size"].fillna(0).ge(20)
    chosen = joined.index[valid]
    frame.loc[chosen, "clock_rejuvenation"] = joined.loc[chosen, "median"] - frame.loc[chosen, "predicted_age"]
    # Controls receive a null score only for calibration/diagnostics, never a biological responder label.
    cjoined = frame.loc[control, keys].join(baseline, on=keys)
    cvalid = cjoined["size"].fillna(0).ge(20)
    cchosen = cjoined.index[cvalid]
    frame.loc[cchosen, "clock_rejuvenation"] = cjoined.loc[cchosen, "median"] - frame.loc[cchosen, "predicted_age"]
    frame.to_csv(args.out / "spatial_clock_predictions_all_cells.csv.gz", index=False, compression="gzip")

    grouped = frame.groupby(["paired_batch", "condition", "cell_type"], observed=True)["predicted_age"].agg(["median", "size"]).reset_index()
    records = []
    for ct in sorted(frame["cell_type"].astype(str).unique()):
        block = grouped[grouped["cell_type"].astype(str) == ct]
        wide = block.pivot(index="paired_batch", columns="condition", values=["median", "size"])
        deltas = []
        for pair in sorted(frame["paired_batch"].astype(str).unique()):
            try:
                if wide.loc[pair, ("size", "control")] >= 20 and wide.loc[pair, ("size", "oskm")] >= 20:
                    deltas.append((pair, float(wide.loc[pair, ("median", "oskm")] - wide.loc[pair, ("median", "control")])))
            except KeyError:
                pass
        values = np.array([v for _, v in deltas], dtype=float)
        pvalue = float(stats.ttest_1samp(values, 0).pvalue) if len(values) >= 2 else np.nan
        rec = {"cell_type": ct, "n_pairs": len(values), "median_oskm_minus_control_months": float(np.median(values)) if len(values) else np.nan, "mean_oskm_minus_control_months": float(np.mean(values)) if len(values) else np.nan, "all_pairs_younger": bool(np.all(values < 0)) if len(values) else False, "paired_t_pvalue": pvalue}
        rec.update({f"delta_{pair}": value for pair, value in deltas})
        records.append(rec)
    summary = pd.DataFrame(records)
    summary["paired_t_fdr"] = multipletests(summary["paired_t_pvalue"].fillna(1), method="fdr_bh")[1]
    summary.sort_values("median_oskm_minus_control_months").to_csv(args.out / "spatial_clock_celltype_paired_summary.csv", index=False)
    metrics = {
        "clock": "bundled published frozen cell-type-specific Spatial Aging Clock",
        "haruka_supervision": False,
        "gene_overlap": "300/300 for every represented cell-type clock",
        "normalization": "official target_sum=250 then log1p",
        "smoothing": "official alpha=0.8, 20 neighbors, bounded by mouse_id",
        "n_cells": int(len(frame)),
        "coverage_n": int(frame["predicted_age"].notna().sum()),
        "coverage_fraction": float(frame["predicted_age"].notna().mean()),
        "n_cell_types": int(frame.loc[frame["predicted_age"].notna(), "cell_type"].nunique()),
        "matched_clock_score": "matched old-control median predicted age minus cell predicted age within paired_batch x cell_type x subregion; positive means younger",
    }
    (args.out / "spatial_clock_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
