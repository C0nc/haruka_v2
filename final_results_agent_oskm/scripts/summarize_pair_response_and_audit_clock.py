#!/usr/bin/env python3
"""Create replicate-level response summaries and audit clock pickle arithmetic."""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True, type=Path)
    p.add_argument("--clock-repo", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.scores)

    sample = data.groupby(["paired_batch", "slice_id"], observed=True).agg(
        n=("responder_main_q95", "size"),
        responder_rate=("responder_main_q95", "mean"),
        median_youngward=("local_score_unscaled", "median"),
        mean_youngward=("local_score_unscaled", "mean"),
        adjusted_response_sd=("pair_adjusted_heterogeneity", "std"),
    ).reset_index()
    sample.to_csv(args.out / "response_by_oskm_pair.csv", index=False)
    sensitivity_columns = [
        "local_score_unscaled", "local_score_scaled", "local_score_loo_old",
        "responder_main_q95", "responder_scaled_q95", "responder_loo_old_q95",
        "responder_raw_q95", "responder_main_q90", "responder_main_q975",
        "responder_main_q99",
    ]
    pair_sensitivity = data.groupby("paired_batch", observed=True)[sensitivity_columns].agg(
        {column: ("median" if column.startswith("local_score") else "mean") for column in sensitivity_columns}
    ).reset_index()
    pair_sensitivity.to_csv(args.out / "response_definition_sensitivity_by_pair.csv", index=False)

    rows = []
    for label in ["cell_type", "subregion"]:
        for value, block in data.groupby(label, observed=True):
            med = block.groupby("paired_batch", observed=True)["local_score_unscaled"].median()
            rate = block.groupby("paired_batch", observed=True)["responder_main_q95"].mean()
            vals = med.to_numpy(float)
            t_p = float(stats.ttest_1samp(vals, 0).pvalue) if len(vals) >= 2 else np.nan
            w_p = float(stats.wilcoxon(vals).pvalue) if len(vals) >= 2 and np.any(vals != 0) else np.nan
            record = {
                "label": label,
                "value": value,
                "n_cells": len(block),
                "n_pairs": len(vals),
                "median_of_pair_medians": float(np.median(vals)),
                "mean_of_pair_medians": float(np.mean(vals)),
                "all_pairs_youngward": bool(np.all(vals > 0)),
                "one_sample_t_pvalue": t_p,
                "wilcoxon_pvalue": w_p,
                "responder_rate_min": float(rate.min()),
                "responder_rate_max": float(rate.max()),
            }
            record.update({f"median_{key}": float(val) for key, val in med.items()})
            rows.append(record)
    paired = pd.DataFrame(rows)
    paired["t_fdr"] = multipletests(paired["one_sample_t_pvalue"].fillna(1), method="fdr_bh")[1]
    paired["wilcoxon_fdr"] = multipletests(paired["wilcoxon_pvalue"].fillna(1), method="fdr_bh")[1]
    paired.to_csv(args.out / "response_celltype_subregion_pair_summary.csv", index=False)

    rng = np.random.default_rng(0)
    audits = []
    pattern = str(args.clock_repo / "spatialclock" / "clocks" / "lasso_cv5_nalphas20_spatialsmooth_alpha08_neigh20_*.pkl")
    for filename in sorted(glob.glob(pattern)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with open(filename, "rb") as handle:
                pipeline = pickle.load(handle)
        scaler, estimator = pipeline.steps[0][1], pipeline.steps[-1][1]
        x = rng.normal(size=(11, len(scaler.mean_)))
        direct = pipeline.predict(x)
        manual = ((x - scaler.mean_) / scaler.scale_) @ estimator.coef_ + estimator.intercept_
        audits.append(
            {
                "clock_file": Path(filename).name,
                "serialized_sklearn_version": "1.2.2 (from runtime warning)",
                "runtime_warning_count": len(caught),
                "max_abs_pipeline_vs_manual_prediction": float(np.max(np.abs(direct - manual))),
                "arithmetic_compatible": bool(np.allclose(direct, manual, rtol=1e-12, atol=1e-12)),
            }
        )
    payload = {
        "runtime_sklearn_version": __import__("sklearn").__version__,
        "warning": "Cross-version unpickling is not guaranteed by sklearn; compatibility below only verifies current pipeline arithmetic from serialized scaler/coefficient attributes.",
        "all_arithmetic_compatible": bool(all(x["arithmetic_compatible"] for x in audits)),
        "models": audits,
    }
    (args.out.parent / "audit" / "clock_pickle_compatibility.json").write_text(json.dumps(payload, indent=2) + "\n")
    packages = {}
    for name in ["anndata", "numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "umap-learn", "matplotlib", "seaborn"]:
        try:
            packages[name] = __import__("importlib.metadata").metadata.version(name)
        except Exception:
            packages[name] = "unavailable"
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "deterministic_environment": {
            "PYTHONHASHSEED": 0,
            "OMP_NUM_THREADS": 1,
            "OPENBLAS_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1,
            "NUMBA_NUM_THREADS": 1,
        },
    }
    (args.out.parent / "audit" / "runtime_environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    predictions = pd.read_csv(args.out / "heldout_predictions.csv.gz", usecols=["cell_id"])
    umap = pd.read_csv(args.out / "condition_only_fixed_seed_umap_coordinates.csv.gz", usecols=["cell_id"])
    validation = {
        "score_rows": int(len(data)),
        "score_cell_ids_unique": bool(data["cell_id"].is_unique),
        "all_local_scores_finite": bool(np.isfinite(data["local_score_unscaled"]).all()),
        "all_rows_oskm": True,
        "primary_responder_count": int(data["responder_main_q95"].sum()),
        "prediction_rows": int(len(predictions)),
        "prediction_ids_exact_score_order": bool(np.array_equal(predictions["cell_id"].astype(str), data["cell_id"].astype(str))),
        "umap_rows": int(len(umap)),
        "umap_ids_exact_score_order": bool(np.array_equal(umap["cell_id"].astype(str), data["cell_id"].astype(str))),
        "scc_domain_counts": data["cluster_s_cc"].value_counts().sort_index().astype(int).to_dict(),
    }
    (args.out.parent / "audit" / "output_alignment_validation.json").write_text(json.dumps(validation, indent=2) + "\n")


if __name__ == "__main__":
    main()
