"""Prepare the official CODEX-CNS myeloid protein dataset for Haruka.

The source is the paper repository's figure7_mg_raw.h5ad.  X contains mean
continuous fluorescence intensity for ten myeloid proteins, not RNA counts.
We follow the publication preprocessing: log1p, donor ComBat, then feature
z-scoring (clipped at 10).  The downstream Haruka protein adapter applies its
own robust [-5, 5] clip/affine shift and an exact summed-MSE reconstruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE / "external" / "CODEX-CNS"


def prepare(source: Path, clusters: Path, exclude_aml: bool) -> ad.AnnData:
    a = ad.read_h5ad(source)
    raw = np.asarray(a.X, dtype=np.float32)
    if np.isnan(raw).any() or np.isinf(raw).any() or (raw < 0).any():
        raise ValueError("Expected finite, nonnegative continuous intensities")

    cluster = pd.read_csv(clusters, index_col=0)
    if not a.obs_names.astype(str).equals(cluster.index.astype(str)):
        raise ValueError("Official raw H5AD and cluster CSV observation indices differ")

    a.obs["donor_id"] = a.obs["ImageID"].astype(str)
    a.obs["condition"] = a.obs["ImageType"].astype(str).map(
        {"Ctl": "control", "AD": "disease"}
    )
    if a.obs["condition"].isna().any():
        raise ValueError("Unexpected ImageType value")
    a.obs["region"] = a.obs["Parent"].astype(str).map(
        {"Grey Matter": "GM", "White Matter": "WM"}
    )
    a.obs["paper_initial_leiden"] = cluster["leiden"].astype(str).to_numpy()
    a.obs["is_aml_control"] = a.obs["donor_id"].eq("Ctl-2")

    distance = a.obs["Distance to nearest Plaque (µm)"].to_numpy(dtype=float)
    available = np.isfinite(distance)
    a.obs["plaque_annotation_available"] = available
    for radius in (0, 5, 30):
        values = np.full(a.n_obs, -1, dtype=np.int8)
        values[available] = (distance[available] <= radius).astype(np.int8)
        a.obs[f"plaque_within_{radius}um"] = values
    score = np.full(a.n_obs, np.nan, dtype=np.float32)
    score[available] = np.exp(-distance[available] / 30.0).astype(np.float32)
    a.obs["plaque_proximity_score_30um"] = score

    if exclude_aml:
        a = a[~a.obs["is_aml_control"].to_numpy()].copy()
        raw = np.asarray(a.X, dtype=np.float32)

    a.layers["raw_intensity"] = raw.copy()
    a.layers["log1p_intensity"] = np.log1p(raw).astype(np.float32)
    a.X = a.layers["log1p_intensity"].copy()
    sc.pp.combat(a, key="donor_id", inplace=True)
    sc.pp.scale(a, zero_center=True, max_value=10)
    a.X = np.asarray(a.X, dtype=np.float32)
    a.layers["protein"] = a.X.copy()
    a.obsm["spatial"] = a.obs[["spatial_X", "spatial_Y"]].to_numpy(
        dtype=np.float32
    )

    for key in ("donor_id", "condition", "region", "paper_initial_leiden"):
        a.obs[key] = a.obs[key].astype("category")
    a.uns["codex_cns_preprocessing"] = {
        "modality": "continuous CODEX mean fluorescence intensity per segmented cell",
        "markers": a.var_names.astype(str).tolist(),
        "transform": "log1p -> ComBat(donor_id) -> feature z-score clipped at +10",
        "downstream_likelihood": "exact summed mean squared error per cell",
        "source_h5ad": str(source.resolve()),
        "source_cluster_csv": str(clusters.resolve()),
        "aml_control_excluded": bool(exclude_aml),
        "aml_control_id": "Ctl-2",
        "paper_url": "https://www.nature.com/articles/s41593-026-02267-3",
        "code_url": "https://github.com/BaharehAjami/CODEX-CNS",
    }
    return a


def audit(a: ad.AnnData, out: Path) -> None:
    rows = []
    for (donor, condition, region), frame in a.obs.groupby(
        ["donor_id", "condition", "region"], observed=True
    ):
        d = frame["Distance to nearest Plaque (µm)"].to_numpy(dtype=float)
        rows.append(
            {
                "donor_id": donor,
                "condition": condition,
                "region": region,
                "n_cells": len(frame),
                "n_plaque_annotated": int(np.isfinite(d).sum()),
                "n_plaque_within_5um": int(np.sum(d <= 5)),
                "n_plaque_within_30um": int(np.sum(d <= 30)),
            }
        )
    pd.DataFrame(rows).to_csv(out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_REPO
        / "otsm_analysis/codex-cns/figure-notebooks/data/figure7_mg_raw.h5ad",
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        default=DEFAULT_REPO
        / "otsm_analysis/codex-cns/figure-notebooks/data/figure7_clusters.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=HERE / "data" / "processed")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for tag, exclude in (("primary_no_aml", True), ("sensitivity_all8", False)):
        a = prepare(args.source, args.clusters, exclude_aml=exclude)
        path = args.out_dir / f"codex_cns_myeloid_{tag}.h5ad"
        a.write_h5ad(path, compression="gzip")
        audit(a, args.out_dir / f"codex_cns_myeloid_{tag}_audit.csv")
        outputs[tag] = {
            "path": str(path.resolve()),
            "n_cells": a.n_obs,
            "n_proteins": a.n_vars,
            "donors": a.obs["donor_id"].astype(str).value_counts().to_dict(),
            "conditions": a.obs["condition"].astype(str).value_counts().to_dict(),
        }
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(outputs, indent=2) + "\n"
    )
    print(json.dumps(outputs, indent=2), flush=True)


if __name__ == "__main__":
    main()
