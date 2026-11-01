"""Prepare CODEX-CNS myeloid protein data with per-donor per-channel normalization.

Motivation (audit 2026-09-20): the shipped `no_combat` variant applies only a
GLOBAL feature z-score, which by construction cannot remove a per-donor
intensity shift.  That shift is present in the raw MFI across all ten markers
(donor-mean spread 0.39-1.31 within-donor SD), i.e. a per-section staining /
exposure floor rather than one bad antibody.  It survives preprocessing, is
inherited by the encoder, and is then amplified ~7x by the 3-hop CellCharter
aggregation, which is what produces the donor islands in the z-CC / s-CC UMAPs.

Fix: classic CODEX per-channel normalization -- subtract each donor's own
per-marker background percentile, divide by its own dynamic range, then log1p
and a global feature z-score as before.

This is LABEL-FREE.  ComBat(donor_id) and within-arm centering both remove the
donor offset more completely, but donor is nested in condition here, so they
also remove (or leak) the AD-vs-control contrast.  This transform never sees
the condition label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

import prepare_codex_cns_no_combat as base

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE / "external" / "CODEX-CNS"


def channel_normalize(raw: np.ndarray, donor: np.ndarray,
                      bg_pct: float, top_pct: float) -> np.ndarray:
    """Per donor, per marker: subtract background percentile, scale by range."""
    out = np.empty_like(raw, dtype=np.float64)
    for d in np.unique(donor):
        m = donor == d
        v = raw[m].astype(np.float64)
        bg = np.percentile(v, bg_pct, axis=0)
        top = np.percentile(v, top_pct, axis=0)
        out[m] = np.clip(v - bg, 0.0, None) / (top - bg + 1e-9)
    return out


def prepare(source: Path, clusters: Path, exclude_aml: bool,
            bg_pct: float, top_pct: float) -> ad.AnnData:
    # Reuse the official obs construction, plaque annotation and AML exclusion.
    a = base.prepare(source, clusters, exclude_aml)

    raw = np.asarray(a.layers["raw_intensity"], dtype=np.float32)
    donor = a.obs["donor_id"].astype(str).to_numpy()
    scaled = channel_normalize(raw, donor, bg_pct, top_pct)

    a.layers["channel_normalized"] = scaled.astype(np.float32)
    a.X = np.log1p(scaled * 100.0).astype(np.float32)
    sc.pp.scale(a, zero_center=True, max_value=10)
    a.X = np.asarray(a.X, dtype=np.float32)
    a.layers["protein"] = a.X.copy()

    a.uns["codex_cns_preprocessing"] = dict(a.uns["codex_cns_preprocessing"])
    a.uns["codex_cns_preprocessing"].update({
        "transform": (f"per-donor per-marker p{bg_pct:g} background subtraction / "
                      f"p{top_pct:g} range scaling -> log1p(x*100) -> global feature "
                      f"z-score clipped at +10; label-free, no ComBat"),
        "channel_norm_background_percentile": bg_pct,
        "channel_norm_top_percentile": top_pct,
        "channel_norm_rationale": "removes per-section staining/exposure floor without "
                                  "using the condition label (donor is nested in condition)",
    })
    return a


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=DEFAULT_REPO /
                   "otsm_analysis/codex-cns/figure-notebooks/data/figure7_mg_raw.h5ad")
    p.add_argument("--clusters", type=Path, default=DEFAULT_REPO /
                   "otsm_analysis/codex-cns/figure-notebooks/data/figure7_clusters.csv")
    p.add_argument("--out-dir", type=Path, default=HERE / "data" / "processed")
    p.add_argument("--bg-pct", type=float, default=25.0)
    p.add_argument("--top-pct", type=float, default=95.0)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    a = prepare(args.source, args.clusters, True, args.bg_pct, args.top_pct)
    tag = "primary_no_aml_channelnorm"
    path = args.out_dir / f"codex_cns_myeloid_{tag}.h5ad"
    a.write_h5ad(path, compression="gzip")
    base.audit(a, args.out_dir / f"codex_cns_myeloid_{tag}_audit.csv")
    print(json.dumps({"path": str(path.resolve()), "n_cells": int(a.n_obs),
                      "n_proteins": int(a.n_vars),
                      "transform": a.uns["codex_cns_preprocessing"]["transform"],
                      "donors": a.obs["donor_id"].astype(str).value_counts().to_dict()},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
