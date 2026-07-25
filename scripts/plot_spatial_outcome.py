#!/usr/bin/env python3
"""Spatial scatter of the Haruka two-latent outcome: ground truth vs inferred domains.

Usage:
    python plot_spatial_outcome.py --in_dir ./outputs/two_latent_viz
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

TRUTHS = ("layer", "lesion", "niche")
INFERRED = ("cluster_haruka_background", "cluster_haruka_salient")

# Sticky palette: same category index always gets the same colour across every panel.
PALETTE = np.array(plt.get_cmap("tab10").colors)


def _codes(values):
    vals = np.asarray(values).astype(str)
    cats = sorted(np.unique(vals))
    lut = {c: i for i, c in enumerate(cats)}
    return np.array([lut[v] for v in vals]), cats


def panel(ax, xy, values, title):
    codes, cats = _codes(values)
    ax.scatter(xy[:, 0], xy[:, 1], c=PALETTE[codes % len(PALETTE)], s=14, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
    return cats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default="./outputs/two_latent_viz")
    p.add_argument("--h5ad", default="spatial_rna_output.h5ad")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    path = os.path.join(args.in_dir, args.h5ad)
    ad = sc.read_h5ad(path)
    print(f"loaded {path}: {ad.n_obs} cells")
    print("slices:", dict(ad.obs["slice_id"].value_counts()) if "slice_id" in ad.obs else "n/a")

    cols = [c for c in TRUTHS + INFERRED if c in ad.obs.columns]
    print("plotting columns:", cols)

    slices = (
        list(dict.fromkeys(ad.obs["slice_id"].astype(str)))
        if "slice_id" in ad.obs.columns
        else ["all"]
    )

    fig, axes = plt.subplots(
        len(slices), len(cols), figsize=(2.5 * len(cols), 2.7 * len(slices)), squeeze=False
    )
    for r, sl in enumerate(slices):
        m = (
            np.ones(ad.n_obs, bool)
            if sl == "all"
            else (ad.obs["slice_id"].astype(str).values == sl)
        )
        xy = np.asarray(ad.obsm["spatial"])[m]
        for c, col in enumerate(cols):
            cats = panel(axes[r][c], xy, ad.obs[col].values[m], col if r == 0 else "")
            if c == 0:
                axes[r][c].set_ylabel(sl, fontsize=9)
            if r == len(slices) - 1:
                axes[r][c].set_xlabel(f"{len(cats)} levels", fontsize=7)

    fig.suptitle(
        "Haruka two-latent outcome — ground truth (left) vs inferred domains (right)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = args.out or os.path.join(args.in_dir, "spatial_outcome.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
