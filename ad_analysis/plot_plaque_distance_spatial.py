#!/usr/bin/env python
"""Create spatial plaque-distance maps for the latest STARmap and CODEX AD runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


CODEX_PX_UM = 0.3774
CODEX_PLAQUE_FILES = {
    "AD-1": "11162021_Bahareh-3155_EDTA_abeta.csv",
    "AD-2": "10202021_BaharehNP_2997-ADCortex_EDTA_abeta.csv",
    "AD-3": "10122021_BaharehNP_ADCortex_EDTA_abeta.csv",
    "AD-4": "12032021-Bahareh_3280_Alz_abeta.csv",
}


def nearest(query: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    if not len(anchors):
        return np.full(len(query), np.nan)
    return cKDTree(anchors).query(query, k=1)[0]


def spatial_panel(ax, xy, values, title, vmax, anchors=None, unit=""):
    points = ax.scatter(
        xy[:, 0], xy[:, 1], c=np.clip(values, 0, vmax), s=2.5,
        cmap="viridis_r", vmin=0, vmax=vmax, linewidths=0, rasterized=True,
    )
    if anchors is not None and len(anchors):
        ax.scatter(
            anchors[:, 0], anchors[:, 1], s=5, c="black", marker="x",
            linewidths=0.35, alpha=0.65, rasterized=True,
        )
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    colorbar = plt.colorbar(points, ax=ax, fraction=0.045, pad=0.02)
    colorbar.set_label(f"distance{unit} (clipped {vmax:g})", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)


def plot_starmap(data_path: Path, out: Path):
    data = ad.read_h5ad(data_path)
    xy = np.asarray(data.obsm["spatial"], dtype=float)
    slices = sorted(
        data.obs.loc[data.obs["condition"].astype(str).eq("disease"), "slice_id"]
        .astype(str).unique()
    )
    records = []
    cached = {}
    for sample in slices:
        rows = np.flatnonzero(data.obs["slice_id"].astype(str).to_numpy() == sample)
        obs = data.obs.iloc[rows]
        tau_anchor = xy[rows[obs["plaque_tau"].to_numpy(dtype=bool)]]
        abeta_anchor = xy[rows[obs["plaque_abeta"].to_numpy(dtype=bool)]]
        any_anchor = np.vstack([abeta_anchor, tau_anchor])
        distances = {
            "Aβ-associated cell": nearest(xy[rows], abeta_anchor),
            "p-tau-associated cell": nearest(xy[rows], tau_anchor),
            "Aβ or p-tau-associated cell": nearest(xy[rows], any_anchor),
        }
        cached[sample] = (rows, abeta_anchor, tau_anchor, distances)
        for local, row in enumerate(rows):
            records.append(
                {
                    "cell_id": str(data.obs_names[row]),
                    "slice_id": sample,
                    "distance_to_abeta_associated_cell_spatial_units": distances["Aβ-associated cell"][local],
                    "distance_to_ptau_associated_cell_spatial_units": distances["p-tau-associated cell"][local],
                    "distance_to_either_associated_cell_spatial_units": distances["Aβ or p-tau-associated cell"][local],
                }
            )
    pd.DataFrame(records).to_csv(out / "starmap_plaque_distance_cells.csv.gz", index=False)

    all_values = np.concatenate(
        [np.concatenate(list(item[3].values())) for item in cached.values()]
    )
    vmax = float(np.nanpercentile(all_values, 98))
    fig, axes = plt.subplots(len(slices), 3, figsize=(14, 5.2 * len(slices)), squeeze=False)
    labels = ["Aβ-associated cell", "p-tau-associated cell", "Aβ or p-tau-associated cell"]
    for row_index, sample in enumerate(slices):
        rows, abeta_anchor, tau_anchor, distances = cached[sample]
        anchors = [abeta_anchor, tau_anchor, np.vstack([abeta_anchor, tau_anchor])]
        for col, label in enumerate(labels):
            spatial_panel(
                axes[row_index, col], xy[rows], distances[label],
                f"{sample}: distance to nearest {label}", vmax,
                anchors=anchors[col], unit=" (spatial units)",
            )
    fig.suptitle(
        "STARmap AD: plaque-distance tissue maps\n"
        "Distances are to annotated plaque-associated cells within the same slice; black × marks anchors",
        y=0.995, fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out / "starmap_spatial_distance_to_abeta_ptau.png", dpi=240)
    plt.close(fig)


def plot_codex(data_path: Path, plaque_dir: Path, out: Path):
    data = ad.read_h5ad(data_path)
    xy = np.asarray(data.obsm["spatial"], dtype=float)
    donors = sorted(data.obs.loc[data.obs["condition"].astype(str).eq("disease"), "donor_id"].astype(str).unique())
    cached = {}
    records = []
    for donor in donors:
        rows = np.flatnonzero(data.obs["donor_id"].astype(str).to_numpy() == donor)
        plaques = pd.read_csv(plaque_dir / CODEX_PLAQUE_FILES[donor], usecols=["class", "x", "y"])
        dense = plaques.loc[plaques["class"].eq("dense"), ["x", "y"]].to_numpy(float)
        diffuse = plaques.loc[plaques["class"].eq("diffuse"), ["x", "y"]].to_numpy(float)
        dense_distance = nearest(xy[rows], dense) * CODEX_PX_UM
        diffuse_distance = nearest(xy[rows], diffuse) * CODEX_PX_UM
        native_distance = data.obs.iloc[rows]["Distance to nearest Plaque (µm)"].to_numpy(float)
        cached[donor] = (rows, dense, diffuse, dense_distance, diffuse_distance, native_distance)
        for local, row in enumerate(rows):
            records.append(
                {
                    "cell_id": str(data.obs_names[row]),
                    "donor_id": donor,
                    "distance_to_dense_abeta_centroid_um": dense_distance[local],
                    "distance_to_diffuse_abeta_centroid_um": diffuse_distance[local],
                    "native_distance_to_any_plaque_um": native_distance[local],
                }
            )
    pd.DataFrame(records).to_csv(out / "codex_plaque_distance_cells.csv.gz", index=False)

    vmax = 200.0
    fig, axes = plt.subplots(len(donors), 3, figsize=(14, 4.5 * len(donors)), squeeze=False)
    for row_index, donor in enumerate(donors):
        rows, dense, diffuse, dense_distance, diffuse_distance, native_distance = cached[donor]
        spatial_panel(
            axes[row_index, 0], xy[rows], dense_distance,
            f"{donor}: dense Aβ centroid", vmax, anchors=dense, unit=" (µm)",
        )
        spatial_panel(
            axes[row_index, 1], xy[rows], diffuse_distance,
            f"{donor}: diffuse Aβ centroid", vmax, anchors=diffuse, unit=" (µm)",
        )
        spatial_panel(
            axes[row_index, 2], xy[rows], native_distance,
            f"{donor}: native nearest plaque", vmax, anchors=None, unit=" (µm)",
        )
    fig.suptitle(
        "CODEX AD: dense/diffuse Aβ plaque-distance tissue maps\n"
        "Centroid distances use official plaque-object coordinates and 0.3774 µm/pixel; black × marks centroids; no tau marker is available",
        y=0.995, fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "codex_spatial_distance_to_dense_diffuse_abeta.png", dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--starmap", required=True, type=Path)
    parser.add_argument("--codex", required=True, type=Path)
    parser.add_argument("--codex-plaque-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plot_starmap(args.starmap, args.out)
    plot_codex(args.codex, args.codex_plaque_dir, args.out)


if __name__ == "__main__":
    main()
