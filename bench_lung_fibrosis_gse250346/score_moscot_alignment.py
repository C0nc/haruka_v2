#!/usr/bin/env python3
"""Score paired-core MOSCOT alignment without using salient labels to fit it."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def cross_arm_agreement(reference: pd.DataFrame, target: pd.DataFrame, columns: list[str]) -> dict:
    nn = NearestNeighbors(n_neighbors=1).fit(reference[columns].to_numpy(float))
    distance, index = nn.kneighbors(target[columns].to_numpy(float))
    matched = reference.iloc[index[:, 0]]
    return {
        "mean_cross_arm_distance": float(distance.mean()),
        "cell_type_agreement": float((matched["cell_type"].to_numpy() == target["cell_type"].to_numpy()).mean()),
        "lineage_agreement": float((matched["lineage"].to_numpy() == target["lineage"].to_numpy()).mean()),
        "CNiche_agreement": float((matched["CNiche"].to_numpy() == target["CNiche"].to_numpy()).mean()),
        "TNiche_agreement": float((matched["TNiche"].to_numpy() == target["TNiche"].to_numpy()).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    landmarks = ad.read_h5ad(args.landmarks)
    aligned = pd.read_csv(args.aligned, index_col=0)
    if len(aligned) != landmarks.n_obs:
        raise ValueError("Aligned table and landmark object differ in length")
    rows = []
    for donor, donor_frame in aligned.groupby("donor", observed=True):
        donor_frame = donor_frame.copy()
        original_scaled = np.zeros((len(donor_frame), 2), dtype=float)
        for _, indices in donor_frame.groupby("slice_id", observed=True).groups.items():
            local = donor_frame.index.get_indexer(indices)
            original_scaled[local] = StandardScaler().fit_transform(
                donor_frame.loc[indices, ["original_x", "original_y"]]
            )
        donor_frame["scaled_original_x"] = original_scaled[:, 0]
        donor_frame["scaled_original_y"] = original_scaled[:, 1]
        reference = donor_frame[donor_frame["condition"] == "reference"]
        for slice_id, target in donor_frame[donor_frame["condition"] == "fibrosis"].groupby("slice_id", observed=True):
            for geometry, columns in (
                ("original_shape_standardized", ["scaled_original_x", "scaled_original_y"]),
                ("moscot_zcc_aligned", ["aligned_x", "aligned_y"]),
            ):
                score = cross_arm_agreement(reference, target, columns)
                score.update({"donor": donor, "target_slice": slice_id, "geometry": geometry})
                rows.append(score)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out_dir / "moscot_alignment_metrics.csv", index=False)

    long = metrics.melt(
        id_vars=["donor", "target_slice", "geometry"],
        value_vars=["cell_type_agreement", "lineage_agreement", "CNiche_agreement", "TNiche_agreement"],
        var_name="metric", value_name="agreement",
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=long, x="metric", y="agreement", hue="geometry", errorbar="sd", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("nearest cross-arm landmark agreement")
    ax.set_title("GSE250346 paired-core alignment: background z-CC MOSCOT")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(args.out_dir / "moscot_alignment_agreement_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
