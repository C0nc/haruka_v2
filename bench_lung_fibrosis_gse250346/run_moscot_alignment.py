#!/usr/bin/env python3
"""Align paired GSE250346 cores with MOSCOT using background z-CC only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from moscot.problems.space import AlignmentProblem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--rank", type=int, default=128)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source = ad.read_h5ad(args.landmarks)
    outputs = []
    audit = []
    for donor in source.obs["donor"].astype(str).unique():
        block = source[source.obs["donor"].astype(str) == donor].copy()
        reference_rows = block.obs["condition"].astype(str) == "reference"
        references = block.obs.loc[reference_rows, "slice_id"].astype(str).unique()
        if len(references) != 1:
            raise ValueError(f"Donor {donor} requires exactly one less-affected reference core")
        reference = references[0]
        problem = AlignmentProblem(block)
        problem.prepare(
            batch_key="slice_id",
            spatial_key="spatial",
            joint_attr="X_z_cc",
            policy="star",
            reference=reference,
            normalize_spatial=True,
            cost={"xy": "sq_euclidean", "x": "sq_euclidean", "y": "sq_euclidean"},
        )
        problem.solve(
            alpha=args.alpha,
            epsilon=args.epsilon,
            rank=args.rank,
            max_iterations=300,
            threshold=1e-3,
            device="gpu",
        )
        problem.align(mode="warp", key_added="X_moscot_aligned")
        aligned = np.asarray(block.obsm["X_moscot_aligned"], dtype=np.float32)
        original = np.asarray(block.obsm["spatial"], dtype=np.float32)
        result = block.obs.copy()
        result["original_x"] = original[:, 0]
        result["original_y"] = original[:, 1]
        result["aligned_x"] = aligned[:, 0]
        result["aligned_y"] = aligned[:, 1]
        outputs.append(result)
        audit.append({
            "donor": donor,
            "reference_slice": reference,
            "target_slices": block.obs.loc[~reference_rows, "slice_id"].astype(str).unique().tolist(),
            "n_landmarks": block.n_obs,
            "alpha": args.alpha,
            "epsilon": args.epsilon,
            "rank": args.rank,
            "joint_feature": "X_z_cc",
        })

        slices = block.obs["slice_id"].astype(str).unique()
        colors = {value: plt.cm.tab10(i % 10) for i, value in enumerate(slices)}
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        for value in slices:
            take = block.obs["slice_id"].astype(str).to_numpy() == value
            axes[0].scatter(original[take, 0], original[take, 1], s=3, color=colors[value], label=value, alpha=0.65)
            axes[1].scatter(aligned[take, 0], aligned[take, 1], s=3, color=colors[value], label=value, alpha=0.65)
        target = block.obs["condition"].astype(str).to_numpy() == "fibrosis"
        cluster = pd.Categorical(block.obs["s_cc_cluster"].astype(str))
        axes[2].scatter(aligned[~target, 0], aligned[~target, 1], s=2, color="lightgray", alpha=0.25, label="less affected reference")
        for code, category in enumerate(cluster.categories):
            take = target & (cluster.codes == code)
            if np.any(take):
                axes[2].scatter(aligned[take, 0], aligned[take, 1], s=3, color=plt.cm.tab20(code % 20), label=f"S{category}", alpha=0.7)
        axes[0].set_title("Original core coordinates")
        axes[1].set_title("MOSCOT aligned by background z-CC")
        axes[2].set_title("Aligned geometry with salient clusters overlaid")
        for ax in axes:
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.legend(markerscale=3, fontsize=6, frameon=False, ncol=2)
        fig.suptitle(f"GSE250346 paired fibrosis cores — donor {donor}")
        fig.tight_layout()
        fig.savefig(args.out_dir / f"moscot_alignment_{donor}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    pd.concat(outputs).to_csv(args.out_dir / "moscot_aligned_landmarks.csv.gz", compression="gzip")
    (args.out_dir / "alignment_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
