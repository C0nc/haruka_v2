#!/usr/bin/env python3
"""Collect sample-separation diagnostics from the unified real-data rerun."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    metric_paths = sorted(args.root.glob("*/sample_batch_diagnostic/sample_separation_metrics.csv"))
    if not metric_paths:
        raise FileNotFoundError(f"No sample diagnostics below {args.root}")
    frames = []
    for path in metric_paths:
        frame = pd.read_csv(path)
        frame.insert(0, "dataset_dir", path.parents[1].name)
        metadata_path = path.with_name("sample_umap_metadata.json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            frame["sample_key"] = metadata.get("sample_key")
            frame["n_condition"] = metadata.get("n_condition")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(args.root / "sample_batch_diagnostic_summary.csv", index=False)

    matrix = result.pivot(index="study", columns="representation", values="sample_kmeans_nmi")
    order = ["z intrinsic", "s intrinsic", "z-CC background", "s-CC salient"]
    matrix = matrix.reindex(columns=[column for column in order if column in matrix])
    height = max(4.0, 0.48 * len(matrix) + 2.0)
    fig, ax = plt.subplots(figsize=(9, height))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="mako_r",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "sample-predictive clustering NMI (lower = better mixing)"},
        ax=ax,
    )
    ax.set_title("Condition-only residual sample structure after decoder batch conditioning")
    ax.set(xlabel="", ylabel="")
    fig.tight_layout()
    fig.savefig(args.root / "sample_batch_diagnostic_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
