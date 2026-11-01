"""Summarize the final correctly annotated BE protein benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_2domain_celltype"
OUT = ROOT / "summary_2domain_celltype_final"
METHOD_DIRS = {
    "Haruka CellType+Expression": "haruka_celltype_final_seed0",
    "contrastiveVI+CC": "contrastivevi_protein_seed0",
    "CINEMA-OT+CC": "cinemaot_cellcharter_seed0",
    "MSE-AE+CC": "cellcharter_protein_seed0",
    "PCA+CC": "pca_protein_seed0",
}


def read_scores(directory):
    rows = []
    for path in sorted(RESULTS.glob(f"*/{directory}/be_two_domain_ari.json")):
        value = json.loads(path.read_text())
        rows.append({
            "section": value["section"],
            "background_domain_ARI": value["background_domain_ari"],
            "salient_domain_ARI": value["salient_domain_ari"],
            "two_domain_mean_ARI": value["two_domain_mean_ari"],
            "score_file": str(path.resolve()),
        })
    return rows


def main():
    rows = []
    for method, directory in METHOD_DIRS.items():
        for row in read_scores(directory):
            row["method"] = method
            rows.append(row)
    detail = pd.DataFrame(rows)
    if len(detail) != 25:
        raise RuntimeError(f"Expected 25 final benchmark rows, found {len(detail)}")
    order = list(METHOD_DIRS)
    detail["method"] = pd.Categorical(detail["method"], order, ordered=True)
    detail = detail.sort_values(["method", "section"])
    metrics = ["background_domain_ARI", "salient_domain_ARI", "two_domain_mean_ARI"]
    grouped = detail.groupby("method", observed=True)[metrics]
    mean, std = grouped.mean().reindex(order), grouped.std(ddof=1).reindex(order)
    OUT.mkdir(exist_ok=True)
    detail.to_csv(OUT / "be_celltype_final_by_section.csv", index=False)
    pd.concat({"mean": mean, "std": std}, axis=1).to_csv(
        OUT / "be_celltype_final_mean_std.csv"
    )

    ablations = {
        "Expression only": "haruka_celltype_expression_only_seed0",
        "Cell Type only": "haruka_celltype_celltype_only_seed0",
        "Equal 16+16": "haruka_celltype_balanced_seed0",
        "Expression 24 + Cell Type 8": "haruka_celltype_final_seed0",
    }
    ablation_rows = []
    for label, directory in ablations.items():
        values = read_scores(directory)
        scores = [v["background_domain_ARI"] for v in values]
        ablation_rows.append({
            "background_representation": label,
            "background_ARI_mean": np.mean(scores),
            "background_ARI_std": np.std(scores, ddof=1),
        })
    pd.DataFrame(ablation_rows).to_csv(
        OUT / "background_celltype_fusion_ablation.csv", index=False
    )

    colors = ["#dc2626", "#8b5cf6", "#f59e0b", "#10b981", "#64748b"]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for ax, metric, title in zip(
        axes, metrics[:2],
        ["Background niche (`neigh_name`)", "Simulated salient domain"],
    ):
        ax.bar(x, mean[metric], yerr=std[metric], color=colors, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=38, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("ARI (mean ± SD across five sections)")
    fig.suptitle("BE CODEX with correct fine `Cell Type` composition", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "be_celltype_final_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(mean.round(4).to_string())
    print("\nSD\n" + std.round(4).to_string())
    print("\nBackground ablation\n" + pd.DataFrame(ablation_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
