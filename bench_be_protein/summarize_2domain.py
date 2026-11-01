"""Aggregate and plot the corrected two-domain BE protein benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_2domain"
OUT = ROOT / "summary_2domain"
DISPLAY = {
    "haruka_mender_protein_mse": "Haruka-MENDER",
    "contrastivevi_cellcharter_protein_mse": "contrastiveVI+CC",
    "cinemaot_protein_cellcharter": "CINEMA-OT+CC",
    "cellcharter_protein_mse": "MSE-AE+CC",
    "pca_cellcharter_protein": "PCA+CC",
}


def main() -> None:
    rows = []
    for path in sorted(RESULTS.glob("*/*/be_two_domain_ari.json")):
        result = json.loads(path.read_text())
        if result["method"] not in DISPLAY:
            continue
        rows.append(
            {
                "section": result["section"],
                "method": DISPLAY[result["method"]],
                "method_id": result["method"],
                "background_domain_ARI": result["background_domain_ari"],
                "salient_domain_ARI": result["salient_domain_ari"],
                "two_domain_mean_ARI": result["two_domain_mean_ari"],
                "score_file": str(path.resolve()),
            }
        )
    detail = pd.DataFrame(rows)
    if len(detail) != 25:
        raise RuntimeError(f"Expected 25 scores, found {len(detail)}")
    order = list(DISPLAY.values())
    detail["method"] = pd.Categorical(detail["method"], order, ordered=True)
    detail = detail.sort_values(["method", "section"])
    metrics = [
        "background_domain_ARI",
        "salient_domain_ARI",
        "two_domain_mean_ARI",
    ]
    grouped = detail.groupby("method", observed=True)[metrics]
    mean = grouped.mean().reindex(order)
    std = grouped.std(ddof=1).reindex(order)
    OUT.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUT / "be_protein_two_domain_by_section.csv", index=False)
    pd.concat({"mean": mean, "std": std}, axis=1).to_csv(
        OUT / "be_protein_two_domain_mean_std.csv"
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), sharey=True)
    colors = ["#3b82f6", "#8b5cf6", "#f59e0b", "#10b981", "#64748b"]
    x = np.arange(len(order))
    for ax, metric, title in zip(
        axes,
        metrics[:2],
        ["Background domain (neigh_name)", "Simulated salient domain"],
    ):
        ax.bar(
            x,
            mean[metric].to_numpy(),
            yerr=std[metric].to_numpy(),
            color=colors,
            capsize=3,
        )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("ARI (mean ± SD across five sections)")
    fig.suptitle("BE CODEX paired simulation: two-domain recovery", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "be_protein_two_domain_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(mean.round(4).to_string())
    print("\nSD")
    print(std.round(4).to_string())


if __name__ == "__main__":
    main()
