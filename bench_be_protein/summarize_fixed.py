"""Add the paired-graph Haruka result to the two-domain baseline comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def main():
    old = pd.read_csv(ROOT / "summary_2domain" / "be_protein_two_domain_by_section.csv")
    rows = []
    for path in sorted(
        (ROOT / "results_2domain_fixed").glob(
            "*/haruka_paired_graph_mse_seed0/be_two_domain_ari.json"
        )
    ):
        value = json.loads(path.read_text())
        rows.append(
            {
                "section": value["section"],
                "method": "Haruka-paired-graph",
                "method_id": value["method"],
                "background_domain_ARI": value["background_domain_ari"],
                "salient_domain_ARI": value["salient_domain_ari"],
                "two_domain_mean_ARI": value["two_domain_mean_ari"],
                "score_file": str(path.resolve()),
            }
        )
    detail = pd.concat([pd.DataFrame(rows), old], ignore_index=True)
    order = [
        "Haruka-paired-graph",
        "Haruka-MENDER",
        "contrastiveVI+CC",
        "CINEMA-OT+CC",
        "MSE-AE+CC",
        "PCA+CC",
    ]
    detail["method"] = pd.Categorical(detail["method"], order, ordered=True)
    detail = detail.sort_values(["method", "section"])
    metrics = ["background_domain_ARI", "salient_domain_ARI", "two_domain_mean_ARI"]
    grouped = detail.groupby("method", observed=True)[metrics]
    mean, std = grouped.mean().reindex(order), grouped.std(ddof=1).reindex(order)
    out = ROOT / "summary_2domain_fixed"
    out.mkdir(exist_ok=True)
    detail.to_csv(out / "be_protein_two_domain_fixed_by_section.csv", index=False)
    pd.concat({"mean": mean, "std": std}, axis=1).to_csv(
        out / "be_protein_two_domain_fixed_mean_std.csv"
    )

    colors = ["#dc2626", "#3b82f6", "#8b5cf6", "#f59e0b", "#10b981", "#64748b"]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, metric, title in zip(
        axes,
        metrics[:2],
        ["Background domain (neigh_name)", "Simulated salient domain"],
    ):
        ax.bar(x, mean[metric], yerr=std[metric], color=colors, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("ARI (mean ± SD, five sections)")
    fig.suptitle("BE CODEX: paired identifiability + residual graph smoothing", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "be_protein_two_domain_fixed_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(mean.round(4).to_string())
    print("\nSD\n" + std.round(4).to_string())


if __name__ == "__main__":
    main()
