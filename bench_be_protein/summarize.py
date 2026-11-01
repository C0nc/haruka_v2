"""Aggregate the five-section BE protein benchmark and draw the ARI comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SUMMARY = ROOT / "summary"

DISPLAY = {
    "haruka_mender_protein_mse": "Haruka-MENDER",
    "contrastivevi_cellcharter_protein_mse": "contrastiveVI+CC",
    "cinemaot_protein_cellcharter": "CINEMA-OT+CC",
    "cellcharter_protein_mse": "MSE-AE+CC",
    "pca_cellcharter_protein": "PCA+CC",
}

FACTORS = {
    "background": "Background composition",
    "spatial": "Background niche",
    "salient": "Salient intrinsic",
    "salient_spatial": "Salient spatial",
}


def main() -> None:
    rows = []
    for path in sorted(RESULTS.glob("*/*/be_four_factor_ari.json")):
        result = json.loads(path.read_text())
        method = result.get("method")
        if method not in DISPLAY:
            continue
        row = {
            "section": result["section"],
            "method": DISPLAY[method],
            "method_id": method,
            "diagonal_mean": result["diagonal_mean"],
            "off_diagonal_mean": result["off_diagonal_mean"],
            "assignment_margin": result["assignment_margin"],
            "score_file": str(path.resolve()),
        }
        row.update({FACTORS[key]: result["diagonal"][key] for key in FACTORS})
        rows.append(row)

    detail = pd.DataFrame(rows)
    expected = len(DISPLAY) * 5
    if len(detail) != expected:
        raise RuntimeError(f"Expected {expected} score files, found {len(detail)}")

    order = list(DISPLAY.values())
    detail["method"] = pd.Categorical(detail["method"], order, ordered=True)
    detail = detail.sort_values(["method", "section"])
    SUMMARY.mkdir(parents=True, exist_ok=True)
    detail.to_csv(SUMMARY / "be_protein_four_factor_by_section.csv", index=False)

    metric_cols = list(FACTORS.values()) + [
        "diagonal_mean",
        "off_diagonal_mean",
        "assignment_margin",
    ]
    grouped = detail.groupby("method", observed=True)[metric_cols]
    mean = grouped.mean().reindex(order)
    std = grouped.std(ddof=1).reindex(order)
    table = pd.concat({"mean": mean, "std": std}, axis=1)
    table.to_csv(SUMMARY / "be_protein_four_factor_mean_std.csv")

    long = detail.melt(
        id_vars=["section", "method"],
        value_vars=list(FACTORS.values()),
        var_name="factor",
        value_name="ARI",
    )
    colors = ["#3b82f6", "#8b5cf6", "#f59e0b", "#10b981", "#64748b"]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2), sharey=True)
    x = np.arange(len(order))
    for ax, factor in zip(axes, FACTORS.values()):
        vals = mean[factor].to_numpy()
        errs = std[factor].to_numpy()
        ax.bar(x, vals, yerr=errs, color=colors, capsize=3, width=0.72)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(factor, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=42, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("ARI (mean ± SD across five sections)")
    fig.suptitle("BE CODEX paired simulation: four-factor recovery", y=1.02)
    fig.tight_layout()
    fig.savefig(SUMMARY / "be_protein_four_factor_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(mean.round(4).to_string())
    print(f"\nSaved summary to {SUMMARY}")


if __name__ == "__main__":
    main()
