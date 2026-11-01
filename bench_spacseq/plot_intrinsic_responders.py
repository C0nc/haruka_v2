"""Visualize per-function and per-guide intrinsic responder scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    prefix = Path(args.prefix)
    cells = pd.read_csv(prefix.with_suffix(".cells.csv"))
    summary = json.loads(prefix.with_suffix(".summary.json").read_text())
    function = pd.DataFrame(summary["by_function"]).sort_values(
        "responder_fraction", ascending=False
    )
    guide = pd.DataFrame(summary["by_guide"])
    order = function.function.tolist()
    guide["guide_number"] = guide.guide_identity.str.rsplit("_", n=1).str[-1]

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    sns.barplot(
        data=function, x="function", y="responder_fraction", order=order,
        color="#4C78A8", ax=axes[0, 0],
    )
    axes[0, 0].axhline(
        summary["control_false_positive_fraction"], color="black", ls="--",
        lw=1.5, label="NT null",
    )
    axes[0, 0].set_title("Intrinsic responder fraction by target function")
    axes[0, 0].set_ylabel("Responder fraction")
    axes[0, 0].tick_params(axis="x", rotation=35)
    axes[0, 0].legend(frameon=False)

    sns.barplot(
        data=guide, x="function", y="responder_fraction", hue="guide_number",
        order=order, palette="Set2", ax=axes[0, 1],
    )
    axes[0, 1].axhline(
        summary["control_false_positive_fraction"], color="black", ls="--", lw=1.5
    )
    axes[0, 1].set_title("Independent sgRNAs within each function")
    axes[0, 1].set_ylabel("Responder fraction")
    axes[0, 1].tick_params(axis="x", rotation=35)
    axes[0, 1].legend(title="guide", frameon=False)

    target = cells[~cells.is_control].copy()
    sns.violinplot(
        data=target, x="function", y="intrinsic_response_percentile",
        order=order, inner="quart", cut=0, color="#72B7B2", ax=axes[1, 0],
    )
    axes[1, 0].axhline(0.95, color="black", ls="--", lw=1.5)
    axes[1, 0].set_title("State-matched NT response percentile")
    axes[1, 0].set_ylabel("Intrinsic response percentile")
    axes[1, 0].tick_params(axis="x", rotation=35)

    sns.scatterplot(
        data=target, x="ecm_minus_epithelial_score",
        y="intrinsic_response_percentile", hue="responder",
        palette={False: "#B7B7B7", True: "#D62728"}, alpha=0.65, s=22,
        linewidth=0, ax=axes[1, 1],
    )
    rho = summary["response_state_spearman"]["rho"]
    axes[1, 1].axhline(0.95, color="black", ls="--", lw=1.5)
    axes[1, 1].set_title(f"Response versus ECM–epithelial state (Spearman ρ={rho:.253f})".replace(".253", ".3"))
    axes[1, 1].set_xlabel("ECM minus epithelial RNA score")
    axes[1, 1].set_ylabel("Intrinsic response percentile")
    axes[1, 1].legend(title="responder", frameon=False)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {output}")


if __name__ == "__main__":
    main()
