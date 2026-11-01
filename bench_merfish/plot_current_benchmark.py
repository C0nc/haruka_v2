"""Publication-style metric comparison for the current raw-only benchmark."""
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"


def main():
    OUT.mkdir(exist_ok=True)
    tab = pd.read_csv(ROOT / "results_hidden256_raw" / "summary.csv")
    h = tab[["diag_mean", "offdiag_mean", "assignment_margin"]].mean()
    hs = tab[["diag_mean", "offdiag_mean", "assignment_margin"]].std(ddof=1)

    # Fixed, raw-only baseline values from the audited benchmark contract.
    names = ["Haruka\n(hidden256)", "CINEMA-OT\n(raw)", "contrastiveVI", "CellCharter", "PCA"]
    margins = np.array([h.assignment_margin, 0.491515, 0.452, 0.221, 0.208])
    errors = np.array([hs.assignment_margin, 0, 0, 0, 0])
    colors = ["#2563eb", "#dc2626", "#64748b", "#64748b", "#64748b"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.2, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(names))
    bars = ax0.bar(x, margins, yerr=errors, capsize=4, color=colors, alpha=.92,
                   edgecolor="white", linewidth=.7)
    ax0.axhline(margins[1], color="#dc2626", ls="--", lw=1, alpha=.65)
    ax0.set_xticks(x, names, fontsize=8)
    ax0.set_ylabel("Assignment margin (diagonal − off-diagonal)")
    ax0.set_ylim(0, .68)
    ax0.set_title("Aggregate recovery", loc="left", weight="bold")
    ax0.grid(axis="y", alpha=.22)
    for b, v in zip(bars, margins):
        ax0.text(b.get_x()+b.get_width()/2, v+.018, f"{v:.3f}", ha="center", fontsize=8)
    ax0.text(.02, .02, "Haruka: mean ± SD over 3 model seeds\nCINEMA-OT shown raw (no cleanup)",
             transform=ax0.transAxes, fontsize=7.5, va="bottom", color="#374151")

    metric_names = ["Diagonal\n(higher is better)", "Off-diagonal\n(lower is better)"]
    xx = np.arange(2)
    width = .34
    b1 = ax1.bar(xx-width/2, [h.diag_mean, h.offdiag_mean], width, yerr=[hs.diag_mean, hs.offdiag_mean],
                 capsize=4, color="#2563eb", label="Haruka hidden256")
    b2 = ax1.bar(xx+width/2, [0.7032356, 0.2117206], width, color="#dc2626", label="CINEMA-OT raw")
    ax1.set_xticks(xx, metric_names, fontsize=8)
    ax1.set_ylim(0, .86)
    ax1.set_ylabel("Score")
    ax1.set_title("Where the margin comes from", loc="left", weight="bold")
    ax1.grid(axis="y", alpha=.22)
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    for bars_ in (b1, b2):
        for b in bars_:
            ax1.text(b.get_x()+b.get_width()/2, b.get_height()+.018, f"{b.get_height():.3f}",
                     ha="center", fontsize=8)
    fig.suptitle("Haruka v2 vs baselines — fixed simulator, raw-only scoring", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(OUT / "current_metric_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "current_metric_comparison.pdf", bbox_inches="tight")
    print(OUT / "current_metric_comparison.png")


if __name__ == "__main__":
    main()
