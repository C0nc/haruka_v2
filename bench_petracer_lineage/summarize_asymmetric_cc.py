"""Compare original and asymmetric PEtracer CellCharter readouts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent / "results_lambda2_22k"
OUT = ROOT / "summary" / "asymmetric_cc"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = {
    "original CC: all cells, 3 hops, no self": "haruka_cellcharter_seed0",
    "asymmetric CC: malignant, 1 hop + self": "haruka_cellcharter_asym_seed0",
}

frames = []
for variant, run_name in RUNS.items():
    for path in sorted(ROOT.glob(f"*/{run_name}/petracer_evaluation/all_clustering_scores.csv")):
        frame = pd.read_csv(path)
        frame["variant"] = variant
        frames.append(frame)
scores = pd.concat(frames, ignore_index=True)
scores.to_csv(OUT / "all_section_scores.csv", index=False)

summary = (
    scores.groupby(["variant", "representation", "truth"])[["ari", "nmi"]]
    .agg(["mean", "std"])
    .reset_index()
)
summary.columns = ["variant", "representation", "truth", "ari_mean", "ari_std", "nmi_mean", "nmi_std"]
summary.to_csv(OUT / "mean_std_scores.csv", index=False)

metrics = [
    ("z", "background_niche", "z → background"),
    ("s", "lineage_family", "s → family"),
    ("s", "lineage_clade", "s → clade"),
    ("z", "lineage_family", "z → family leakage"),
    ("s", "background_niche", "s → background leakage"),
]
fig, axis = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(metrics))
width = 0.36
for index, variant in enumerate(RUNS):
    means, errors = [], []
    for rep, truth, _ in metrics:
        row = summary[
            summary["variant"].eq(variant)
            & summary["representation"].eq(rep)
            & summary["truth"].eq(truth)
        ].iloc[0]
        means.append(row["ari_mean"])
        errors.append(row["ari_std"])
    axis.bar(x + (index - 0.5) * width, means, width, yerr=errors, capsize=3, label=variant)
axis.axhline(0, color="0.7", linewidth=0.8)
axis.set_xticks(x, [label for _, _, label in metrics], rotation=20, ha="right")
axis.set_ylabel("ARI, mean ± SD across four sections")
axis.set_title("PEtracer asymmetric salient CellCharter readout")
axis.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "before_after_ari.png", dpi=200, bbox_inches="tight")
plt.close(fig)

focus = []
for variant in RUNS:
    for rep, truth, label in metrics:
        row = summary[
            summary["variant"].eq(variant)
            & summary["representation"].eq(rep)
            & summary["truth"].eq(truth)
        ].iloc[0]
        focus.append({"variant": variant, "metric": label, "ari_mean": row["ari_mean"], "ari_std": row["ari_std"], "nmi_mean": row["nmi_mean"], "nmi_std": row["nmi_std"]})
focus = pd.DataFrame(focus)
focus.to_csv(OUT / "primary_comparison.csv", index=False)
print(focus.to_string(index=False))
