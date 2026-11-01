"""Aggregate the four independent M1-T1 paired-section evaluations."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(os.environ.get(
    "HARUKA_PETRACER_RESULTS",
    Path(__file__).resolve().parent / "results_lambda2_22k",
))
RUN = os.environ.get("HARUKA_PETRACER_RUN", "haruka_cellcharter_asym_seed0")
OUT = ROOT / "summary"
OUT.mkdir(parents=True, exist_ok=True)

frames = []
for path in sorted(ROOT.glob(f"*/{RUN}/petracer_evaluation/all_clustering_scores.csv")):
    frames.append(pd.read_csv(path))
scores = pd.concat(frames, ignore_index=True)
scores.to_csv(OUT / "all_sections_scores.csv", index=False)

summary = (
    scores.groupby(["representation", "truth"])[["ari", "nmi"]]
    .agg(["mean", "std"])
    .reset_index()
)
summary.columns = ["representation", "truth", "ari_mean", "ari_std", "nmi_mean", "nmi_std"]
summary.to_csv(OUT / "mean_std_scores.csv", index=False)

truths = ["background_niche", "lineage_family", "lineage_clade"]
representations = ["z", "s", "z_latent", "s_latent", "raw_expression_pca", "paired_delta_oracle"]
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for axis, truth in zip(axes, truths):
    subset = scores[scores["truth"].eq(truth)]
    values = [subset.loc[subset["representation"].eq(rep), "ari"].to_numpy() for rep in representations]
    axis.boxplot(values, tick_labels=["z-CC", "s-CC", "z latent", "s latent", "raw", "paired Δ"], showmeans=True)
    axis.axhline(0, color="0.75", linewidth=0.8)
    axis.set_title(truth.replace("_", " "))
    axis.set_ylabel("ARI")
    axis.tick_params(axis="x", rotation=35)
fig.suptitle("PEtracer M1-T1 paired benchmark (default asymmetric z-CC/s-CC)")
fig.tight_layout()
fig.savefig(OUT / "ari_boxplot.png", dpi=180, bbox_inches="tight")

print(summary[summary["truth"].isin(truths)].to_string(index=False))
