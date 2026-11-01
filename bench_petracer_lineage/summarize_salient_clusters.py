"""Aggregate PEtracer s-CC cluster interpretation across sections."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get(
    "HARUKA_PETRACER_RESULTS",
    Path(__file__).resolve().parent / "results_lambda2_22k",
))
RUN = os.environ.get("HARUKA_PETRACER_RUN", "haruka_cellcharter_seed0")
OUT = ROOT / "summary" / "salient_cluster_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)

association_paths = sorted(ROOT.glob(f"*/{RUN}/salient_cluster_diagnostic/salient_associations.csv"))
profile_paths = sorted(ROOT.glob(f"*/{RUN}/salient_cluster_diagnostic/s_cc_cluster_profiles.csv"))
associations = pd.concat([pd.read_csv(path) for path in association_paths], ignore_index=True)
profiles = pd.concat([pd.read_csv(path) for path in profile_paths], ignore_index=True)
associations.to_csv(OUT / "all_section_associations.csv", index=False)
profiles.to_csv(OUT / "all_cluster_profiles.csv", index=False)

summary = (
    associations.groupby(["representation", "annotation", "association"], as_index=False)
    .agg(score_mean=("score", "mean"), score_std=("score", "std"), ari_mean=("ari", "mean"), ari_std=("ari", "std"))
)
summary.to_csv(OUT / "association_mean_std.csv", index=False)

s_cc = summary[summary["representation"].eq("s_cc")].sort_values("score_mean", ascending=False)
fig, axis = plt.subplots(figsize=(10, 7))
axis.barh(np.arange(len(s_cc)), s_cc["score_mean"], xerr=s_cc["score_std"], color="#4C9F9B", alpha=0.85)
axis.set_yticks(np.arange(len(s_cc)), s_cc["annotation"])
axis.invert_yaxis()
axis.set_xlabel("Association with six s-CC clusters (NMI or η²; mean ± SD)")
axis.set_title("What PEtracer s-CC clusters reflect across four sections")
fig.tight_layout()
fig.savefig(OUT / "s_cc_association_ranking.png", dpi=200, bbox_inches="tight")
plt.close(fig)

focus = summary[summary["annotation"].isin([
    "lineage_family", "lineage_clade", "lineage_eligible", "cell_subtype",
    "tissue_niche", "boundary_bin", "within_tumor", "paired_change_fraction",
    "s_latent_norm", "fov",
])]
print(focus.sort_values(["representation", "score_mean"], ascending=[True, False]).to_string(index=False))
