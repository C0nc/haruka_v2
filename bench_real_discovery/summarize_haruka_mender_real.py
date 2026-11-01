"""Create a compact comparison and biological interpretation report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "bench_real_discovery/results_haruka_mender"
OLD = ROOT / "bench_real_discovery/results_clean_cvi_symmetric_no_postfit"


def main() -> None:
    spac_new = json.loads((NEW / "spacseq/haruka_mender_seed0/analysis/metrics.json").read_text())
    spac_old = json.loads((OLD / "spacseq/haruka_both_clean_cvi_symmetric_no_postfit_seed0/analysis/metrics.json").read_text())
    ad_new = pd.read_csv(NEW / "ad/haruka_mender_seed0/analysis/four_slot_factor_matrix.csv", index_col=0)
    ad_old = pd.read_csv(OLD / "ad/haruka_both_clean_cvi_symmetric_no_postfit_seed0/analysis/four_slot_factor_matrix.csv", index_col=0)
    plaque_new = pd.read_csv(NEW / "ad/haruka_mender_seed0/plaque_analysis/plaque_recovery_summary.csv")
    plaque_old = pd.read_csv(OLD / "ad/haruka_both_clean_cvi_symmetric_no_postfit_seed0/plaque_analysis/plaque_recovery_summary.csv")

    def spac(result, slot, key):
        return result["representations"][slot][key]

    rows = [
        {"dataset": "SPARC", "factor": "cell type", "slot": "background", "metric": "LOSO BA",
         "Haruka-MENDER": spac(spac_new, "background", "cell_type_leave_slice_out_balanced_accuracy"),
         "previous Haruka": spac(spac_old, "background", "cell_type_leave_slice_out_balanced_accuracy")},
        {"dataset": "SPARC", "factor": "tissue niche", "slot": "background spatial", "metric": "LOSO BA",
         "Haruka-MENDER": spac(spac_new, "spatial", "niche_leave_slice_out_balanced_accuracy"),
         "previous Haruka": spac(spac_old, "spatial", "niche_leave_slice_out_balanced_accuracy")},
        {"dataset": "SPARC", "factor": "sgRNA", "slot": "salient intrinsic", "metric": "LOSO BA",
         "Haruka-MENDER": spac(spac_new, "salient", "target_anchor_perturbation_leave_slice_out_balanced_accuracy"),
         "previous Haruka": spac(spac_old, "salient", "target_anchor_perturbation_leave_slice_out_balanced_accuracy")},
        {"dataset": "SPARC", "factor": "sgRNA", "slot": "salient spatial", "metric": "within-slice CV BA",
         "Haruka-MENDER": spac(spac_new, "salient_spatial", "target_anchor_perturbation_balanced_accuracy"),
         "previous Haruka": spac(spac_old, "salient_spatial", "target_anchor_perturbation_balanced_accuracy")},
        {"dataset": "SPARC", "factor": "sgRNA", "slot": "salient spatial", "metric": "LOSO BA",
         "Haruka-MENDER": spac(spac_new, "salient_spatial", "target_anchor_perturbation_leave_slice_out_balanced_accuracy"),
         "previous Haruka": spac(spac_old, "salient_spatial", "target_anchor_perturbation_leave_slice_out_balanced_accuracy")},
        {"dataset": "AD", "factor": "cell type", "slot": "background", "metric": "grouped BA",
         "Haruka-MENDER": ad_new.loc["cell_type", "background"],
         "previous Haruka": ad_old.loc["cell_type", "background"]},
        {"dataset": "AD", "factor": "tissue domain", "slot": "background spatial", "metric": "grouped BA",
         "Haruka-MENDER": ad_new.loc["tissue_domain", "spatial"],
         "previous Haruka": ad_old.loc["tissue_domain", "spatial"]},
        {"dataset": "AD", "factor": "AD intrinsic program", "slot": "salient intrinsic", "metric": "cross-specimen rho",
         "Haruka-MENDER": ad_new.loc["intrinsic_program", "salient"],
         "previous Haruka": ad_old.loc["intrinsic_program", "salient"]},
        {"dataset": "AD", "factor": "AD local program", "slot": "salient spatial", "metric": "cross-specimen rho",
         "Haruka-MENDER": ad_new.loc["local_program", "salient_spatial"],
         "previous Haruka": ad_old.loc["local_program", "salient_spatial"]},
    ]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(NEW / "real_data_model_comparison.csv", index=False)

    long = comparison.melt(
        id_vars=["dataset", "factor", "slot", "metric"],
        value_vars=["Haruka-MENDER", "previous Haruka"],
        var_name="model", value_name="score",
    )
    long["label"] = long["dataset"] + ": " + long["factor"] + "\n" + long["slot"]
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(13, 5.5))
    sns.barplot(data=long, x="label", y="score", hue="model", ax=ax)
    ax.axhline(0, color="black", lw=.8)
    ax.set_xlabel("")
    ax.set_ylabel("evaluation score")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.set_title("Haruka-MENDER real-data decomposition versus previous clean Haruka")
    fig.tight_layout()
    fig.savefig(NEW / "real_data_model_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    guide = pd.read_csv(NEW / "spacseq/haruka_mender_seed0/biology/guide_reproducibility.csv")
    engagement = pd.read_csv(NEW / "spacseq/haruka_mender_seed0/biology/raw_target_engagement.csv")
    intrinsic = guide.nlargest(3, "intrinsic_cross_slice_cosine")
    spatial = guide.nlargest(3, "spatial_cross_slice_cosine")
    abeta = plaque_new.query("slot == 'salient' and plaque == 'plaque_abeta'").iloc[0]
    tau = plaque_new.query("slot == 'spatial' and plaque == 'plaque_tau'").iloc[0]
    salsp_tau = plaque_new.query("slot == 'salient_spatial' and plaque == 'plaque_tau'").iloc[0]
    old_salsp_tau = plaque_old.query("slot == 'salient_spatial' and plaque == 'plaque_tau'").iloc[0]

    report = f"""# Haruka-MENDER on real SPARC-seq and Jill AD

## Training contract

- All samples were jointly trained; no holdout and no post-fit regression.
- Training used only binary control/condition membership.
- SPARC sgRNA identity, cell type, and niche were evaluation-only.
- AD cell type, tissue domain, p-tau, and amyloid-beta were evaluation-only.
- Haruka-MENDER uses ContrastiveVI intrinsic encoders, dual niche-composition
  decoders, and independent MENDER ring aggregation; it has no SIMVI or GAT.

## SPARC-seq

The background intrinsic cell-type LOSO score is
{spac(spac_new, 'background', 'cell_type_leave_slice_out_balanced_accuracy'):.3f};
background spatial niche LOSO is
{spac(spac_new, 'spatial', 'niche_leave_slice_out_balanced_accuracy'):.3f}.

The salient-spatial representation separates sgRNA contexts within a slice
({spac(spac_new, 'salient_spatial', 'target_anchor_perturbation_balanced_accuracy'):.3f}
BA; chance 0.071), but does not transfer to an unseen slice
({spac(spac_new, 'salient_spatial', 'target_anchor_perturbation_leave_slice_out_balanced_accuracy'):.3f}).
Salient intrinsic is also near chance across slices
({spac(spac_new, 'salient', 'target_anchor_perturbation_leave_slice_out_balanced_accuracy'):.3f}).

Exploratory cross-slice direction candidates are intrinsic
{', '.join(intrinsic.guide)} and spatial {', '.join(spatial.guide)}. However,
none of the 14 named targets shows significant raw-RNA knockdown after BH
correction (minimum q={engagement.BH_q.min():.3f}). These are therefore
response-associated hypotheses, not validated on-target discoveries.

## Jill AD

The decomposition assigns cell identity to background ({ad_new.loc['cell_type', 'background']:.3f}),
tissue domain to background spatial ({ad_new.loc['tissue_domain', 'spatial']:.3f}),
and the intrinsic AD program to salient intrinsic
({ad_new.loc['intrinsic_program', 'salient']:.3f}).

The strongest reproducible disease discovery is amyloid-beta adjacency in
salient intrinsic: BA={abeta.balanced_accuracy:.3f}, AUROC={abeta.roc_auc:.3f}
under reciprocal disease-replicate transfer. p-tau is strongest in background
spatial: BA={tau.balanced_accuracy:.3f}, AUROC={tau.roc_auc:.3f}.

The new salient-spatial MENDER block is not a successful plaque representation:
p-tau BA={salsp_tau.balanced_accuracy:.3f}, versus
{old_salsp_tau.balanced_accuracy:.3f} for the previous GAT-based clean model.
Thus the supported new biological conclusion is an intrinsic A-beta-associated
AD state plus a shared spatial p-tau tissue context, not a distinct AD-specific
spatial plaque field.
"""
    (NEW / "REAL_DATA_RESULTS.md").write_text(report)
    print(comparison.to_string(index=False))
    print("\n" + report)


if __name__ == "__main__":
    main()
