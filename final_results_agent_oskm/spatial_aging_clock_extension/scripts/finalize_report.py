#!/usr/bin/env python3
"""Create the self-contained Spatial Aging Clock extension report and manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/yancui/haruka_v2/final_results_agent_oskm/spatial_aging_clock_extension")
REPO = Path("/home/yancui/haruka_v2/bench_oskm_reprogramming/external/SpatialAgingClock")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def held_bootstrap() -> pd.DataFrame:
    fold = pd.read_csv(ROOT / "tables/held_pair_prediction_folds.csv")
    fold = fold[fold.split == "leave_pair_out"]
    rng = np.random.default_rng(20260913)
    rows = []
    for (rep, endpoint), d in fold.groupby(["representation", "endpoint"], observed=True):
        for metric in ["r2", "spearman", "mae"]:
            x = d[metric].to_numpy(float)
            boots = np.mean(x[rng.integers(0, len(x), (10000, len(x)))], axis=1)
            rows.append({"representation": rep, "endpoint": endpoint, "metric": metric,
                         "mean_across_held_pairs": float(x.mean()),
                         "pair_bootstrap_ci_low": float(np.percentile(boots, 2.5)),
                         "pair_bootstrap_ci_high": float(np.percentile(boots, 97.5)),
                         "n_pairs": len(x), "bootstrap_replicates": 10000})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "tables/held_pair_prediction_pair_bootstrap_ci.csv", index=False)
    return out


def main() -> None:
    held_ci = held_bootstrap()
    pc = pd.read_csv(ROOT / "tables/pc_predicted_age_associations.csv")
    dpt = pd.read_csv(ROOT / "tables/dpt_predicted_age_associations.csv")
    held = pd.read_csv(ROOT / "tables/held_pair_prediction_metrics.csv")
    leak = pd.read_csv(ROOT / "tables/sample_leakage_pc_metrics.csv")
    pair = pd.read_csv(ROOT / "tables/clock_oskm_control_pair_shifts.csv")
    ct = pd.read_csv(ROOT / "tables/clock_oskm_control_celltype_pair_shifts.csv")
    validation = json.loads((ROOT / "validation_160k/run_metadata.json").read_text())["validation"]
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(["git", "-C", str(REPO), "remote", "get-url", "origin"], text=True).strip()
    versions = {p: md.version(p) for p in ["anndata", "scanpy", "squidpy", "numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib", "seaborn"]}
    versions.update({"spatial_aging_clock_git_commit": commit, "spatial_aging_clock_origin": remote,
                     "clock_pickle_training_sklearn": "1.2.2", "analysis_seed": 20260913})
    (ROOT / "software_and_source_versions.json").write_text(json.dumps(versions, indent=2) + "\n")

    commands = """#!/usr/bin/env bash
# Run from /home/yancui/haruka_v2/final_results_agent_oskm
# 1. Exact official-cache equivalence check (160,000 cells)
conda run --no-capture-output -n haruka2 python spatial_aging_clock_extension/scripts/run_official_clock_parallel.py --data /home/yancui/haruka_v2/bench_oskm_reprogramming/data/oskm_old_full_684253.h5ad --clock-repo /home/yancui/haruka_v2/bench_oskm_reprogramming/external/SpatialAgingClock --out /home/yancui/haruka_v2/final_results_agent_oskm/spatial_aging_clock_extension/validation_160k --workers 32 --restrict-cell-ids /home/yancui/haruka_v2/final_results_agent_oskm/clock/source_spatial_clock_cell_scores_160k.csv.gz --expected-cache /home/yancui/haruka_v2/final_results_agent_oskm/clock/source_spatial_clock_cell_scores_160k.csv.gz

# 2. Published frozen clock on all 684,253 old-control + old-OSKM cells
conda run --no-capture-output -n haruka2 python spatial_aging_clock_extension/scripts/run_official_clock_parallel.py --data /home/yancui/haruka_v2/bench_oskm_reprogramming/data/oskm_old_full_684253.h5ad --clock-repo /home/yancui/haruka_v2/bench_oskm_reprogramming/external/SpatialAgingClock --out /home/yancui/haruka_v2/final_results_agent_oskm/spatial_aging_clock_extension/clock_full --workers 32

# 3. Within-OSKM-slice s-CC diffusion pseudotime
conda run --no-capture-output -n haruka2 python spatial_aging_clock_extension/scripts/run_within_sample_dpt.py --cc-dir /home/yancui/haruka_v2/results_real_compgated_nb_full_oskm/oskm_old_full_684253/haruka_z_s_cellcharter_cc3_condition_only --umap-table /home/yancui/haruka_v2/final_results_agent_oskm/tables/condition_only_fixed_seed_umap_coordinates.csv.gz --out /home/yancui/haruka_v2/final_results_agent_oskm/spatial_aging_clock_extension/dpt --workers 4

# 4. Leakage-aware associations, bootstrap intervals, and figures
env OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 conda run --no-capture-output -n haruka2 python spatial_aging_clock_extension/scripts/analyze_clock_haruka_extension.py --clock /home/yancui/haruka_v2/final_results_agent_oskm/spatial_aging_clock_extension/clock_full/spatial_clock_predictions.csv.gz --data /home/yancui/haruka_v2/bench_oskm_reprogramming/data/oskm_old_full_684253.h5ad --cc-dir /home/yancui/haruka_v2/results_real_compgated_nb_full_oskm/oskm_old_full_684253/haruka_z_s_cellcharter_cc3_condition_only --umap /home/yancui/haruka_v2/final_results_agent_oskm/tables/condition_only_fixed_seed_umap_coordinates.csv.gz --dpt /home/yancui/haruka_v2/final_results_agent_oskm/spatial_aging_clock_extension/dpt/within_sample_scc_dpt.csv.gz --out /home/yancui/haruka_v2/final_results_agent_oskm/spatial_aging_clock_extension

# 5. Report and hashes
conda run -n haruka2 python spatial_aging_clock_extension/scripts/finalize_report.py
"""
    (ROOT / "exact_commands.sh").write_text(commands)

    zraw = held[(held.representation == "z_cc") & (held.endpoint == "predicted_age")].iloc[0]
    sraw = held[(held.representation == "s_cc") & (held.endpoint == "predicted_age")].iloc[0]
    zadj = held[(held.representation == "z_cc") & held.endpoint.str.contains("centered")].iloc[0]
    sadj = held[(held.representation == "s_cc") & held.endpoint.str.contains("centered")].iloc[0]
    dp = dpt[(dpt.endpoint == "dpt_min_pc1") & (dpt.scope == "pooled_within_slice")].iloc[0]
    dpa = dpt[(dpt.endpoint == "dpt_min_pc1") & (dpt.scope == "pooled_within_slice_celltype_subregion")].iloc[0]
    ds = dpt[(dpt.endpoint == "dpt_max_pc1") & (dpt.scope == "pooled_within_slice")].iloc[0]
    pc_top = pc.iloc[pc.within_slice_spearman.abs().argsort()[::-1][:6]]
    top_lines = "\n".join(
        f"- {r.representation} PC{int(r.pc)}: rho={r.within_slice_spearman:.3f}, spatial-block 95% CI [{r.within_slice_block_boot_ci_low:.3f}, {r.within_slice_block_boot_ci_high:.3f}], directions {int(r.positive_slices)} positive/{int(r.negative_slices)} negative slices; anatomy-adjusted rho={r.within_slice_celltype_subregion_spearman:.3f}."
        for _, r in pc_top.iterrows())
    pair_lines = "\n".join(
        f"- {r.paired_batch}: OSKM-control median matched-stratum shift {r.median_stratum_delta_months:+.3f} months; {100*r.fraction_strata_younger:.1f}% of evaluable cell-type×subregion strata younger."
        for _, r in pair.iterrows())
    report = f"""# Spatial Aging Clock extension: full old-control/old-OSKM cohort

## Bottom line

Eric Sun's published frozen cell-type clocks were directly applicable to all 684,253 cells (300/300 genes for every represented clock; 100% coverage). The exact parallel implementation reproduced the pre-existing 160,000-cell official run to maximum absolute error {validation['max_abs_difference']:.3g}. The full-cohort clock does **not** support a cohort-wide OSKM rejuvenation claim: only B9 shifted younger after matching cell type and subregion; B10, B13, and B14 shifted older. None of 15 cell types survived BH correction across the four pairs.

The background z-CC representation predicted the independent clock endpoint better than salient s-CC in strict leave-pair-out evaluation. Salient s-CC nevertheless contained reproducible but weaker clock-age axes. Because s-CC remains highly sample structured, these are associations, not evidence that its geometry is an aging trajectory.

## Published predictor audit

- Source: `{remote}`, clean local commit `{commit}`.
- Frozen model family: `lasso_cv5_nalphas20_spatialsmooth_alpha08_neigh20`, 18 cell-type pickle pipelines and matching mean files.
- Applied exactly as documented: total-count normalization to 250, log1p, spatial fixed-kNN (k=20), binary row-normalized adjacency, alpha=0.8 propagation, maximum 30 iterations, tolerance 0.01, then the frozen StandardScaler+Lasso pipeline.
- Smoothing was bounded by `mouse_id`/slice and by cell type. Parallelization was only over these mathematically independent groups.
- Compatibility: pickles record sklearn 1.2.2; analysis used {versions['scikit-learn']}. Direct cache validation (n={validation['n']:,}) gave mean absolute difference {validation['mean_abs_difference']:.3g}, maximum {validation['max_abs_difference']:.3g}, Pearson {validation['pearson']:.16g}.
- Young reference cells were not used. Haruka was not retrained.

## Clock shift under OSKM

Positive values below mean clock-older OSKM; negative values mean clock-younger OSKM. Each pair summary first computes OSKM-control medians within cell-type×subregion strata with at least 20 cells per condition, then summarizes strata equally.

{pair_lines}

At cell-type resolution, median four-pair shifts ranged from Neuroblast -1.899 months (only three evaluable pairs) and NSC -0.834 (three pairs) to Astrocyte +2.853 and VLMC +2.878 months. All BH-adjusted Wilcoxon q-values were at least {ct.fdr_bh.min():.3f}. Four biological pairs are too few for a sensitive confirmatory cell-type test.

## Held-pair prediction and leakage

PCA was refit inside every training fold, so held paired batches never influenced scaling or PC axes. The model was fixed Ridge(alpha=100) on PC1-PC10.

- Raw predicted age: z-CC R2={zraw.held_pair_r2:.3f}, Spearman={zraw.held_pair_spearman:.3f}, MAE={zraw.held_pair_mae:.3f} months; s-CC R2={sraw.held_pair_r2:.3f}, Spearman={sraw.held_pair_spearman:.3f}, MAE={sraw.held_pair_mae:.3f}. Spearman was positive in all four held pairs for both representations.
- After within-slice×cell-type×subregion median centering: z-CC R2={zadj.held_pair_r2:.3f}, Spearman={zadj.held_pair_spearman:.3f}; s-CC R2={sadj.held_pair_r2:.3f}, Spearman={sadj.held_pair_spearman:.3f}. Again, all four held-pair Spearman directions were positive.
- Random-cell splits inflated raw R2 by {zraw.r2_leakage_inflation:.3f} for z-CC and {sraw.r2_leakage_inflation:.3f} for s-CC; centered-endpoint inflation was {zadj.r2_leakage_inflation:.3f} and {sadj.r2_leakage_inflation:.3f}.
- Sample leakage remains severe in s-CC: PC1 slice eta-squared={leak[(leak.representation=='s_cc') & (leak.pc==1)].slice_eta_squared.iloc[0]:.3f}, and deterministic k-means(4) sample NMI={leak[leak.representation=='s_cc'].kmeans4_slice_nmi_pc1_10.iloc[0]:.3f}. z-CC k-means sample NMI was {leak[leak.representation=='z_cc'].kmeans4_slice_nmi_pc1_10.iloc[0]:.4f}.

Fold-resampling 95% intervals (10,000 resamples of the four held-pair metrics) are in `tables/held_pair_prediction_pair_bootstrap_ci.csv`. They quantify between-pair uncertainty but are necessarily coarse with n=4.

## PC associations with predicted spatial age

Primary correlations are pooled percentile-rank correlations within each OSKM slice. CIs are 500-replicate, slice-stratified 8×8 spatial-block bootstraps. Nominal cell-level P values for all 20 PCs were BH-adjusted together, but inference is based primarily on block CIs and direction consistency because neighboring cells are dependent.

{top_lines}

z-CC therefore carries the strongest clock-age axis (PC2). s-CC PC3 is the cleanest salient axis by all-four-slice direction consistency and persistence after cell-type/subregion adjustment. PC signs are arbitrary and should not be interpreted across refits.

## Within-slice s-CC diffusion pseudotime

DPT was fit independently in each OSKM slice after within-slice scaling and PCA(30), with k=30 and fixed seed. The primary root was chosen independently of clock age as the minimum within-slice s-CC PC1 cell; no orientation was flipped to improve correlation.

- Primary root: pooled within-slice rho={dp.spearman:.3f}, spatial-block 95% CI [{dp.ci_low:.3f}, {dp.ci_high:.3f}], negative in all four slices. After within-slice×cell-type×subregion rank adjustment, rho={dpa.spearman:.3f}.
- Opposite-root sensitivity: pooled rho={ds.spearman:.3f}, CI [{ds.ci_low:.3f}, {ds.ci_high:.3f}], with only 2/4 slices positive and 2/4 negative.

The marked root sensitivity means this static DPT is descriptive manifold position, not a robust temporal ordering, lineage, or causal rejuvenation trajectory.

## Outputs

Key figures:

- `figures/pc_predicted_age_associations.png`
- `figures/condition_only_umaps_predicted_age_and_dpt.png`
- `figures/spatial_maps_predicted_age_all_slices.png`
- `figures/spatial_maps_scc_dpt_oskm_slices.png`
- `figures/dpt_predicted_age_by_oskm_slice.png`
- `figures/held_pair_vs_random_cell_clock_prediction.png`

Key tables:

- `clock_full/spatial_clock_predictions.csv.gz` — all 684,253 predictions.
- `tables/oskm_clock_dpt_pc_scores.csv.gz` — aligned OSKM scores, UMAP, DPT, and PC1-PC10.
- `tables/pc_predicted_age_associations.csv` and `tables/pc_predicted_age_by_slice.csv`.
- `tables/dpt_predicted_age_associations.csv`.
- `tables/held_pair_prediction_metrics.csv`, fold metrics, and pair-bootstrap intervals.
- `tables/clock_oskm_control_pair_shifts.csv` and cell-type shifts.
- `tables/sample_leakage_pc_metrics.csv`.

Exact rerun commands are in `exact_commands.sh`; software/source versions are in `software_and_source_versions.json`; SHA-256 hashes are in `output_sha256.txt`.

## Claims and limitations

1. The clock is an independent published expression-to-age predictor, not a Haruka-trained proxy, and its deployment is validated exactly on the earlier official cache.
2. The strongest transferable clock information resides in z-CC, not s-CC; this is evidence that much clock age is background/compositional context rather than OSKM-specific salience.
3. s-CC does add weaker clock-age-associated axes, but sample structure and DPT root sensitivity prevent calling them rejuvenation trajectories.
4. Only one of four matched biological pairs shows an overall younger clock shift. Therefore no broad OSKM clock rejuvenation claim is warranted.
5. Cell-level P values are anti-conservative under spatial dependence. Spatial-block intervals, pair holdout, direction consistency, and n=4 pair uncertainty receive priority.
6. UMAP and DPT are descriptive. Neither establishes lineage, directionality, or causation.
"""
    (ROOT / "REPORT.md").write_text(report)

    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and "partials" not in p.parts and p.name != "output_sha256.txt")
    with (ROOT / "output_sha256.txt").open("w") as out:
        for path in files:
            out.write(f"{sha256(path)}  {path.relative_to(ROOT)}\n")
    print(json.dumps({"report": str(ROOT / "REPORT.md"), "n_hashed_outputs": len(files), "commit": commit}, indent=2))


if __name__ == "__main__":
    main()
