# Spatial Aging Clock extension: full old-control/old-OSKM cohort

## Bottom line

Eric Sun's published frozen cell-type clocks were directly applicable to all 684,253 cells (300/300 genes for every represented clock; 100% coverage). The exact parallel implementation reproduced the pre-existing 160,000-cell official run to maximum absolute error 1.42e-14. The full-cohort clock does **not** support a cohort-wide OSKM rejuvenation claim: only B9 shifted younger after matching cell type and subregion; B10, B13, and B14 shifted older. None of 15 cell types survived BH correction across the four pairs.

The background z-CC representation predicted the independent clock endpoint better than salient s-CC in strict leave-pair-out evaluation. Salient s-CC nevertheless contained reproducible but weaker clock-age axes. Because s-CC remains highly sample structured, these are associations, not evidence that its geometry is an aging trajectory.

## Published predictor audit

- Source: `https://github.com/sunericd/SpatialAgingClock.git`, clean local commit `2b832f43da5c9426f246178a557d0b688e4b0546`.
- Frozen model family: `lasso_cv5_nalphas20_spatialsmooth_alpha08_neigh20`, 18 cell-type pickle pipelines and matching mean files.
- Applied exactly as documented: total-count normalization to 250, log1p, spatial fixed-kNN (k=20), binary row-normalized adjacency, alpha=0.8 propagation, maximum 30 iterations, tolerance 0.01, then the frozen StandardScaler+Lasso pipeline.
- Smoothing was bounded by `mouse_id`/slice and by cell type. Parallelization was only over these mathematically independent groups.
- Compatibility: pickles record sklearn 1.2.2; analysis used 1.7.2. Direct cache validation (n=160,000) gave mean absolute difference 6.13e-16, maximum 1.42e-14, Pearson 1.
- Young reference cells were not used. Haruka was not retrained.

## Clock shift under OSKM

Positive values below mean clock-older OSKM; negative values mean clock-younger OSKM. Each pair summary first computes OSKM-control medians within cell-type×subregion strata with at least 20 cells per condition, then summarizes strata equally.

- B10: OSKM-control median matched-stratum shift +3.286 months; 6.8% of evaluable cell-type×subregion strata younger.
- B13: OSKM-control median matched-stratum shift +2.134 months; 14.7% of evaluable cell-type×subregion strata younger.
- B14: OSKM-control median matched-stratum shift +0.946 months; 38.9% of evaluable cell-type×subregion strata younger.
- B9: OSKM-control median matched-stratum shift -0.555 months; 76.0% of evaluable cell-type×subregion strata younger.

At cell-type resolution, median four-pair shifts ranged from Neuroblast -1.899 months (only three evaluable pairs) and NSC -0.834 (three pairs) to Astrocyte +2.853 and VLMC +2.878 months. All BH-adjusted Wilcoxon q-values were at least 0.750. Four biological pairs are too few for a sensitive confirmatory cell-type test.

## Held-pair prediction and leakage

PCA was refit inside every training fold, so held paired batches never influenced scaling or PC axes. The model was fixed Ridge(alpha=100) on PC1-PC10.

- Raw predicted age: z-CC R2=0.206, Spearman=0.446, MAE=4.194 months; s-CC R2=0.067, Spearman=0.272, MAE=4.558. Spearman was positive in all four held pairs for both representations.
- After within-slice×cell-type×subregion median centering: z-CC R2=0.068, Spearman=0.266; s-CC R2=0.018, Spearman=0.186. Again, all four held-pair Spearman directions were positive.
- Random-cell splits inflated raw R2 by 0.048 for z-CC and 0.081 for s-CC; centered-endpoint inflation was 0.007 and 0.047.
- Sample leakage remains severe in s-CC: PC1 slice eta-squared=0.383, and deterministic k-means(4) sample NMI=0.385. z-CC k-means sample NMI was 0.0056.

Fold-resampling 95% intervals (10,000 resamples of the four held-pair metrics) are in `tables/held_pair_prediction_pair_bootstrap_ci.csv`. They quantify between-pair uncertainty but are necessarily coarse with n=4.

## PC associations with predicted spatial age

Primary correlations are pooled percentile-rank correlations within each OSKM slice. CIs are 500-replicate, slice-stratified 8×8 spatial-block bootstraps. Nominal cell-level P values for all 20 PCs were BH-adjusted together, but inference is based primarily on block CIs and direction consistency because neighboring cells are dependent.

- z_cc PC2: rho=-0.325, spatial-block 95% CI [-0.367, -0.281], directions 0 positive/4 negative slices; anatomy-adjusted rho=-0.211.
- z_cc PC6: rho=0.238, spatial-block 95% CI [0.205, 0.272], directions 4 positive/0 negative slices; anatomy-adjusted rho=0.196.
- s_cc PC6: rho=0.169, spatial-block 95% CI [0.121, 0.215], directions 3 positive/1 negative slices; anatomy-adjusted rho=0.079.
- s_cc PC3: rho=0.165, spatial-block 95% CI [0.124, 0.214], directions 4 positive/0 negative slices; anatomy-adjusted rho=0.159.
- s_cc PC10: rho=-0.154, spatial-block 95% CI [-0.189, -0.117], directions 1 positive/3 negative slices; anatomy-adjusted rho=-0.100.
- z_cc PC5: rho=-0.141, spatial-block 95% CI [-0.185, -0.093], directions 0 positive/4 negative slices; anatomy-adjusted rho=-0.133.

z-CC therefore carries the strongest clock-age axis (PC2). s-CC PC3 is the cleanest salient axis by all-four-slice direction consistency and persistence after cell-type/subregion adjustment. PC signs are arbitrary and should not be interpreted across refits.

## Within-slice s-CC diffusion pseudotime

DPT was fit independently in each OSKM slice after within-slice scaling and PCA(30), with k=30 and fixed seed. The primary root was chosen independently of clock age as the minimum within-slice s-CC PC1 cell; no orientation was flipped to improve correlation.

- Primary root: pooled within-slice rho=-0.142, spatial-block 95% CI [-0.188, -0.094], negative in all four slices. After within-slice×cell-type×subregion rank adjustment, rho=-0.078.
- Opposite-root sensitivity: pooled rho=-0.034, CI [-0.083, 0.013], with only 2/4 slices positive and 2/4 negative.

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
