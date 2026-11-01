# CODEX broad-cell-type annotation and sample composition

Completed: broad segmentation-supported labels and sample/gray-white-matter composition plots. No new Haruka training or astrocyte/oligodendrocyte CODEX pseudotime has been performed.

Labels: GFAP masks = astrocyte; IBA1 masks = myeloid (microglia/macrophage unresolved); OLIG2 masks = oligodendrocyte/OPC (unresolved); NeuN masks = neuron. Special H2A.x-named GFAP files follow the original notebook mapping, not filename guessing. These are segmentation identities, not de novo molecular subtype predictions. Cross-mask overlap has not been independently deduplicated.

Ctl-2 is excluded as the AML control, matching the prior cohort. All seven other donors have GFAP, IBA1 and NeuN segmentation; only AD-1, AD-4 and Ctl-1 have locally available OLIG2 segmentation. Missing OLIG2 is NA, not zero.

## Results

- `codex_broad_celltype_annotations.csv.gz`: per-segmentation-instance label, marker intensity, donor, condition, region and coordinates.
- `segmentation_coverage_missing_is_NA.csv`: modality coverage.
- `common_three_lineages_sample_region_stacked.png`: seven-donor proportions with the SAME three-class denominator.
- `complete_four_lineages_sample_region_stacked.png`: complete four-class subset; three donors only.
- `equal_GM_WM_weight_proportions.csv`: region-standardized descriptive proportions.
- `source_inventory.csv`: source files and hashes.
- `analysis_metadata.json`: reproducibility and limitations.

Whole-section four-class proportions differ: astrocytes 10.9% / 12.1% / 7.0%, and neurons 14.0% / 9.0% / 35.7% in AD-1 / AD-4 / Ctl-1. Equal-weight gray/white-matter standardization changes these values (see CSV), demonstrating anatomical coverage contributes to apparent sample differences. This is descriptive sample composition, not a validated AD effect or temporal cell conversion.

## Previous glial pseudotime plots (STARmap, NOT CODEX)

`/home/yancui/haruka_v2/final_results_nb/ad_shared/trajectory/glia_terminal_analysis/enrichment_and_composition/starmap_within_lineage_dpt_cell_subtype_stacked.png`

The corresponding `shared_salientcc_dpt_cells.csv.gz` contains both `dpt_pseudotime` (replicate x lineage) and `shared_scc_dpt` (replicate-wide). `starmap_within_lineage_ct_sub_composition.csv` contains five-bin within-lineage ct_sub compositions. DPT roots use the plaque-distant decile; pseudotime is inferred state ordering, not measured time. Existing CODEX trajectories remain myeloid-only and cannot be relabeled as astrocyte/oligodendrocyte trajectories.
