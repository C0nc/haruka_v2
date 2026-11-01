# GSE250346 lung-fibrosis Haruka analysis

## Design

- Data: corrected GEO GSE250346 Xenium Seurat object, 343 genes.
- Fit: all 1,405,182 pulmonary-fibrosis cells; no holdout or downsampling.
- Reference arm: 531,567 cells from 15 `Less Affected` cores (<75% pathology).
- Condition arm: 873,615 cells from 20 `More Affected` cores (>=75% pathology).
- Spatial graph boundary: tissue core (`slice_id`).
- Decoder-only technical covariate: sequencing `run` (three levels represented in both arms).
- Cell-type neighborhood target: deposited `final_CT` annotation.
- Haruka: shared three-objective decoder, z=s=32, hidden=256, lambda=25, Poisson assay loss, 150 epochs.
- CellCharter: symmetric three-hop aggregation performed separately within every core.
- MOSCOT: full-rank fused GW, alpha=0.5, epsilon=0.05; physical geometry plus background z-CC; alignment only within the five genuinely paired donors. Salient s-CC never defines the alignment.

## Main results

- More-affected core pathology versus mean decoded salient composition-shift magnitude: Spearman rho=0.459, p=0.0417.
- Pathology versus mean s-CC magnitude: rho=0.443, p=0.0506.
- Pathology versus mean background z-CC magnitude: rho=-0.157, p=0.509.
- On more-affected cells, z-CC recovers deposited cellular niches more strongly than s-CC (CNiche NMI 0.314 versus 0.244). This is consistent with z-CC retaining tissue organization.
- s-CC clusters identify response niches enriched for B/plasma cells, neutrophil-subpleural fibroblast-mesothelial structure, airway epithelial states, transitional AT2, and SPP1+/interstitial macrophage states.
- s-CC also contains strong donor-specific spatial heterogeneity after CellCharter. Intrinsic s donor NMI is 0.122, rising to 0.478 after s-CC, while run NMI remains only 0.038. This is spatial donor/core heterogeneity rather than evidence of sequencing-run leakage.
- Across six paired less-to-more core comparisons, MOSCOT alignment improves nearest cross-arm agreement by +0.013 for exact cell type, +0.027 for lineage, +0.069 for CNiche, and +0.039 for TNiche versus independently standardized raw shapes.

## Primary outputs

- `figures/condition_only_zcc_scc_umap.png`: more-affected-cell z-CC and s-CC UMAP audit.
- `figures/pathology_vs_representation_magnitude.png`: direct background versus salient pathology relationship.
- `figures/s_cc_cluster_biology_summary.png`: decoded composition shifts and pathology-associated response clusters.
- `moscot_fullrank/moscot_alignment_*.png`: converged paired-donor spatial alignments with s-CC clusters overlaid.
- `moscot_fullrank/moscot_alignment_agreement_summary.png`: cross-arm alignment agreement summary.
- `tables/`: cell-level assignments, cluster enrichments, decoded shifts, severity tests, and clustering metrics.

The earlier `moscot/` directory is a retained diagnostic low-rank pass that emitted non-convergence warnings. Use only `moscot_fullrank/` for interpretation.
