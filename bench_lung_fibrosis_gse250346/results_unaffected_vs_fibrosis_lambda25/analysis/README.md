# Corrected GSE250346 Haruka fibrosis analysis

## Design

- Haruka control: all 225,137 cells from 10 `Unaffected` tissue cores.
- Haruka condition: all 1,405,182 cells from 35 pulmonary-fibrosis cores,
  comprising 531,567 `Less Affected` and 873,615 `More Affected` cells.
- `Less Affected` versus `More Affected` is not supplied as a model label.  It
  is used only after fitting to test severity information inside the condition
  arm.
- All three GEO sequencing runs contain control, Less Affected, and More
  Affected cores.  `batch` (GEO run) is injected only into the three shared
  decoders; the spatial graph is bounded by `slice_id` (tissue core).
- Cell composition uses the deposited `final_CT` annotation (47 cell types).
- Final Haruka configuration: one-stage shared assay/composition/neighbor-mean
  decoders, z=s=32, hidden=256, lambda=25 directional z<-s independence,
  Poisson assay objective, 150 epochs.
- z and s use identical three-hop CellCharter aggregation.  The graph has
  9,681,334 edges and zero cross-core edges.

## Main severity result

Severity was evaluated on one centroid per independent fibrosis core, with a
leave-one-donor-out classifier (35 cores, 26 donors):

| representation | ROC AUC | average precision | balanced accuracy |
|---|---:|---:|---:|
| background z-CC | 0.763 | 0.817 | 0.683 |
| salient s-CC | **0.957** | **0.971** | **0.825** |
| decoded salient composition shift | 0.910 | 0.938 | 0.742 |

Across the 35 fibrosis cores, mean response magnitude tracks continuous
pathology for s-CC (Spearman rho=0.724, p=8.82e-7) and decoded composition
shift (rho=0.747, p=2.62e-7), but not z-CC (rho=-0.099, p=0.571).  In the five
paired donors, all five s-CC and decoded-shift differences point upward from
Less to More Affected; with n=5, the exact two-sided Wilcoxon p-value is 0.0625.

These results support the intended decomposition at the severity level: z-CC
retains tissue/background information, while s-CC more strongly represents
the extent of fibrosis within the condition arm.  Donor identity is also
detectable in s-CC, so the donor-held-out result is the primary metric rather
than cell-level clustering.

## Paired transport and initial-state test

Full-rank MOSCOT was fitted for six Less-to-More core pairs across five donors
using physical coordinates plus z-CC as the alignment geometry.  s-CC enters
only through a robust positive target marginal.  All 12 uniform/salient-
weighted fits converged.  Salient weighting most strongly increases the
pooled Mesothelial -> Activated Fibrotic FB association (+0.177 probability)
while decreasing Mesothelial -> Alveolar FB (-0.135).

The stricter initial-state question does **not** pass.  For 854 unique
Less-Affected Alveolar FB landmarks, future fate was defined as transport to
Activated Fibrotic FB versus Alveolar FB.  Donor-held-out AUC on the 428
within-donor extreme-quartile cells is 0.583 for z-CC and 0.555 for s-CC;
intrinsic z/s and decoded shifts are similarly weak (0.520-0.576).  Therefore
the corrected model supports a global severity-response claim, but not a claim
that its initial salient state predicts this specific fibroblast terminal
choice.

The raw initial expression contrast is biologically coherent but exploratory:
POSTN, SPCS2, AXL, SFRP4, CTHRC1, and SLC2A1 are higher in all five donors in
the fibrotic-primed group.  With only five paired donors, the minimum two-sided
p-value is 0.0625 and no gene survives FDR correction.

## Primary outputs

- `figures/less_vs_more_zcc_scc_umap.png`: condition-only z-CC and s-CC UMAP,
  colored by severity.
- `figures/severity_core_level_summary.png`: donor-held-out AUC and continuous
  pathology correlations.
- `figures/condition_only_zcc_scc_umap.png`: annotation/run/response-cluster
  audit.
- `tables/severity_donor_heldout_metrics.csv`: primary independent-core metric.
- `tables/core_pathology_correlations.csv`: continuous severity associations.
- `tables/paired_donor_severity_deltas.csv`: paired-direction sensitivity test.
- `moscot_spatiotemporal_salient_marginal/`: converged paired transport tables
  and heatmaps.
- `initial_state_fate/`: reproducible initial Alveolar-FB fate test and gene
  program heatmap.
