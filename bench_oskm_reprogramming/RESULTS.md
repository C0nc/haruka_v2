# OSKM partial-reprogramming analysis

## Design

- Source: Zenodo record 13883177, `reprogramming.h5ad` (MD5 verified).
- Control: `old_notreatment`; condition: `old_treatment` (OSKM).
- Four paired slide batches: B9, B10, B13, B14, each represented in both arms.
- Training subset: 20,000 cells from every old mouse, stratified by cell type x
  subregion (80,000 control + 80,000 OSKM; all eight mice included).
- `young_notreatment` is excluded from training and used only as a 20,000-cell
  external youngward-response reference.
- Spatial graphs are bounded by `mouse_id`; paired `slide_id` is injected into
  the decoder. There is no holdout and no cell-type/region supervision.

## Haruka configuration

- 32-dimensional z and s; hidden width 256; Poisson likelihood.
- Contrastive reference 75 epochs; joint spatial fit 150 epochs.
- Wasserstein penalty 0.2; z/s independence lambda 2; decoder-only paired-batch
  injection; batch size 2,048.
- Default new CellCharter readout: z-CC = all cells, three hops without focal
  block; s-CC = within cell lineage, one hop plus focal block, control s zero.

## Main result

The expression-based youngward evaluation qualitatively recovers positive OSKM
shifts in NSCs (median 0.119) and neuroblasts (median 0.109), consistent with the
published direction. However, this Haruka fit does **not** assign that responder
heterogeneity to the salient representation:

| Readout | Leave-one-OSKM-mouse-out response R2 | Spearman |
|---|---:|---:|
| z-CC | 0.387 | 0.575 |
| s-CC | -0.152 | -0.019 |

Background z-CC recovers brain subregions (ARI 0.446; NMI 0.546). Salient s-CC
does not recover response or anatomy and instead shows paired-slide structure
(slide NMI 0.452). Raw s has much lower slide NMI (0.075), showing that the
within-lineage spatial aggregation amplifies weak sample shifts; nevertheless,
raw s also fails the response prediction (R2 -0.144), so the underlying issue is
not only CellCharter. Raw s distinguishes OSKM from control only modestly across
held-out paired slides (AUROC 0.596; raw z 0.542).

Therefore the expression result is biologically suggestive, but the current
s/s-CC clusters should **not** be presented as validated OSKM response states.
The next model iteration should use lineage-balanced target sampling or loss
weights and an explicitly paired, sample-invariant salient objective during
training rather than post-fit centering.

## Outputs

- Prepared data: `data/oskm_old_matched_160k.h5ad`
- Young reference: `data/young_reference_20k.h5ad`
- Haruka embeddings: `results/oskm_reprogramming/haruka_mender_seed0/`
- New z-CC/s-CC: `results/oskm_reprogramming/haruka_newcc_lambda2/`
- Metrics and tables: `results/oskm_reprogramming/haruka_newcc_lambda2_analysis/`
- Condition-only UMAP: `figures/oskm_condition_only_zcc_scc_umap.png`
- Condition-only tissue maps: `figures/oskm_condition_spatial_maps.png`
- Response summary: `figures/oskm_salient_response_summary.png`
