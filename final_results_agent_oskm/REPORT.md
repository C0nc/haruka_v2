# Rigorous full-cohort Haruka analysis of OSKM partial reprogramming

## Bottom line

The completed Haruka run is internally aligned and matches the intended old-control versus old-OSKM design, so no retraining was needed.  The stricter paired analysis does **not** support a uniform, cohort-wide rejuvenation claim.  Three OSKM/control pairs (B9, B10, B13) shift youngward, while B14 shifts in the opposite direction.  No cell type or subregion has a positive primary youngward median in all four pairs, and no cell-type/subregion pair-level test survives multiple-testing correction.

Haruka's salient spatial representation does contain information about the most extreme responders: leave-pair-out q95 responder AUROC is 0.696 for s-CC versus 0.628 for z-CC, and the combined z-CC+s-CC model reaches 0.735.  However, salient does not replace background.  For strictly pair-adjusted continuous response, z-CC has the strongest single-space rank association (Spearman 0.309 versus 0.196 for s-CC), every representation has negative overall R², and B14 is the failure fold.  Thus the strongest defensible conclusion is that salient adds information for extreme-response classification, while background anatomy/state remains important for response magnitude.

Residual sample structure is substantial after spatial aggregation.  Deterministic s-CC clusters have sample NMI 0.397, the fixed-seed s-CC UMAP sample clustering has NMI 0.350, and six of eight s-CC domains are sample-dominated by the prespecified flag.  Domain 5 is the only responder-rich domain that passes the sample-dominance flag; domain 0 is a reproducible responder-poor comparator.  The numerically poorest domain, 7, is entirely B10 and is not interpreted as biology.

## Design and model audit

- Training data: 684,253 cells by 300 genes; integer-valued raw counts, finite two-dimensional coordinates, unique observation and gene names.
- Training contrast: 342,771 old controls versus 341,482 old OSKM cells.
- Matched pairs: B9 (OC903/OT902), B10 (OC1138/OT1125), B13 (OC1083/OT1084), B14 (OC1226/OT1160).
- `graph_id` equals `slice_id` for every cell. Spatial graphs and CC aggregation are slice-bounded. The decoder batch key is also `slice_id`.
- Young reference: 330,924 cells by the same ordered 300 genes across four young-control slices; evaluation only and absent from Haruka fitting.
- Full-model observation names exactly match the H5AD order. Direct z and s are 684,253 by 32.
- Condition-only CC rows exactly equal the 341,482 OSKM rows in source order. z-CC and s-CC are 341,482 by 96, symmetric three-hop, and exclude the focal-cell block.
- Model metadata confirms one-stage composition-gated NB Haruka, z=s=32, hidden width 256, encoder/decoder layers 1/2, 150 epochs, k=20, spatial weight 100, z/s independence 10, decoder-only 8-dimensional slice embeddings with three separate projectors, and seed 0.

The corrected machine-readable design is in `experiment_design.json`; the independent alignment evidence is in `audit/alignment_and_design_audit.json`.

## Response and null definition

All 300 genes were library-normalized and log transformed.  Within each cell type × subregion and paired batch, each OSKM cell was projected onto the direction from its matched old-control expression centroid to the pooled young-control centroid.  A value near 0 is old-control-like and a value near 1 is young-reference-like.  Scores were averaged across a k=15 nearest-neighbor graph bounded by slice.

Matched old-control cells were scored and smoothed identically.  An OSKM cell is a primary responder only if its local score is strictly above the empirical 95th percentile of matched old controls in the same paired batch × cell type × subregion.  Controls define the null and are never labeled biological non-responders.  Of 341,482 OSKM cells, 340,794 used the primary cell-type×subregion direction, 666 used a cell-type fallback, and 22 had no own-cell direction but received a local neighbor score.  All 341,482 have a local score.

The primary call identifies 35,297 responders (10.34%). Sensitivities are coherent but not identical:

| Definition | Responders | Rate | Agreement with primary |
|---|---:|---:|---:|
| local q90 | 59,351 | 17.38% | 92.96% |
| local q95, primary | 35,297 | 10.34% | 100% |
| local q97.5 | 20,928 | 6.13% | 95.79% |
| local q99 | 11,153 | 3.27% | 92.93% |
| unsmoothed q95 | 23,504 | 6.88% | 88.41% |
| variance-scaled q95 | 37,071 | 10.86% | 96.61% |
| leave-pair-out old-direction q95 | 31,431 | 9.20% | 88.84% |

The leave-pair-out direction sensitivity reveals a genuine reference-choice issue: B14's median is -0.236 with its matched-control direction but +0.597 using the other old-control pairs' direction; its matched-null responder rate rises from 0.86% to 7.46%. This does not erase the paired result—it shows that between-control-pair state differences materially affect the definition of “youngward.”

## Paired biological response

| Pair | OSKM cells | Primary responder rate | Median youngward score |
|---|---:|---:|---:|
| B10 / OT1125 | 85,371 | 10.47% | 0.123 |
| B13 / OT1084 | 85,500 | 19.61% | 0.193 |
| B14 / OT1160 | 88,644 | 0.86% | -0.236 |
| B9 / OT902 | 81,967 | 10.78% | 0.118 |

NSCs and neuroblasts have the largest median-of-pair medians (0.340 and 0.213) and high overall responder rates (29.1% and 17.8%), but each is rare (158 and 163 OSKM cells) and negative in B14.  Ependymal cells have a 25.6% responder rate; VSMC has a median score of 0.120.  The most responsive subregions by overall median/rate include VEN (0.171; 24.9%), CTX_L1/MEN (0.121; 14.8%), and CTX_L4/5/6 (0.052; 15.0%).  Again, every one is negative in B14.  These are heterogeneity rankings, not replicated cohort-wide treatment effects.

## What salient adds beyond background

Strict leave-one-paired-batch/OSKM-mouse-out evaluation used deterministic standardized ridge models with a fixed alpha of 100 and LSQR solver.  The fixed alpha avoids outcome-specific hyperparameter leakage and repeated large decompositions.

| Representation | Pair-adjusted response Spearman | Pair-adjusted response R² | q95 responder AUROC | q95 responder AUPRC |
|---|---:|---:|---:|---:|
| z | 0.229 | -0.001 | 0.584 | 0.132 |
| s | 0.165 | -0.001 | 0.609 | 0.152 |
| z-CC | **0.309** | -0.055 | 0.628 | 0.177 |
| s-CC | 0.196 | -0.395 | **0.696** | 0.213 |
| z+s | 0.233 | -0.017 | 0.614 | 0.149 |
| z-CC+s-CC | 0.224 | -0.455 | **0.735** | **0.271** |

For the less stringent anatomy-centered endpoint, s-CC has Spearman 0.310 versus 0.271 for z-CC; for unadjusted local response the combination reaches 0.379.  These rank associations coexist with negative R², meaning calibration across held-out pairs is poor.  B14 is especially problematic: pair-adjusted Spearman is -0.055 for s-CC and -0.078 for z-CC, while its q95 responder prevalence is only 0.86%.  Even though the combined classifier has AUROC 0.656 in B14, s-CC alone is near chance at 0.517.

Direct z and s only modestly distinguish OSKM from control in held-pair tests (AUROC 0.564 and 0.555).  This is not strong empirical evidence that s is a globally condition-specific axis.  The strongest positive evidence for salient is the extreme-responder classification increment; the strongest counterevidence is poor continuous calibration and residual sample structure.

## s-CC domains

Clustering used deterministic MiniBatchKMeans (k=8, n_init=20, seed 0) on a fixed, sample-balanced 50,000-cell fit set and predicted all OSKM cells.  Alternative-seed agreement is moderate (mean ARI 0.504; range 0.407–0.561).

Responder enrichment conditions exactly on paired batch × cell type × subregion margins.  The conditional z statistics use cells, not independent mice, so pair consistency and sample-dominance flags take priority over their very small P values.

- Domain 5 is responder-rich and not sample-flagged: 32,540 cells, 12.69% responders, 1.50× observed/expected after adjustment, mean pair-adjusted response +0.036, maximum slice fraction 0.480, sample entropy 0.815.  Three of four pair-specific rates exceed their pair-wide rate, but domain rates range from 2.9% to 39.9%.
- Domain 0 is the interpretable responder-poor comparator: 32,380 cells, 3.08% responders, 0.39× observed/expected, mean pair-adjusted response -0.058, maximum slice fraction 0.329, sample entropy 0.978, and all four pair-specific rates are below their pair-wide rates.
- Domain 7 is numerically poorest (1.13%; 0.15×; mean adjusted response -0.248) but is 100% B10 and is excluded from biological interpretation.
- Domains 1, 3, 4, 6, and 7 are sample-dominated. Domain 2 also crosses the maximum-slice threshold. Only domains 0 and 5 cleanly pass the prespecified flag.

## Genes, programs, cell types, subregions, and neighborhoods

After centering expression within paired batch × cell type × subregion, domain 5 is enriched for Neuron-Inhibitory cells (log2 enrichment 1.16) and STR_LS/NDB (1.99).  Its strongest positive genes include `Crym`, `Sparc`, `Efnb2`, `Sox11`, `Rims2`, `Cdh2`, `Drd1`, `Dlx6os1`, and `Tac1`; its strongest negative genes include `Gfap`, `Drd2`, `Cd4`, `Ntm`, `Reln`, `Clu`, `Slc1a2`, `Aldoc`, and `Jun`.

Domain 0 is enriched for Ependymal (2.47), Oligodendrocyte (1.26), Microglia (0.67), CC/ACO (2.41), and VEN (2.19).  It has higher `Gfap`, `C4b`, `Dlx1`, `C3`, `Sulf2`, `Sox2`, `B2m`, `Apod`, and `Thbs4`, and lower `Drd1`, `Spock2`, `Efnb2`, `Penk`, `Crym`, `Tac1`, and `Adamts3`.

Across all OSKM cells, responders show higher adjusted `Drd1`, `Sox11`, `Nr4a2`, `Crym`, `Tac1`, and `Spock2`, and lower `Gfap`, `C4b`, `Enpp2`, `Atp1a2`, `Gja1`, `B2m`, `Apoe`, and `Clu`.  The strongest program association is lower astrocyte-reactivity score in responders (-0.201 standardized units; Spearman with local response -0.154).  Interferon/inflammation, microglial-phagolysosomal, and myelin programs are also modestly lower.  Canonical reprogramming/stemness and cell-cycle programs are not elevated.  Because the same 300 genes define response, these molecular associations are descriptive and partly circular; none of the pair-level gene tests survives FDR (minimum FDR about 0.55 with four pairs).

The domain neighborhood tables use 20-nearest-neighbor graphs separately within each OSKM slice.  In domain 5, notable enrichments include Neuroblast→Neuroblast (log2 7.22; 220 edges), Ependymal→Ependymal (6.10), and macrophage↔VLMC/VSMC (3.36–3.81).  Domain 0 has strong Ependymal, VSMC, macrophage, VLMC, excitatory-neuron, and MSN homotypic enrichments.  These describe learned niches but are not independent validation because cell-type composition is a Haruka reconstruction target.

## Spatial Aging Clock

The published frozen clock is applicable at the feature level: every represented cell-type clock has 300/300 genes.  Its official smoother is bounded by mouse.  A redundant full-cell rerun was stopped after the first cell type took about an hour, implying a many-hour run.  The existing completed 160,000-cell cache was instead mapped one-to-one into the full cohort, yielding 80,000 OSKM evaluation cells (20,000 per pair).  Cross-version unpickling emits an sklearn 1.2.2→1.7.2 warning; all 18 serialized pipelines produce exactly the same values as a manual scaler/coefficient calculation in the current runtime, but sklearn does not guarantee cross-version compatibility.

Haruka and clock scores have modest concordance: raw Spearman 0.127 and pair×cell-type×subregion-adjusted Spearman 0.169.  Adjusted correlations are positive in all pairs but range from 0.043 in B14 to 0.295 in B9.  Haruka's score predicts a q95 clock responder with AUROC 0.565/AUPRC 0.072, and q95 responder Jaccard is only 0.064.  These endpoints share ordering information but do not identify the same cells.

The clock itself does not support broad rejuvenation: no cell type is younger in all four pairs and no cell-type shift survives FDR.  Ependymal has a slightly younger median shift (-0.42 months) but inconsistent pairs; most cell types have positive OSKM-minus-control predicted ages.  For pair-adjusted clock heterogeneity, direct s is the best tested representation (R² 0.114, Spearman 0.322), slightly above direct z (0.082, 0.292); z-CC and s-CC are weaker (0.062/0.263 and 0.036/0.222).  Thus direct salient adds some independently clock-aligned signal, while spatial s-CC aggregation does not improve the clock endpoint.

## Claims and limitations

Supported:

1. Response is heterogeneous across cells, anatomy, and matched pairs; B14 is qualitatively discordant under the primary paired definition.
2. s-CC adds information for classifying extreme expression-defined responders, especially when combined with z-CC.
3. Background z-CC is at least as important as salient for continuous response ordering after strict pair adjustment.
4. Domain 5 versus domain 0 is the most defensible responder-rich/poor s-CC contrast after sample-dominance filtering.
5. Clock concordance is modest and positive after adjustment, but cell-level responder identity is largely non-overlapping.

Not supported:

1. A uniform OSKM rejuvenation effect across all four pairs.
2. A calibrated salient-only continuous predictor that generalizes to every mouse.
3. Biological interpretation of sample-dominated domains, especially domain 7.
4. Independent validation of genes/programs or cell-type neighborhoods from this targeted panel.
5. Temporal lineage, causal trajectories, or causal cell-cell effects from UMAP/static spatial data.

The treatment is perfectly confounded with slice identity, decoder conditioning cannot guarantee batch-free latents, there are only four matched pairs, and all pair-level inference is low resolution.  UMAPs are descriptive.  No trajectory or transport claim was made.

## Artifact map

- `experiment_design.json`: corrected design and deterministic analysis contract.
- `audit/`: row/feature alignment, clock compatibility, runtime versions, and hashes.
- `tables/oskm_cell_scores_and_domains.csv.gz`: per-OSKM-cell scores, responder sensitivities, domains, and programs.
- `tables/representation_heldout_metrics.csv`: main representation comparison.
- `tables/representation_heldout_metrics_by_pair.csv`: held-out fold diagnostics.
- `tables/scc_domain_response_adjusted.csv`: responder-rich/poor domains and sample flags.
- `tables/scc_domain_genes_pair_anatomy_adjusted.csv`: adjusted domain genes.
- `tables/responder_genes_pair_anatomy_adjusted.csv`: adjusted responder genes.
- `domain_annotation/`: sample-bounded cell-type and cell-cell neighborhood enrichments.
- `clock/`: copied source cache, mapped cell table, held-out metrics, and paired clock summary.
- `figures/`: fixed-seed OSKM-only UMAPs and response/clock summaries.
- `scripts/` and `exact_commands.sh`: complete reproducible analysis code and invocation.
