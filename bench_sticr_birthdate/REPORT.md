# STICR P4 forebrain: early- vs late-born design check (2026-10-17)

Nothing trained. Built `sticr_opc_isocortex.h5ad` only.

**Verdict:** neuronal early-vs-late is NOT identifiable (birth order = layer/anatomy and maturity). Glial (OPC) design is only weakly identifiable: late OPCs sit near white matter.

## Data (raw/, ~14 GB)
- geo/GSE341480_p4_forebrain_lineage_metadata.rds (+ geo_metadata.tsv): 416,390 cells, clone IDs, injection age, region, clusters; 288,826 clones (59,761 multicellular).
- ucsc_rna/: meta.tsv + counts mtx (6.2 GB).
- ucsc_merfish/: meta.tsv (1.6 GB) + mtx (6.1 GB, 500 genes + 50 blanks); 9.27M cells, x/y, 79 sections (AP_order), CCF region, paper cluster label per cell (no confidence). Allen ABC has no developing MERSCOPE set; UCSC is the source.
- Label region prefix = dissection (not recoverable from expression); used the cell's CCF position instead (changes 8% of arm calls).

## Birthdate (multicellular clones, normalised per injection-age total within region; fraction labelled >= E16)
- Cortex excitatory inside-out: L5/6 0.01-0.03, L4 0.04, L2/3 0.29, Immature 0.62 (late = also less mature).
- Cortex inhibitory: MGE/CGE 0.00-0.11, Meis2_Sp8 0.47. OB inhibitory 0.04-0.12 vs 0.42.
- Cortex OPC: clusters 4/8 0.16/0.12 vs 5/16 0.39/0.52 (matches paper).

## Anatomy confound (per-section medians; AUC arm from coords / neighbour composition)
| Design | Mixing | AUC coords / neighbours |
|---|---|---|
| Excitatory, Isocortex | 0.70 | 0.94 / 0.68 |
| Inhibitory, OB | 0.24 | 0.95 / 0.83 |
| Astrocytes, HIP+RHP | 0.80 | 0.72 / 0.68 |
| OPC 4/8 vs 5/16, Isocortex | 0.86 | 0.75 / 0.70 |
| Same, white-matter-distance matched | - | 0.64 / 0.63 |
| OPC pooled with fibre tracts | 0.69 | 0.82 / 0.80 |

Late OPCs 115 um from white matter vs 385 um for early.
Own MapMyCells run vs paper labels: 74% cluster / 87% broad class; cortical OPC arm agreement 64% early / 72% late; cluster 16 recovered 82%.

## Recommended design (not fitted)
- OPCs, Isocortex only: control = OPC_4 + OPC_8; disease = OPC_5 + OPC_16.
- Batch key = section (35 sections, one brain -> pseudo-replicates).
- 23,886 vs 3,893 cells; robustness subsets: mapping consensus (14.3k / 2.6k), white-matter matched (9.5k / 3.9k).
- Confounds: white-matter distance (neighbour-only baseline AUC 0.70); labels inferred from own cluster (no barcodes in MERFISH); labelling time != birthdate; ~30% arm-level label noise.

## Files
`sticr_opc_isocortex.h5ad` (1,086,186 x 500): layers['counts'], obsm['spatial'], condition (control/disease/halo), cell_type (broad), section, consensus, wm_matched, mapping columns. Halo = each arm cell's 20 NN + other extracted cells (not full tissue). Tables in results/.
