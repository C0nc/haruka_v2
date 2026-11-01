# Haruka CODEX-CNS AD plaque analysis (2026)

This isolated project applies Haruka to the continuous single-cell CODEX-CNS
proteomics from Sanchez-Molina *et al.*, Nature Neuroscience (2026). It does
not edit or reuse outputs from `bench_real_discovery`'s earlier AD analysis.

## Official sources

- Paper: https://www.nature.com/articles/s41593-026-02267-3
- Official code and analysis-ready cell tables:
  https://github.com/BaharehAjami/CODEX-CNS
- Main image records: https://zenodo.org/records/20725083,
  https://zenodo.org/records/20706336, and
  https://zenodo.org/records/20726163
- Main segmentation/plaque-mask records:
  https://zenodo.org/records/21342581,
  https://zenodo.org/records/21347021,
  https://zenodo.org/records/21388042,
  https://zenodo.org/records/21417215, and
  https://zenodo.org/records/21419095

`external/CODEX-CNS` is a shallow clone of the official repository. Its
`figure7_mg_raw.h5ad` contains 48,146 IBA1+ cells from four AD and four control
frontal-cortex donors, ten continuous protein intensities, xy centroids,
GM/WM, and distance to the nearest plaque. `download_official_data.sh` is a
resumable, checksum-verified download of all eight main QPTIFFs and dense and
diffuse plaque-instance masks for all four AD samples.

## Design and preprocessing

The primary analysis excludes `Ctl-2` (sample 3628), identified in the paper
repository notebook as the AML control. This avoids using a systemic myeloid
malignancy as a background reference for an immune-protein contrast. A full
eight-donor sensitivity H5AD is also produced.

The matrix is fluorescence intensity, not sequencing counts. Preprocessing is
the paper's `log1p -> ComBat(donor) -> feature z-score, max_value=10` workflow.
Haruka's protein adapter then clips z-scores to [-5,5], shifts/scales to [0,2],
and uses exact per-cell summed squared error. No NB, ZINB, Poisson, library-size
count model, or RNA likelihood is used.

Plaque labels are never supplied to Haruka. They are evaluation-only. Plaque
recovery is tested only among AD gray-matter myeloid cells by leave-one-donor-
out prediction at overlap, 5-um, and 30-um thresholds, plus a continuous
distance-decay score. Paper initial Leiden and GM/WM labels provide annotation
readouts. CellCharter builds identical three-hop Delaunay readouts for z and s,
bounded by donor and condition; control s is zeroed before aggregation.

## Commands

```bash
conda run -n haruka2 python prepare_codex_cns.py
conda run -n haruka2 python ../bench_be_protein/run_haruka.py \
  --data data/processed/codex_cns_myeloid_primary_no_aml.h5ad \
  --out_root results --batch_key donor_id --sample_key donor_id
conda run -n haruka2 python ../bench_real_discovery/aggregate_symmetric_condition_cellcharter.py \
  --data data/processed/codex_cns_myeloid_primary_no_aml.h5ad \
  --input-emb results/codex_cns_myeloid_primary_no_aml/haruka_mender_mse_seed0 \
  --out results/codex_cns_myeloid_primary_no_aml/haruka_cellcharter_cc3_seed0 \
  --sample-key donor_id
conda run -n haruka2 python evaluate_readouts.py \
  --data data/processed/codex_cns_myeloid_primary_no_aml.h5ad \
  --emb results/codex_cns_myeloid_primary_no_aml/haruka_cellcharter_cc3_seed0 \
  --out results/codex_cns_myeloid_primary_no_aml/haruka_cellcharter_cc3_seed0/evaluation
```
