# Real PEtracer Haruka design audit

Status: **blocked_no_real_control**. No Haruka model was trained, and no responder biology was inferred.

## Decision

The original files are targeted MERFISH RNA from tumor-bearing mice and contain physical sections, tumor labels, cell annotations, and lineage/phylogeny fields, but no untreated, vehicle, sham, or otherwise explicit binary control/condition arm. The requested control-versus-condition Haruka analysis is therefore not identifiable from these files.

The paper methods also rule out treating mouse names as conditions. M1 was injected on day 12 after PEmax induction and sacrificed 39 days post-injection; M2 and M3 were injected on day 6 and sacrificed at 28 and 35 days. M1 also used a different recipient genotype. A M2-versus-M3 time comparison has one mouse per time point, so time is perfectly confounded with animal, and the feature panels differ.

## Verified source contract

- M1: 262,062 cells, 124 features, sections M1-1 through M1-4.
- M2: 106,660 cells, 124 features, one physical section M2-1 containing six tumor labels.
- M3: 145,954 cells, 175 features, sections M3-1 through M3-4 from one tumor.
- Modality: targeted MERFISH RNA; raw integer counts are in `layers/counts`; dense `X` is scaled and is not the NB input.
- Spatial coordinates: `obsm/spatial`.
- Physical-section/graph boundary and prospective decoder batch key: `obs['sample']`.
- Cell-type key: `obs['cell_subtype']`.
- No condition key or control value exists.
- M1 and M2 share all 124 features. M3 shares 110 features with them and has a different 175-feature panel.

## Existing-code audit

The older `bench_petracer_lineage` paired workflow is a simulation benchmark, not a real biological comparison. `build_paired_lineage.py` copies observed cells into two arms and synthesizes lineage-keyed expression in the condition arm; downstream scripts depend on exact `pair_id`, simulation metadata, and `gt_*` fields. Those artifacts were excluded from biological claims.

## Closest valid next steps

1. Preferred: add replicated untreated/vehicle or explicit perturbation-control animals with compatible panels, comparable endpoints, and multiple sections per animal. The requested Haruka/responder analysis would then be identifiable.
2. Valid now: perform a single-condition descriptive lineage/spatial analysis on the original data, using lineage only after unsupervised embedding and making no responder or treatment-effect claims. This is not a control-versus-condition Haruka task.
3. Exploratory only: compare M2 (28 d) with M3 (35 d) on the 110 common genes, explicitly treating it as an unreplicated animal-confounded pilot. It cannot support generalizable response claims.

## Artifact index

- `experiment_design.json`: machine-readable decision, verified keys, unrun requested configuration, and alternatives.
- `claims_limitations.md`: concise claims and non-claims.
- `tables/source_files.csv`: assay and file-level audit.
- `tables/sample_inventory.csv`: section counts and paper-derived experimental timing.
- `tables/feature_panel_overlap.csv`: exact panel overlaps.
- `tables/condition_candidate_audit.csv`: every evaluated proxy contrast and rejection reason.
- `tables/code_assumption_audit.csv`: simulation-dependency audit of existing scripts.
- `tables/sample_by_tumor.csv` and `tables/sample_by_cell_type.csv`: observed composition.
- `figures/design_audit.png`: section sizes and mouse-specific timeline.
- `hashes/source_sha256.txt` and `hashes/code_sha256.txt`: provenance.
- `commands/exact_commands.sh`: reproducible inspection/package commands; explicitly records that training was not run.
- `logs/metadata_audit.log`: concise inspection log.

## Primary context

- Koblan et al., *High-resolution spatial mapping of cell state and lineage dynamics in vivo with PEtracer*, Science (2025): https://doi.org/10.1126/science.adx3800
- Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC12766569/
- Authors' code and data map: https://github.com/jweissmanlab/PEtracer-2025
