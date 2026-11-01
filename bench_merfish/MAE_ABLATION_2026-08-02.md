# Haruka pure-GAT MAE ablation — 2026-08-02

> **Later raw-only contract.** This report establishes the MAE-90 choice, but its
> cleaned comparison is no longer the primary benchmark. The current default adds
> a post-MAE training-time predictability loss and exports a private intrinsic
> background head. It scores raw Haruka at 0.499 ± 0.008 versus raw CINEMA-OT at
> 0.492. See `TRAINING_TIME_ORTHOGONALIZATION_2026-08-02.md`.

## Decision

Masked/denoising reconstruction helps Haruka materially. The corrected benchmark
default is now `mae_epochs=90` within 150 total epochs. MAE 60 remains a reasonable
trade-off when salient-spatial domain recovery is prioritized over the primary
four-factor assignment margin.

## Valid experimental contract

- Fixed 72,200-cell v5 dataset: `simv4_g0.8_r0.3_p0.15_d1_s0.h5ad`.
- Pure GAT only (`cov_gene_num=0`, `me_weight=0`); raw count training layer.
- Matched decoder batch `timepoint`; k=8; salient/spatial width 8; 150 epochs.
- MAE epochs 0, 30, 60, and 90; `mae_epochs=0` is the true disabled control.
- Explicit model seeds 0, 1, and 2; fixed scorer seed 0.
- Raw and paired-slot-cleaned representations were emitted from each single fit,
  so cleanup comparisons do not contain retraining noise.
- Primary score is leave-slice-out dependence assignment margin. ARI and oracle
  were skipped because they do not affect this ablation's primary statistic.

## Seeding bug found and fixed

`HarukaSpatial.setup_model(seed=...)` previously accepted the RNA model seed but
did not apply it. Only multiomics and clustering paths called `seed_everything`,
and scvi's data splitter retained its independent global seed. Sequential variants
therefore inherited RNG state, while result directory names falsely implied the
requested seed had been used.

The setup path now resets Lightning/Python RNGs and `scvi.settings.seed`, records
`model_seed` and `rng_reset_at_setup` in metadata, and has a regression test. A
one-epoch diagnostic produced byte-identical arrays for two seed-7 fits in all four
slots and different arrays for seed 8. Results under `results_mae_ablation_seeded_*`
are valid; the earlier unseeded preliminary MAE roots are not evidence.

## Primary results

Mean ± sample SD across explicit model seeds:

| MAE epochs | raw diagonal | raw off-diagonal | raw margin | cleaned diagonal | cleaned off-diagonal | cleaned margin |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.691 ± 0.027 | 0.364 ± 0.015 | 0.326 ± 0.032 | 0.700 ± 0.029 | 0.217 ± 0.011 | 0.483 ± 0.039 |
| 30 | 0.715 ± 0.024 | 0.329 ± 0.016 | 0.386 ± 0.033 | 0.732 ± 0.020 | 0.196 ± 0.006 | 0.537 ± 0.022 |
| 60 | 0.724 ± 0.017 | 0.320 ± 0.007 | 0.404 ± 0.013 | 0.742 ± 0.016 | **0.187 ± 0.008** | 0.555 ± 0.011 |
| **90** | **0.733 ± 0.012** | **0.310 ± 0.032** | **0.423 ± 0.025** | **0.746 ± 0.013** | 0.189 ± 0.017 | **0.557 ± 0.010** |

Cleaned per-seed margins:

| model seed | MAE 0 | MAE 30 | MAE 60 | MAE 90 |
|---:|---:|---:|---:|---:|
| 0 | 0.483 | 0.556 | **0.563** | 0.549 |
| 1 | 0.521 | 0.541 | 0.543 | **0.553** |
| 2 | 0.444 | 0.512 | 0.558 | **0.568** |

Every nonzero MAE duration improved the margin over MAE 0 for every seed. Mean
paired cleaned gains over MAE 0 were +0.054 (30), +0.072 (60), and +0.074 (90).
The 90-vs-60 mean difference is only +0.002, so these settings are effectively a
plateau on three seeds rather than a strong claim that 90 intrinsically dominates.

## What MAE changes

On cleaned representations, increasing MAE 0 → 90 changed mean intended recovery:

- salient → intrinsic state: 0.599 → 0.747;
- spatial → shared niche: 0.994 → 1.000;
- salient-spatial → treated domain: 0.206 → 0.236.

MAE 60 has the best salient-spatial treated-domain recovery (0.257 versus 0.236 at
90) and slightly lower spatial → intrinsic leakage (0.058 versus 0.067). MAE 90
wins the aggregate score because its salient-intrinsic gain is larger than that
trade-off.

## CINEMA-OT comparison

The identical label-free cleanup applied to CINEMA-OT gives margin 0.545221
(diagonal 0.692283, off-diagonal 0.147061). This is reproduced by combining the
validated cleaned background/salient-spatial rows with CINEMA-OT's unchanged
salient/spatial rows. Raw CINEMA-OT is 0.491515.

- MAE 0 cleaned Haruka (0.483 mean) does not beat cleaned CINEMA-OT.
- MAE 60 cleaned Haruka averages 0.555, but seed 1 is slightly below CINEMA-OT.
- MAE 90 cleaned Haruka averages 0.557 and all three seeds exceed CINEMA-OT
  (0.549, 0.553, 0.568).
- Raw Haruka remains below raw CINEMA-OT even at MAE 90 (0.423 mean versus 0.492),
  so the paired-slot cleanup remains essential to the current win.

This is still one fixed simulator dataset. Three model seeds establish a paired
training effect but do not replace independent, reproducibly rebuilt simulator
seeds and rerunning every baseline on those datasets.

## Artifacts

- Cleaned embeddings and scores: `results_mae_ablation_seeded_ortho/`
- Raw embeddings and scores: `results_mae_ablation_seeded_raw/`
- Per-run fixed-scorer table: `results_mae_ablation_seeded_fixedscore_summary.csv`
- Aggregate fixed-scorer table: `results_mae_ablation_seeded_fixedscore_aggregate.csv`
- CINEMA merged fairness row: `results_mae_ablation_seeded_ortho/cinema_comparison_merged.json`
