# Training-time paired-slot separation — 2026-08-02

> **Capacity update.** The staged/private design remains, but hidden width 256 now
> replaces 128 as the default. It scores 0.546 ± 0.024 raw versus CINEMA-OT 0.492.
> See `CAPACITY_TRAINING_ABLATION_2026-08-02.md`. The 0.499 result below is the
> width-128 precursor retained for provenance.

## Outcome

The benchmark now has a raw-only primary contract. The best current Haruka is a
pure-GAT two-stage model:

- 150 epochs total, with MAE corruption for epochs 0–89.
- At epoch 90, a differentiable ridge-predictability penalty begins and ramps to
  weight 1 over 10 epochs.
- The background encoder has 32 internal dimensions. Its trailing 8 dimensions
  are the GAT source and remain internal; `get_latent_representation("background")`
  directly returns the 24-dimensional private intrinsic head.
- No post-hoc ridge residualization or other cleanup is applied at evaluation.
- Raw CINEMA-OT is the sole primary CINEMA comparison.

| Method | assignment margin |
|---|---:|
| **Haruka staged pure GAT, raw** | **0.498793 ± 0.007678** |
| CINEMA-OT, raw | 0.491515 |
| contrastiveVI | 0.452 |
| CellCharter | 0.221 |
| PCA | 0.208 |

Haruka's mean lead over raw CINEMA-OT is 0.007277. This is narrow.

## Seed-level validation

| model seed | raw diagonal | raw off-diagonal | raw margin | lead vs CINEMA raw |
|---:|---:|---:|---:|---:|
| 0 | 0.748124 | 0.250286 | 0.497838 | +0.006323 |
| 1 | 0.734065 | 0.227162 | 0.506903 | +0.015388 |
| 2 | 0.714769 | 0.223132 | 0.491636 | +0.000121 |

All three model seeds exceed raw CINEMA-OT on this fixed dataset, but seed 2 is
effectively tied. This supports a fixed-dataset result, not a broad superiority
claim. Independent simulator datasets remain required.

## Why the original regularizer was insufficient

The GAT reads the trailing eight dimensions of the background posterior. Scoring
all 32 dimensions as "background" therefore included the GAT's spatial carrier by
construction. A loss that tried to make the full background latent independent of
the GAT output was fighting an architectural identity: stronger weights degraded
the intended factors without removing that carrier.

The private-head contract fixes the definition rather than transforming an output:
spatial carrier dimensions remain available internally to the decoder and GAT,
while the model API returns only the background dimensions intended to represent
intrinsic variation. The staged loss then discourages residual shared linear
subspace after MAE has learned stable signal.

## Important limitation

CINEMA-OT still recovers the treated salient-spatial domain much more strongly
(about 0.591 versus Haruka's 0.209 mean). Haruka wins the aggregate four-factor
assignment through lower overall leakage; it does not win every quadrant.

## Artifacts

- Raw model outputs: `results_stage_private_raw/`
- Per-seed score files: `results_stage_private_raw/metrics_seed{0,1,2}.jsonl`
- Compact table: `results_stage_private_raw/summary.csv`
- Simultaneous-loss controls: `results_paired_reg_screen_raw/`
- Staged weight screen: `results_paired_reg_stage_screen_raw/`
