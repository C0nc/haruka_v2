# Haruka capacity and training-length ablation — 2026-08-02

## Decision

Increase network hidden width, not latent width or training length. The current
raw-only default is:

- pure GAT, hidden width 256;
- internal background width 32 with an exported 24-dimensional private head;
- salient/spatial widths 8;
- 150 total epochs: MAE through epoch 89, separation stage from epoch 90;
- no post-hoc cleanup.

This model has 505,218 trainable parameters versus 253,570 at hidden width 128
(1.99x). The exported representation dimensions are unchanged.

## Controlled seed-0 screen

| configuration | diagonal | off-diagonal | raw margin |
|---|---:|---:|---:|
| hidden 128, latent 32/8, 150 epochs | 0.748124 | 0.250286 | 0.497838 |
| hidden 128, latent 32/8, 250 epochs | 0.750980 | 0.286502 | 0.464477 |
| hidden 128, latent 64/16, 150 epochs | 0.769302 | 0.347645 | 0.421657 |
| hidden 128, latent 64/16, 250 epochs | 0.775874 | 0.422029 | 0.353845 |
| **hidden 256, latent 32/8, 150 epochs** | 0.727412 | **0.163112** | **0.564300** |
| hidden 256, latent 32/8, 250 epochs | 0.756935 | 0.224684 | 0.532251 |

Larger latent spaces and longer training raise or preserve the intended diagonal,
but raise leakage even more. Hidden width 256 improves the objective by learning a
cleaner nonlinear mapping while keeping bottleneck sizes fixed.

## Three-seed validation of hidden 256 / 150 epochs

| model seed | raw diagonal | raw off-diagonal | raw margin | lead vs CINEMA raw |
|---:|---:|---:|---:|---:|
| 0 | 0.727412 | 0.163112 | 0.564300 | +0.072784 |
| 1 | 0.724490 | 0.169196 | 0.555294 | +0.063779 |
| 2 | 0.721657 | 0.203296 | 0.518361 | +0.026846 |

Mean ± sample SD:

- diagonal: 0.724520 ± 0.002877
- off-diagonal: 0.178535 ± 0.021659
- assignment margin: **0.545985 ± 0.024343**
- raw CINEMA-OT: 0.491515
- mean Haruka lead: **0.054470**

All three Haruka seeds exceed raw CINEMA-OT. The weakest lead is 0.026846, much
larger than the previous width-128 model's weakest lead of 0.000121.

## What improved

The diagonal is slightly lower than the width-128 model (0.725 versus 0.732), but
off-diagonal leakage falls sharply (0.179 versus 0.234). Background-to-niche leak
falls from 0.295 to 0.034. The gain is therefore better
factor assignment, not a metric win from larger exported embeddings.

Treated salient-spatial recovery improves from 0.209 to 0.260,
but CINEMA-OT remains much stronger on that individual quadrant (~0.591). Haruka's
claim remains aggregate four-factor assignment, not dominance on every component.

## Scope

This is a three-model-seed validation on one fixed simulator dataset. Independent
dataset seeds remain necessary before claiming broad superiority.

## Artifacts

- Screen scores and table: `results_capacity_screen_raw/`
- Hidden-256 validation outputs and table: `results_hidden256_raw/`
- Compact screen CSV: `results_capacity_screen_raw/summary.csv`
- Compact seed CSV: `results_hidden256_raw/summary.csv`
