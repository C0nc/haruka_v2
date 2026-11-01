# Haruka MERFISH benchmark audit — 2026-08-02

> **Current capacity update.** Doubling encoder/decoder hidden width from 128 to
> 256, while keeping latent widths and the 150-epoch schedule fixed, raises raw
> end-to-end Haruka to 0.546 ± 0.024 (0.564/0.555/0.518) versus raw CINEMA-OT
> 0.4915. Longer 250-epoch training and doubled latent widths both hurt. See
> `CAPACITY_TRAINING_ABLATION_2026-08-02.md`. This supersedes older headline
> values below.

> **Current raw-only superseding result.** Cleanup is now internalized as a second
> training stage and the public background head excludes internal GAT-carrier
> dimensions. Raw end-to-end Haruka scores 0.499 ± 0.008 across model seeds
> 0/1/2 (0.498/0.507/0.492), versus raw CINEMA-OT 0.4915. No post-hoc transform is
> used for either primary result. See
> `TRAINING_TIME_ORTHOGONALIZATION_2026-08-02.md`. This supersedes cleaned-primary
> claims below; those remain diagnostic history.

> **Superseding RNG/MAE correction.** The RNA `setup_model(seed=...)` argument was
> accepted but not applied, so the old “model-seed stability” table below does not
> demonstrate explicit model-seed stability and its 0.584 mean must not be used.
> The RNG path is now fixed and byte-repeatability checked. A corrected pure-GAT
> 0/30/60/90 MAE sweep across explicit model seeds selects MAE 90: cleaned margin
> 0.557 ± 0.010 (per-seed 0.549/0.553/0.568), versus identically cleaned CINEMA-OT
> 0.545. Raw MAE-90 Haruka remains below raw CINEMA-OT. See
> `MAE_ABLATION_2026-08-02.md`; this correction supersedes conflicting claims below.

## Validity decision

Haruka results produced before this audit with `cov_gene_num > 0` are
provisional. Baseline results are reusable because their runners already train
from the raw-count layer. The current benchmark runner is pure GAT only: it fixes
`cov_gene_num=0`, `me_weight=0`, does not expose COVET CLI options, and does not
load `scenvi`. Valid current runs live under `results_puregat*/` and must contain:

```json
{
  "training_layer": "counts",
  "count_contract_validated": true,
  "pure_gat": true,
  "covet_enabled": false
}
```

## Root causes found

1. **Count-data contract violation.** `run_haruka.py` kept raw counts in
   `layers["counts"]` and log-normalised `X`, but `construct_cov()` rebuilt the
   augmented object from `X` and silently dropped the count layer. The Poisson
   likelihood therefore trained on fractional log expression. The augmented
   object now preserves raw gene counts, appends the same COVET target block,
   and refuses setup if raw counts were expected but lost.
2. **Incorrect library size.** The inference path summed genes and appended
   continuous COVET targets when computing the library. It now uses only
   `x[:, :n_input]`.
3. **Condition-confounded decoder batch.** Every `slice_id` level is exclusively
   control or disease (`T1_control`, `T1_disease`, and so on), so the old
   `batch_key="slice_id"` encoded the treatment label itself. Every `timepoint`
   level contains both conditions and is the actual matched-pair nuisance used by
   CINEMA-OT. The runner now defaults to `timepoint` and rejects a batch key with
   single-condition levels unless an explicit diagnostic override is passed.
4. **Symmetric `both` branch is not identifiable enough.** The salient spatial
   branch can reproduce background anatomy unless it is zeroed on controls, and
   a wider salient branch copies more factors. The current tree already contains
   the control-zeroing fix and pair penalties; the low-width `both_sal8` ablation
   is the most promising leakage/recall compromise. These changes reduce routes
   for leakage but do not guarantee factor ownership.
5. **Non-reproducible simulator seed.** `build_sim_v4.py` used Python's salted
   built-in `hash((seed, timepoint, condition))`; rebuilding the same advertised
   seed in another process changed cells and counts. It now derives slice seeds
   with BLAKE2b and records `rng_seed_scheme="blake2b-v1"`. Existing v5 files are
   internally comparable, but a rebuilt dataset must not be mixed with their old
   baseline outputs.
6. **Stale truth-type contract.** The v4/v5 simulator writes four categorical
   states and no `niche_score` or `sal_domain_score` columns. The scorer already
   fell back to categorical grouped macro AUROC, so stored numbers are valid; its
   constants, comments, and simulator metadata now state that contract explicitly.
7. **Dataset and metric failures in earlier benchmark versions.** Anatomy was
   initially predictive of the implanted domain, a receiver's niche leaked into
   its own row, ligand strength did not scale with effect size, PCA-50 removed
   the weak niche factor, and a binary middle-tercile target produced a U-shaped
   problem. The current builder uses stratified domains, rho=0, scaled ligand
   marks, raw ridge probes, macro one-vs-rest AUROC, and leave-slice-out scoring.

## Paired causal check of the count fix

Same 36,100-cell dataset, seed, `both_mae60` configuration, 150 epochs and
scorer; only the count contract/library fix changes.

| metric | old log input | corrected counts | change |
|---|---:|---:|---:|
| diagonal dependence | 0.670 | 0.708 | +0.038 |
| off-diagonal dependence | 0.398 | 0.432 | +0.035 |
| assignment margin | 0.272 | 0.276 | +0.004 |
| salient → salient intrinsic | 0.681 | 0.819 | +0.138 |
| background architecture ARI | 0.515 | 0.694 | +0.179 |
| spatial niche ARI | 0.962 | 0.993 | +0.030 |
| niche-in-intrinsic leakage | 0.337 | 0.292 | -0.045 |

The contract bug materially suppressed real signal, but correcting it also makes
several off-diagonal factors easier to recover. The nearly unchanged assignment
margin shows that the remaining limitation is factor identifiability/leakage,
not simply underfitting caused by log input.

## Full 72,200-cell comparison (seed 0)

The pure-GAT `both_sal8_timebatch` run used the exact v5 dataset and scorer as the
stored baselines. Pre-fix Haruka rows are omitted from the valid ranking.

| method | diagonal | off-diagonal | margin | excess off-diagonal |
|---|---:|---:|---:|---:|
| **Haruka pure GAT + paired-slot cleanup** | **0.759** | 0.157 | **0.603** | 0.010 |
| CINEMA-OT + identical cleanup (fairness control) | 0.692 | **0.147** | 0.545 | **0.001** |
| Haruka pure GAT, matched timepoint (raw slots) | 0.750 | 0.236 | 0.514 | 0.090 |
| CINEMA-OT | 0.703 | 0.212 | 0.492 | 0.065 |
| contrastiveVI 2x2 | 0.605 | 0.153 | 0.452 | 0.007 |
| Haruka pure GAT, confounded slice batch | 0.757 | 0.313 | 0.445 | 0.166 |
| Haruka legacy COVET, count-corrected | 0.736 | 0.383 | 0.353 | 0.236 |
| CellCharter 2x2 | 0.564 | 0.342 | 0.221 | 0.196 |
| PCA 2x2 | 0.706 | 0.499 | 0.208 | 0.352 |

Pure-GAT Haruka leads seed-0 by 0.111 over stored CINEMA-OT and by 0.058 after
giving CINEMA-OT the identical label-free cleanup.
Its intended diagonals are
`background → arch = 1.000`, `spatial → niche = 1.000`,
`salient → salient intrinsic = 0.745`, and
`salient_spatial → salient domain = 0.293`. Its salient-spatial domain recovery is
still below CINEMA-OT's 0.610, so Haruka does not win every quadrant. It wins the
assignment by recovering the intended factors strongly while reducing excess
off-diagonal leakage to 0.010.

Removing COVET itself raised the old slice-batch margin from 0.353 to 0.445. Fixing
the batch key then raised it to 0.514 by reducing off-diagonal dependence from
0.313 to 0.236, with only a small diagonal reduction (0.757 to 0.750). This is the
main resolved model/dataset interface failure.

The final cleanup is a fixed, label-free ridge orthogonalization: subtract from
`background` and `salient_spatial` only the linear component predictable from the
shared `spatial` GAT slot. It uses no condition or ground-truth column and runs
before post-hoc neighborhood aggregation. On the full seed-0 run it changes the
raw margin 0.514 → 0.603. It is enabled by default in `run_haruka.py` and recorded
in metadata; `--no-orthogonalize-paired-slots` retains the raw-slot ablation.
As a fairness control, applying the same transform to CINEMA-OT changes its margin
0.492 → 0.545: useful, but still below final Haruka. CINEMA-OT retains much better
salient-spatial target recovery after cleanup (0.591 vs Haruka 0.293), while Haruka
wins the full four-factor assignment.

## Superseded model-seed stability on the fixed v5 dataset

The dataset is held fixed and only Haruka's training/scoring seed changes; CINEMA-
OT is deterministic here. Dependence-only scoring skips secondary ARI and the
dataset-constant oracle recomputation, neither of which affects the margin.

| model seed | raw slots | paired-slot cleanup | paired gain |
|---:|---:|---:|---:|
| 0 | 0.514 | **0.603** | +0.089 |
| 1 | 0.442 | **0.565** | +0.123 |
| 2 | 0.492 | **0.585** | +0.093 |
| mean ± sample SD | 0.482 ± 0.037 | **0.584 ± 0.019** | +0.102 mean |

All three final Haruka runs exceed both stored CINEMA-OT (0.492) and the identical-
cleanup fairness control (0.545); the weakest Haruka seed leads the latter by
0.020. This addresses the old bimodal training concern for this fixed dataset,
but it is not a substitute for independent simulator seeds.

The historical pre-fix `both_sal8` row had margin 0.452, but it is not a valid
competing result because it trained a Poisson likelihood on log expression. The
corrected current tree has higher diagonal recovery (0.690 to 0.736) and higher
off-diagonal recovery (0.238 to 0.383). Do not attribute that full-run difference
to the count fix alone because the current tree also includes the control-zeroing
and branch-penalty changes described above; the paired half-size experiment is
the causal count-fix comparison.

## Negative ablations and remaining limitation

Matched half-size tests rule out two tempting GAT changes:

| ablation | margin | salient-spatial → own domain | salient-spatial → shared niche |
|---|---:|---:|---:|
| current k=8, center-conditioned query | **0.375** | **0.110** | 0.661 |
| k=64 | 0.360 | 0.096 | **0.616** |
| query-free attention | 0.276 | 0.000 | 0.990 |

More neighbors slightly reduce niche copying but lose more target signal. Removing
the center query is catastrophic: the GAT needs the receiver state to select which
neighbors matter. Both remain diagnostics; the current method keeps k=8 and
`gat_self_query=True`.

The benchmark also gives CINEMA-OT a legitimate home-field advantage:
`composition_shift=0` guarantees same-type control matches for every treated cell,
and CINEMA-OT is run per timepoint with a 300-dimensional treatment-effect matrix
plus three-hop CellCharter aggregation. Haruka uses an 8-dimensional one-hop GAT
slot. This should be reported, not “fixed” after seeing scores; future simulator
rebuilds should evaluate a preregistered composition-shift ladder for every method.

## Interpretation guardrails

- Dependence (`DEP`) with leave-slice-out evaluation is primary; ARI is retained
  only for discrete-factor clusterability.
- Compare `assignment_margin`, `oracle_offdiag`, and `excess_offdiag`, not just a
  large diagonal entry.
- The old model-seed table is superseded because its RNA seed was not applied.
  The corrected MAE-90 sweep has three explicit model seeds above cleaned CINEMA-OT,
  but should not be called generally better until independent simulator seeds are
  rebuilt under the fixed RNG contract and every baseline is rerun.
- Jobs launched before the correction must be rerun; their baseline outputs can
  be retained.
