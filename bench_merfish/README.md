# MERFISH salient-domain benchmark

A real-tissue replacement for `haruka_sim.py`. Every Haruka-v2 number in the repo
root comes from a synthetic lattice; this one runs on the **Wyss-Coray mouse-brain
spatial-aging MERFISH atlas** (`process_all_brain.h5ad`, cohort `aging_coronal`,
300 genes, 2.47M cells) with a controlled effect injected so ground truth exists.

> **2026-08-02 benchmark correction.** The current runner is pure GAT only and
> uses `timepoint`, whose levels contain both conditions, as the decoder batch.
> The primary contract is now raw-only: MAE representation learning is followed
> by an in-training ridge-predictability separation stage, and the public
> background representation excludes internal GAT-carrier dimensions. No post-hoc
> cleanup is used for the headline score; raw CINEMA-OT is the sole primary CINEMA
> baseline. A controlled capacity screen selects hidden width 256 at the original
> 150-epoch schedule: raw Haruka is 0.546 ± 0.024 versus CINEMA-OT 0.492.
> Post-hoc orthogonalization remains available only as a diagnostic.
> Legacy runs with `cov_gene_num > 0` are provisional and must not be compared
> with the baselines.
> `construct_cov()` rebuilt the augmented `AnnData` from log-normalised `X` and
> silently discarded `layers["counts"]`, so the Poisson gene likelihood trained on
> log expression. It also computed the library size over genes *and* the appended
> continuous COVET targets. The baselines already used the count layer and are not
> affected. The runner now preserves raw counts through augmentation, refuses this
> invalid contract, and records `training_layer` plus
> `count_contract_validated` in `meta.json`; the likelihood library uses only
> the gene block. Historical Haruka result directories remain for provenance, but
> corrected COVET runs are retained only as diagnostics. Current model comparisons
> live under `results_puregat*/`; see `BENCHMARK_AUDIT_2026-08-02.md`.

## The design in one paragraph

Four **timepoints**, each a pair of age-matched mice: one is `control`, one is
`disease`. No cell appears in both arms, so the two arms differ by real
mouse-to-mouse, section and slide-batch variation — the nuisance a contrastive
method claims to quarantine into its background latent. Into the treated arm we
inject two **subtle salient spatial domains**: smooth Gaussian random fields
defined in a shared unit coordinate frame (so the same domains appear on every
treated slice, not per-slice noise), each driving a disjoint 30-gene program with
a graded sigmoid edge. Counts are recovered from the atlas's log-normalised matrix
and **both** arms are Poisson-resampled, so neither arm carries a distinguishable
noise level. Effect size `delta` is swept, `delta=0` included as a false-positive
arm.

| timepoint | control (age, slide) | disease (age, slide) |
|---|---|---|
| T1_young   | mouse 53, 3.4 mo, B3  | mouse 1,  3.8 mo, A8 |
| T2_mid     | mouse 19, 15.5 mo, A6 | mouse 70, 15.8 mo, B2 |
| T3_old     | mouse 75, 18.8 mo, B4 | mouse 80, 19.8 mo, B4 |
| T4_veryold | mouse 38, 26.7 mo, A7 | mouse 93, 28.5 mo, B5 |

Slide batch is alternated between arms across timepoints so `condition` is not a
proxy for batch; T3 shares one slide outright and is the internal check that the
effect survives when technical variation is minimal.

## The one thing that makes this benchmark valid

**The implanted domain is balanced against anatomy and cell-type composition by
construction.** The field is thresholded *within each `subregion x celltype`
stratum* at that stratum's own quantile, so `P(domain | subregion, celltype)` is
constant. Without this, a spatially smooth domain in a brain slice correlates with
the anatomy that is also spatially organised, and every non-contrastive baseline
collects free credit.

The first version of this benchmark did *not* do that. It rejected candidate
fields on `NMI(domain, subregion) < 0.03` and passed — while a logistic probe on
the one-hot subregion label predicted domain membership at **AUROC 0.64**. NMI is
the wrong statistic for a 24%/76% split across 7 classes: small per-class
enrichments barely move it and are trivially learnable. `self_check.py` caught it.
After stratification:

| check | before | after |
|---|---|---|
| C: anatomy alone predicts the domain | 0.641 | **0.510** |
| B: oracle probe on the *control* arm | 0.603 | **0.528** |
| D: null arm (delta=0) | 0.607 | **0.537** |

The stratification pulls against the thing that makes this a *spatial* benchmark,
so `self_check.py` also measures whether the domain survived as a contiguous
region — fraction of a cell's 12 nearest same-slice neighbours sharing its label:

| labelling | k-NN agreement |
|---|---|
| random label, same prevalence (floor) | 0.633 |
| real anatomical `subregion` (reference) | 0.836 |
| unstratified domain | 0.973 |
| **implanted domain (subregion × celltype stratified)** | **0.927** |

The implanted domain is more spatially coherent than real anatomy. This check
exists because a rendered scatter at 1.2-pt markers reads as salt-and-pepper when
the minority class is 24%, and judging it by eye gives the wrong answer — it did.
Stratifying on `subregion` alone would have been nearly as good (contiguity 0.940,
anatomy AUROC 0.515); adding `celltype` buys 0.005 of leakage for 0.013 of
contiguity.

`self_check.py` must pass before any method number is worth reading.

## Effect-size ladder (seed 0)

`delta` scales a log-multiplier on the 30 program genes. "Incremental signal" is
the oracle probe's treated-arm AUROC minus the same probe on control cells with
the same geometry — what the injection actually contributes.

| delta | fold change on program genes | oracle AUROC (treated) | incremental |
|---|---|---|---|
| 0    | 1.00 | 0.537 | 0.010 |
| 0.3  | 1.12 | 0.559 | 0.030 |
| 0.6  | 1.25 | 0.613 | 0.084 |
| 1.2  | 1.56 | 0.715 | 0.190 |
| 2.4  | 2.42 | 0.840 | 0.312 |

24% of cells fall in a domain; 4 slices per arm, 9,000 cells per slice, 72,000
cells total.

## Methods compared

| method | env | background rep | salient rep |
|---|---|---|---|
| `pca` | cellcharter | log-normalised expression | same |
| `cellcharter` | cellcharter | scVI(batch=slice) + 3-hop neighbourhood concat | same |
| `cinemaot` | vio | confounder ICs | per-cell OT treatment effect |
| `contrastivevi` | cellcharter | upstream `scvi.external.ContrastiveVI` | " |
| `contrastivevi_cc` | cellcharter | same fit + CellCharter aggregation | " |
| `haruka_vanilla` | haruka2 | patched ContrastiveVI, `n_spatial_latent=0` | " |
| `haruka_background` | haruka2 | `n_spatial_latent=16, spatial_attach=background` | " |
| `haruka_both` | haruka2 | `spatial_attach=both` (the 2x2) | " |

`contrastivevi_cc` is the control that makes the Haruka numbers interpretable:
Haruka's `extract_rep` also ends in `cc.gr.aggregate_neighbors`, so without it any
Haruka win could just be the aggregation step. It is fit **once** with
`contrastivevi` so the two arms share a training seed.

`cinemaot` is run **per timepoint** with `ref_label="disease"`. The natural
direction would index the effect matrix by control cells, where the implant does
not exist, and CINEMA-OT would score zero for a reason unrelated to its ability.

Single-representation methods (`pca`, `cellcharter`) are scored by putting their
one embedding into both slots — that is their actual claim, and the numbers show
what it costs.

## Downstream is identical for everyone

**z-score → PCA(50) → full-covariance `GaussianMixture`**, in `common.py`. Haruka's
own `cluster()` routes to `cellcharter.tl.Cluster` (torchgmm, GPU-only,
`random_state` not honoured) and CINEMA-OT ships no clusterer; scoring three
methods with three clusterers would confound the representation with the
partitioner. The sklearn reimplementation is deterministic and matches the config
the Haruka-v2 README settled on.

One departure from `HarukaSpatial.extract_rep`: it returns representations for the
**target arm only**, which makes `ARI(background, condition)` — the background
latent's most important property — impossible to measure. `run_haruka.py` pulls
representations for all cells and applies the same two cellcharter calls to the
full object.

## Layout

| file | role |
|---|---|
| `prep_cache.py` | one-time extraction of the 10 mice from the 6.9 GB atlas |
| `build_sim.py` | the simulator: pairs, domains, programs, Poisson emission |
| `self_check.py` | validates the *benchmark*; run before trusting any method number |
| `run_baselines.py` | pca / cellcharter / contrastivevi / contrastivevi_cc |
| `run_cinemaot.py` | CINEMA-OT (env `vio`) |
| `run_haruka.py` | Haruka v2 spatial variants (env `haruka2`) |
| `score.py` | the shared metric suite |
| `report.py` | aggregation, tables, figure |
| `run_one.sh` | all methods on one dataset, one GPU |

## Running it

```bash
python prep_cache.py                                   # once, ~3 min
python build_sim.py --deltas 0 0.3 0.6 1.2 2.4 --seeds 0 1 2
python self_check.py --data data/sim_d*.h5ad           # must pass
./run_one.sh 1.2 0 4                                   # delta, seed, gpu
python report.py --metrics results/metrics.jsonl
```

Deltas are independent; run them on separate GPUs in parallel.

## Read detection as an increment over each method's own null, not as a raw AUROC

On the **delta = 0** build — nothing injected at all — the salient-side probe still
scores well above chance against the counterfactual domain label:

| method | `sal_AUROC_domain` at delta = 0 |
|---|---|
| cellcharter | 0.673 |
| contrastivevi_cc | 0.618 |
| haruka_background | 0.602 |
| cinemaot | 0.558 |
| contrastivevi | 0.555 |

This is not false detection and it is not a bug. The implanted domain is a smooth
field defined in a coordinate frame **shared across slices** — it has to be, or it
would be per-slice noise that no method could reasonably be asked to find — so
tissue *position* alone predicts the label, and it does so consistently across
animals, which is precisely what leave-one-animal-out grouping cannot protect
against. Any representation carrying positional information clears 0.5 for free,
and 3-hop neighbourhood aggregation supplies it in abundance. Note the ordering:
the two most heavily spatially-smoothed methods have the two highest floors.

So the headline number is `sal_AUROC_domain_over_null` — each method's score minus
its own delta = 0 score. A high floor is not cheating, it is smoothness; the
increment is what survives once smoothness is accounted for. `report.py` computes
this and prints it above the raw table.

This is the second time the null arm has paid for itself: it caught the
anatomy-leakage flaw in the domain construction, and now it calibrates a metric
that is uninterpretable in absolute terms.

## Known caveats

* **CINEMA-OT's FastICA does not converge** on this data at its default tolerance
  (`ConvergenceWarning` on every timepoint). The confounder components it returns
  are therefore a partially-converged decomposition. `cinemaot_unweighted`
  constructs its `FastICA` internally with no way to raise `max_iter` from the
  call site, so this is a property of the method as shipped rather than a
  configuration choice made here — but any CINEMA-OT number below should be read
  as "CINEMA-OT 0.0.5 with its own defaults", not "the best CINEMA-OT can do".
* The number of confounder components CINEMA-OT keeps varies per timepoint (it
  drops ICs whose Chatterjee coefficient against the treatment label exceeds
  `thres`). They are zero-padded to a common width; since each timepoint gets an
  independent ICA the components are not comparable across timepoints anyway.
* Subsampling to 9,000 cells per slice thins the tissue by ~8x, widening the k-NN
  radius by roughly `sqrt(8)`. The domains are far larger than that radius, so no
  spatial method is starved, but absolute neighbourhood scales are not those of
  the full sections.

## Environment notes

* `haruka2` is a clone of the `cellcharter` env with
  `contrastivevi_patch/*.py` copied over `scvi/external/contrastivevi/` and
  `scenvi` installed. It carries scvi-tools **1.4.0**, not the 1.2.2 the patch was
  written against; the patch replaces those modules wholesale and the surrounding
  scvi API it touches is unchanged between the two.
* The pure-GAT benchmark does not import `scenvi`. The old COVET API is lazy-loaded
  only when a library client explicitly requests `cov_gene_num > 0`; its JAX
  compatibility shim is isolated to that legacy call.
* This host has 256 cores and an OpenBLAS built for 128 threads; uncapped,
  sklearn's PCA/GMM dies with "tried to allocate too many memory regions". The
  runner scripts export `OMP_NUM_THREADS=8` and friends.

---

# The 2x2 benchmark (`build_sim_2x2.py`) — current design

The single-domain benchmark above scores two things: an implanted spatial domain
and anatomy. `spatial_attach="both"` emits FOUR latents, so on that benchmark two
of them have nothing legitimate to own and the configuration is penalised for
partitioning correctly. This design gives each latent a factor.

|  | intrinsic (own expression) | spatial (neighbours) |
|---|---|---|
| **background** (shared across arms) | `background` → `gt_arch` | `spatial` → `gt_niche` |
| **salient** (treated arm only) | `salient` → `gt_sal_intrinsic` | `salient_spatial` → `gt_sal_domain` |

* **`gt_arch`** — real anatomy, the atlas's 7 subregions. Not injected.
* **`gt_niche`** — 35% of cells are senders broadcasting one of 2 ligand classes
  and expressing a 20-gene marker block at `delta x 2.0`. A receiver's niche is
  the Gaussian-kernel ligand balance over its 100 nearest neighbours **with itself
  excluded**; label = terciles. The receiver does **not** respond to its own niche
  (`FACTOR_WEIGHTS["niche"] = 0.0`) — this is `haruka_sim.py`'s rho = 0 regime and
  is what makes a cell's own row independent of its own niche.
* **`gt_sal_intrinsic`** — 3 cell states assigned i.i.d. **in space**, 30-gene
  programs, treated arm only (controls carry counterfactual labels so the
  false-positive probe is defined). Spatially i.i.d. on purpose: this is the
  factor neighbourhood smoothing cannot help with, and without it every metric
  rewards blurring.
* **`gt_sal_domain`** — the implanted domains from the first design, unchanged.

Scoring is a **representation-by-truth dependence matrix**, summarised by
`assignment_margin = mean(diagonal) - mean(off-diagonal)`. A model that dumps
everything into one latent posts a high diagonal AND a high off-diagonal and wins
nothing — the failure a single salient score cannot see. All v4/v5 truths are
categorical generative states and are scored by grouped macro one-vs-rest AUROC;
ARI remains a secondary clusterability readout. Earlier scorer comments referring
to nonexistent `niche_score` and `sal_domain_score` columns were stale; the code's
categorical fallback already produced the stored numbers, so this correction does
not change the ranking.

## Validation (`self_check_2x2.py`, 12 builds, 3 seeds)

Each factor must be recoverable ONLY from the source its quadrant claims:

| factor | own expression | neighbours' | control arm |
|---|---|---|---|
| `gt_arch` | 0.94 | 0.94 | — |
| `gt_niche` | **0.48–0.51** | 0.58 → 0.67 → 0.74 | — |
| `gt_sal_intrinsic` | 0.51 → 0.91 → 0.98 | **0.49–0.51** | **0.50** |
| `gt_sal_domain` | — | 0.90 → 0.96 | 0.72 floor |

## What the self-checks caught, and what they got wrong

Two genuine construction bugs, both of which would have invalidated the
comparison:

1. **The receiver responded to its own niche** (own-AUROC 0.862 when it must be at
   chance). The niche was written into the cell's own row, so an intrinsic latent
   could read it off and the spatial latents would have been credited for
   something requiring no spatial reasoning. Fixed by rho = 0.
2. **The ligand mark did not scale with delta** while every other factor did, so
   the niche was progressively swamped as effect size rose — 0.652 at delta 1.2
   down to 0.502 at delta 2.4 on the same construction.

And three false alarms, all of which were **broken metrics, not broken data** —
the more expensive lesson:

3. **"The domain got shredded"** — judged by eye from a scatter plot. k-NN label
   agreement was 0.927 versus a 0.633 random floor and 0.836 for real anatomy: the
   implanted domain is more spatially coherent than mouse brain anatomy is.
4. **"The niche is unrecoverable"** — the probe ran PCA-50 first, and PCA of a
   neighbourhood-averaged matrix is almost entirely anatomy, so a low-variance
   factor is truncated away. Ridge on raw genes found it at R² 0.30 on the same
   data. Probes now default to `n_pcs=None`; clustering still uses PCA-50.
5. **"Seed 1's niche is dead"** (flat 0.50 at every delta, four rounds of
   investigation) — the three-class tercile label was binarised as "≠ the majority
   class", and with terciles exactly balanced the tie-break can select the MIDDLE
   tercile, making the target U-shaped in the underlying score and unlearnable by
   any linear probe. Measured across seeds: `≠ low` 0.82/0.84/0.88, `≠ MIDDLE`
   0.49/0.51/0.51, `≠ high` 0.83/0.84/0.87. Multi-class truths now use one-vs-rest
   macro AUROC, which is invariant to class ordering.

The general lesson: a failing number is evidence about the *measurement* at least
as often as about the data, and "flat across every effect size" is the signature
of a degenerate metric rather than a weak signal — a real but weak factor still
shows a dose response.

## Results (96 rows: 8 methods x 4 deltas x 3 seeds)

Full tables in `results_2x2/report_2x2.txt`, figure in `results_2x2/benchmark_2x2.png`,
raw rows in `results_2x2/metrics_2x2.csv`.

### Assignment margin (diag - offdiag)

| method | d=0 | 0.6 | 1.2 | 2.4 |
|---|---|---|---|---|
| haruka_vanilla | 0.244 | 0.246 | 0.236 | 0.200 |
| contrastivevi | 0.141 | 0.131 | 0.286 | **0.414** |
| contrastivevi_cc | 0.222 | 0.214 | 0.180 | 0.206 |
| cellcharter | 0.166 | 0.146 | 0.130 | 0.126 |
| haruka_background | 0.106 | 0.098 | 0.111 | 0.092 |
| cinemaot | 0.027 | 0.011 | 0.016 | 0.011 |
| **haruka_both** | **-0.006** | **-0.006** | **-0.011** | **-0.012** |

NOT comparable across differing latent counts -- `haruka_vanilla` leads because it
is scored on 2 easy diagonal cells while `haruka_both` is scored on 4. Use the
per-quadrant tables.

### Per quadrant

`background -> arch` (ARI): haruka_background 0.488-0.511, haruka_vanilla ~0.49,
contrastivevi_cc ~0.46, cellcharter 0.39-0.49, contrastivevi ~0.28,
**haruka_both 0.18-0.23**, cinemaot 0.02-0.06.

`spatial -> niche` (ARI): **0.000-0.001 in all 24 rows**. Probe AUROC shows a faint
dose response (haruka_background 0.488->0.567, haruka_both 0.476->0.631) but a plain
neighbour-mean of raw expression reaches 0.67-0.72 on the same labels. The
SIMVI-style branch recovers LESS of the interaction factor than unweighted averaging.

`salient -> sal_intr` (ARI at d=2.4): contrastivevi **0.568**, every other method
0.000. Probe AUROC is 0.85-0.99 for nearly all of them, so 3-hop aggregation does not
erase the factor -- it makes it unclusterable. `extract_rep` always aggregates, so
every Haruka variant inherits this.

`salient_spatial -> sal_dom` (probe AUROC): 0.589 -> 0.651 -> 0.713 -> **0.943**.
Clean dose response, +0.354 over its own null floor, and no other method has this
latent. This is the one quadrant where the 2x2 earns its keep.

### The `both` inversion

`haruka_both` is the only method with a NEGATIVE margin at every delta. At d=2.4
anatomy sits at 0.300 in `salient` and 0.328 in `salient_spatial` while `background`
gets 0.184; probe AUROC on `background -> arch` is 0.704 vs 0.958-0.962 for every
other Haruka variant, so the background latent is genuinely degraded rather than
merely unclustered. It is fully present at delta=0, where nothing is injected, so it
is a property of attaching the spatial branch to both encoders -- not a response to
the signal, and not an artefact of the benchmark having too few factors (the whole
point of this redesign was to give all four latents something to own; it did not
change the outcome).

This independently replicates on real tissue what the repo README reports on the
synthetic lattice (0/12 seeds for background stability under `both` vs 7-8/12 under
`background`), and supports its recommendation of `spatial_attach="background"`.

### Cross-cutting observation

Detection and clusterability diverge in almost every quadrant. Reporting ARI alone
would call several real signals absent; reporting AUROC alone would call CellCharter
the best salient-domain detector (0.969 raw) when it clusters at 0.005 because its
single embedding is the anatomy map. Both read-outs are needed.
