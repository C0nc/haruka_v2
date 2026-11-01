# Nonlinear identifiable benchmark — 2026-08-03

## Purpose

This is the benchmark for testing whether Haruka's nonlinear GAT representation has an advantage over raw CINEMA-OT, while keeping the salient-spatial factor identifiable.

## Dataset contract

Dataset: `data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad`

- same 72,200-cell lattice and background factors as the stress dataset;
- `effect_mode=gate` (multiplicative, cell-type-specific nonlinear gate), not the additive mode;
- `gate_strength=0.8`, `type_gate=true`;
- `sal_rates=(0.1, 0.5, 0.9)` and `delta_sp=1.0` so the spatial factor is identifiable;
- pooled oracle ceiling for `gt_sal_domain`: **0.955**.

The original `simv4_g0.8_r0.3_p0.15_d1_s0.h5ad` remains the weak-effect stress test.

## Compared methods

- **Haruka**: pure GAT, hidden width 256, `spatial_attach=both`, 8-dimensional salient/spatial slots, staged MAE + paired-slot regularization.
- **CINEMA-OT raw**: per-timepoint ICA confounder plus entropic OT treatment-effect residual. Spatial slots are only the standard fixed CellCharter aggregation of CINEMA's intrinsic outputs; CINEMA has no learned nonlinear spatial encoder.

## Results

| method | salient-spatial dependence | salient-spatial probe AUROC | assignment margin |
|---|---:|---:|---:|
| Haruka hidden256 | 0.978 | 0.991 | **0.531** |
| CINEMA-OT raw | **0.989** | **0.998** | 0.426 |

Haruka wins the aggregate assignment margin by **+0.105**. CINEMA-OT is marginally stronger on the deliberately strengthened single salient-domain quadrant, but Haruka is better at assigning factors to the correct slots with less cross-slot leakage.

Artifacts:

- Haruka: `results_easy/simv4_g0.8_r0.8_p1_d1_s0/haruka_both_sal8_timebatch_seed0`
- CINEMA-OT: `results_easy_cinema/simv4_g0.8_r0.8_p1_d1_s0/cinemaot_seed0`
- CINEMA metrics: `results_easy_cinema/metrics_raw_primary.jsonl`
