#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yancui/haruka_v2
PY=/home/yancui/.conda/envs/haruka2/bin/python
FINAL=$ROOT/final_results_nb
RESULTS=$ROOT/results_real_compgated_nb_final
CC_NAME=haruka_z_s_cellcharter_cc3_condition_only
STARMAP_CC=$ROOT/results_real_compgated_lambda10_samplebatch_sepheads/ad/$CC_NAME
CODEX_CC=$ROOT/results_real_compgated_lambda10_samplebatch_sepheads/codex_cns_myeloid_primary_no_aml_no_combat/$CC_NAME
DIST=$FINAL/ad_shared/plaque_distance
TRAJ=$FINAL/ad_shared/trajectory
SOURCE=$ROOT/ad_analysis_compgated_lambda10_batchremoved
SEED=0

export PYTHONHASHSEED=$SEED
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

mkdir -p "$DIST" "$TRAJ"
# These four files are deterministic functions of the raw coordinates/labels and
# are model-independent. Reuse avoids requiring the unavailable original CODEX
# plaque-centroid CSV bundle; all model-dependent analyses below are rerun.
cp "$SOURCE/starmap_plaque_distance_cells.csv.gz" "$DIST/"
cp "$SOURCE/codex_plaque_distance_cells.csv.gz" "$DIST/"
cp "$SOURCE/starmap_spatial_distance_to_abeta_ptau.png" "$DIST/"
cp "$SOURCE/codex_spatial_distance_to_dense_diffuse_abeta.png" "$DIST/"

"$PY" "$ROOT/ad_analysis/analyze_scc_pc1_plaque_distance.py" \
  --starmap-emb "$STARMAP_CC" --codex-emb "$CODEX_CC" --out "$DIST"
"$PY" "$ROOT/ad_analysis/run_scc_dpt_cellrank.py" \
  --datasets starmap codex --out "$TRAJ" \
  --starmap-embedding "$STARMAP_CC" --codex-embedding "$CODEX_CC" \
  --distance-root "$DIST" --seed "$SEED"
"$PY" "$ROOT/ad_analysis/analyze_initial_molecular_terminal_fates.py" --root "$TRAJ"
"$PY" "$ROOT/ad_analysis/plot_scc_dpt_spatial_velocity.py" --root "$TRAJ"
"$PY" "$ROOT/ad_analysis/plot_cellrank_sparse_spatial_projection.py" --root "$TRAJ"
"$PY" "$ROOT/ad_analysis/analyze_glial_terminal_fates.py" \
  --root "$TRAJ" --out "$TRAJ/glia_terminal_analysis"
"$PY" "$ROOT/ad_analysis/analyze_glial_enrichment_pseudotime_composition.py" \
  --root "$TRAJ" --starmap-embedding "$STARMAP_CC" --codex-embedding "$CODEX_CC" \
  --out "$TRAJ/glia_terminal_analysis/enrichment_and_composition"
