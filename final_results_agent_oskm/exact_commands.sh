#!/usr/bin/env bash
set -euo pipefail

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

PY=/home/yancui/.conda/envs/haruka2/bin/python
ROOT=/home/yancui/haruka_v2/final_results_agent_oskm
DATA=/home/yancui/haruka_v2/bench_oskm_reprogramming/data/oskm_old_full_684253.h5ad
YOUNG=/home/yancui/haruka_v2/bench_oskm_reprogramming/data/young_reference_full_330924.h5ad
MODEL=/home/yancui/haruka_v2/results_real_compgated_nb_full_oskm/oskm_old_full_684253/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0
CC=/home/yancui/haruka_v2/results_real_compgated_nb_full_oskm/oskm_old_full_684253/haruka_z_s_cellcharter_cc3_condition_only
CLOCK_CACHE=/home/yancui/haruka_v2/bench_oskm_reprogramming/results_compgated_lambda10/analysis/spatial_aging_clock/spatial_clock_cell_scores.csv.gz

"$PY" /home/yancui/haruka_v2/bench_real_discovery/analyze_domain_celltype_enrichment.py \
  --data "$DATA" --cc-dir "$CC" --out-dir "$ROOT/domain_annotation" \
  --sample-key slice_id --cell-type-key cell_type --clusters 8 --neighbors 20 \
  --max-fit 50000 --seed 0

"$PY" "$ROOT/scripts/run_rigorous_oskm_analysis.py" \
  --data "$DATA" --young "$YOUNG" --model-dir "$MODEL" --cc-dir "$CC" \
  --domain-assignments "$ROOT/domain_annotation/domain_assignments.csv.gz" \
  --existing-umap /home/yancui/haruka_v2/final_results_nb/oskm/umap/sample_umap_coordinates.csv.gz \
  --out "$ROOT" --seed 0 --clusters 8 --response-neighbors 15

"$PY" "$ROOT/scripts/compare_clock_to_haruka.py" \
  --clock "$CLOCK_CACHE" \
  --scores "$ROOT/tables/oskm_cell_scores_and_domains.csv.gz" \
  --model-dir "$MODEL" --cc-dir "$CC" --out "$ROOT/clock"

"$PY" "$ROOT/scripts/summarize_pair_response_and_audit_clock.py" \
  --scores "$ROOT/tables/oskm_cell_scores_and_domains.csv.gz" \
  --clock-repo /home/yancui/haruka_v2/bench_oskm_reprogramming/external/SpatialAgingClock \
  --out "$ROOT/tables"

"$PY" "$ROOT/scripts/write_hashes.py" \
  --root "$ROOT" --data "$DATA" --young "$YOUNG" \
  --model-dir "$MODEL" --cc-dir "$CC"
