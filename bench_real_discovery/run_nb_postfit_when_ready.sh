#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yancui/haruka_v2
PY=/home/yancui/.conda/envs/haruka2/bin/python
RESULTS=$ROOT/results_real_compgated_nb_final
MODEL_NAME=haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0
CC_NAME=haruka_z_s_cellcharter_cc3_condition_only

wait_for_model() {
  local pid=$1
  local meta=$2
  while kill -0 "$pid" 2>/dev/null && [[ ! -f "$meta" ]]; do
    sleep 30
  done
  [[ -f "$meta" ]] || { echo "Model failed before writing $meta" >&2; exit 1; }
}

KRAS_MODEL=$RESULTS/mouselung_kras_xenium_clean_noC3/$MODEL_NAME
LUNG_MODEL=$RESULTS/gse250346_unaffected_vs_all_fibrosis/$MODEL_NAME

wait_for_model 1719469 "$LUNG_MODEL/meta.json"
"$PY" "$ROOT/bench_kras_xenium/rebuild_condition_only_salient_cc.py" --data "$ROOT/bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad" --input-dir "$LUNG_MODEL" --output-dir "$RESULTS/gse250346_unaffected_vs_all_fibrosis/$CC_NAME" --condition-key condition --condition-value fibrosis --slice-key slice_id
"$ROOT/bench_real_discovery/run_final_six_dataset_analysis.sh" lung_fibrosis

wait_for_model 1718032 "$KRAS_MODEL/meta.json"
"$PY" "$ROOT/bench_kras_xenium/rebuild_condition_only_salient_cc.py" --data /home/yancui/mouselung_kras_xenium_clean_noC3.h5ad --input-dir "$KRAS_MODEL" --output-dir "$RESULTS/mouselung_kras_xenium_clean_noC3/$CC_NAME" --condition-key condition_label --condition-value KRASi_treated --slice-key slice_id
"$ROOT/bench_real_discovery/run_final_six_dataset_analysis.sh" kras
"$PY" "$ROOT/bench_real_discovery/build_final_results_manifest.py" --out "$ROOT/final_results_nb" --seed 0
"$PY" "$ROOT/bench_real_discovery/finalize_final_results.py" --root "$ROOT/final_results_nb"
