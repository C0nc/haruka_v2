#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yancui/haruka_v2
PY=/home/yancui/.conda/envs/haruka2/bin/python
MOSCOT_PY=/home/yancui/.conda/envs/moscot/bin/python
RESULTS=$ROOT/results_real_compgated_lambda10_samplebatch_sepheads
MODEL_NAME=haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0
CC_NAME=haruka_z_s_cellcharter_cc3_condition_only
TASK=${1:?usage: run_requested_compgated_paired_analyses.sh TASK}

run_python() {
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" "$@"
}

case "$TASK" in
  sparc)
    DATA=$ROOT/bench_spacseq/data/spac_lung_functional_guides_k8_hvg3000.h5ad
    BASE=$RESULTS/spac_lung_functional_guides_k8_hvg3000
    MODEL=$BASE/$MODEL_NAME
    CC=$BASE/$CC_NAME
    OUT=$BASE/paired_analysis
    mkdir -p "$OUT"
    run_python "$ROOT/bench_spacseq/score_functional_guides.py" \
      --data "$DATA" --emb_dir "$MODEL" --out "$OUT/functional_guide_metrics.json"
    run_python "$ROOT/bench_spacseq/score_intrinsic_responders.py" \
      --data "$DATA" --emb_dir "$MODEL" --out_prefix "$OUT/state_matched_nt_response"
    run_python "$ROOT/bench_spacseq/analyze_scc_cluster_properties.py" \
      --data "$DATA" --emb-dir "$CC" \
      --response-csv "$OUT/state_matched_nt_response.cells.csv" \
      --out-dir "$OUT/scc_cluster_properties"
    ;;
  starmap_ad)
    BASE=$RESULTS/ad
    run_python "$ROOT/bench_real_discovery/analyze_ad_plaque.py" \
      --data "$ROOT/bench_real_discovery/data/ad.h5ad" \
      --emb "$BASE/$MODEL_NAME" --out "$BASE/paired_analysis/plaque_cross_section"
    ;;
  codex_ad)
    BASE=$RESULTS/codex_cns_myeloid_primary_no_aml_no_combat
    run_python "$ROOT/bench_codex_cns_2026/evaluate_readouts.py" \
      --data "$ROOT/bench_codex_cns_2026/data/processed/codex_cns_myeloid_primary_no_aml_no_combat.h5ad" \
      --emb "$BASE/$MODEL_NAME" --out "$BASE/paired_analysis/donor_heldout_plaque"
    ;;
  oskm)
    BASE=$RESULTS/oskm_old_matched_160k
    run_python "$ROOT/bench_oskm_reprogramming/analyze_oskm_salient.py" \
      --data "$ROOT/bench_oskm_reprogramming/data/oskm_old_matched_160k.h5ad" \
      --young "$ROOT/bench_oskm_reprogramming/data/young_reference_20k.h5ad" \
      --emb "$BASE/$CC_NAME" --out "$BASE/paired_analysis/rejuvenation"
    ;;
  kras)
    BASE=$RESULTS/mouselung_kras_xenium_clean_noC3
    run_python "$ROOT/bench_kras_xenium/analyze_kras_preprint_style.py" \
      --data /home/yancui/mouselung_kras_xenium_clean_noC3.h5ad \
      --cc-dir "$BASE/$CC_NAME" --haruka-dir "$BASE/$MODEL_NAME" \
      --out-dir "$BASE/paired_analysis/treated_replicate_aware"
    ;;
  lung)
    BASE=$RESULTS/gse250346_unaffected_vs_all_fibrosis
    OUT=$BASE/paired_analysis/fibrosis_severity
    run_python "$ROOT/bench_lung_fibrosis_gse250346/analyze_fibrosis_cc.py" \
      --data "$ROOT/bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad" \
      --cc-dir "$BASE/$CC_NAME" --haruka-dir "$BASE/$MODEL_NAME" --out-dir "$OUT"
    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      "$MOSCOT_PY" "$ROOT/bench_lung_fibrosis_gse250346/analyze_initial_state_fate.py" \
      --landmarks "$OUT/moscot_landmarks.h5ad" \
      --data "$ROOT/bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad" \
      --haruka-dir "$BASE/$MODEL_NAME" --out-dir "$OUT/initial_state_fate" --device gpu
    ;;
  *)
    echo "Unknown task: $TASK" >&2
    exit 2
    ;;
esac
