#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yancui/haruka_v2
PY=/home/yancui/.conda/envs/haruka2/bin/python
MOSCOT_PY=/home/yancui/.conda/envs/moscot/bin/python
RESULTS=$ROOT/results_real_compgated_nb_final
FINAL=$ROOT/final_results_nb
MODEL_NAME=haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0
CC_NAME=haruka_z_s_cellcharter_cc3_condition_only
SEED=0
if [[ $# -ne 1 ]]; then
  echo "usage: run_final_six_dataset_analysis.sh TASK" >&2
  exit 2
fi
TASK=$1

export PYTHONHASHSEED=$SEED
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

run_umap() {
  local max_plot=${6:-25000}
  "$PY" "$ROOT/bench_real_discovery/plot_sample_batch_umaps.py" \
    --data "$1" --cc-dir "$2" --out-dir "$3/umap" \
    --study "$4" --sample-key "$5" --max-plot "$max_plot" --seed "$SEED"
}

case "$TASK" in
  kras)
    DATA=/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad
    BASE=$RESULTS/mouselung_kras_xenium_clean_noC3
    MODEL=$BASE/$MODEL_NAME
    CC=$BASE/$CC_NAME
    OUT=$FINAL/kras
    mkdir -p "$OUT/biology"
    run_umap "$DATA" "$CC" "$OUT" "KRAS inhibitor Xenium" slice_id
    "$PY" "$ROOT/bench_kras_xenium/analyze_kras_preprint_style.py" \
      --data "$DATA" --cc-dir "$CC" --haruka-dir "$MODEL" \
      --out-dir "$OUT/biology/global" --seed "$SEED"
    "$PY" "$ROOT/bench_kras_xenium/analyze_kras_macrophage_condition_only.py" \
      --data "$DATA" --cc-dir "$CC" --out-dir "$OUT/biology/macrophage" --seed "$SEED"
    "$PY" "$ROOT/bench_kras_xenium/analyze_kras_relative_response.py" \
      --data "$DATA" --cc-dir "$CC" --haruka-dir "$MODEL" \
      --global-assignments "$OUT/biology/global/tables/cell_cluster_assignments.csv.gz" \
      --out-dir "$OUT/biology/responder" --seed "$SEED"
    "$PY" "$ROOT/bench_kras_xenium/analyze_kras_tumor_condition_only.py" \
      --cc-dir "$CC" \
      --assignments "$OUT/biology/responder/tables/treated_tumor_relative_responder_assignments.csv.gz" \
      --out-dir "$OUT/biology/tumor_trajectory" --seed "$SEED"
    ;;
  starmap_ad)
    DATA=$ROOT/bench_real_discovery/data/ad.h5ad
    BASE=$ROOT/results_real_compgated_lambda10_samplebatch_sepheads/ad
    MODEL=$BASE/$MODEL_NAME
    CC=$BASE/$CC_NAME
    OUT=$FINAL/starmap_ad
    mkdir -p "$OUT/biology"
    run_umap "$DATA" "$CC" "$OUT" "STARmap AD" slice_id
    "$PY" "$ROOT/bench_real_discovery/analyze_ad_plaque.py" \
      --data "$DATA" --emb "$MODEL" --out "$OUT/biology/plaque_cross_section" --seed "$SEED"
    "$PY" "$ROOT/bench_real_discovery/analyze_domain_celltype_enrichment.py" \
      --data "$DATA" --cc-dir "$CC" --out-dir "$OUT/biology/domain_annotation" \
      --sample-key slice_id --cell-type-key cell_type --seed "$SEED"
    ;;
  codex_ad)
    DATA=$ROOT/bench_codex_cns_2026/data/processed/codex_cns_myeloid_primary_no_aml_no_combat.h5ad
    BASE=$ROOT/results_real_compgated_lambda10_samplebatch_sepheads/codex_cns_myeloid_primary_no_aml_no_combat
    MODEL=$BASE/$MODEL_NAME
    CC=$BASE/$CC_NAME
    OUT=$FINAL/codex_ad
    mkdir -p "$OUT/biology"
    run_umap "$DATA" "$CC" "$OUT" "CODEX AD" donor_id
    "$PY" "$ROOT/bench_codex_cns_2026/evaluate_readouts.py" \
      --data "$DATA" --emb "$MODEL" --out "$OUT/biology/donor_heldout_plaque" --seed "$SEED"
    "$PY" "$ROOT/bench_real_discovery/analyze_domain_celltype_enrichment.py" \
      --data "$DATA" --cc-dir "$CC" --out-dir "$OUT/biology/domain_annotation" \
      --sample-key donor_id --cell-type-key paper_initial_leiden --seed "$SEED"
    ;;
  sparc_seq)
    DATA=$ROOT/bench_spacseq/data/spac_lung_functional_guides_k8_hvg3000.h5ad
    BASE=$RESULTS/spac_lung_functional_guides_k8_hvg3000
    MODEL=$BASE/$MODEL_NAME
    CC=$BASE/$CC_NAME
    OUT=$FINAL/sparc_seq
    mkdir -p "$OUT/biology"
    run_umap "$DATA" "$CC" "$OUT" "SPARC-seq" slice_id
    "$PY" "$ROOT/bench_spacseq/score_functional_guides.py" \
      --data "$DATA" --emb_dir "$MODEL" --out "$OUT/biology/functional_guide_metrics.json" --seed "$SEED"
    "$PY" "$ROOT/bench_spacseq/score_intrinsic_responders.py" \
      --data "$DATA" --emb_dir "$MODEL" --out_prefix "$OUT/biology/state_matched_nt_response"
    "$PY" "$ROOT/bench_spacseq/analyze_scc_cluster_properties.py" \
      --data "$DATA" --emb-dir "$CC" \
      --response-csv "$OUT/biology/state_matched_nt_response.cells.csv" \
      --out-dir "$OUT/biology/scc_cluster_properties" --seed "$SEED"
    "$PY" "$ROOT/bench_spacseq/plot_intrinsic_responders_compact.py" \
      --prefix "$OUT/biology/state_matched_nt_response" \
      --out "$OUT/biology/state_matched_nt_responder_summary.png"
    "$PY" "$ROOT/bench_spacseq/compare_response_heterogeneity.py" \
      --data "$DATA" --emb-dir "$MODEL" \
      --out-prefix "$OUT/biology/external_response_heterogeneity" --seed "$SEED"
    "$PY" "$ROOT/bench_spacseq/plot_direct_response_analysis.py" \
      --data "$DATA" --emb "$MODEL" \
      --prediction-prefix "$OUT/biology/external_response_heterogeneity" \
      --out-dir "$OUT/biology/direct_response_visualization" --seed "$SEED"
    "$PY" "$ROOT/bench_real_discovery/analyze_domain_celltype_enrichment.py" \
      --data "$DATA" --cc-dir "$CC" --out-dir "$OUT/biology/domain_annotation" \
      --sample-key slice_id --cell-type-key cell_type --seed "$SEED"
    ;;
  oskm)
    DATA=$ROOT/bench_oskm_reprogramming/data/oskm_old_full_684253.h5ad
    BASE=$ROOT/results_real_compgated_nb_full_oskm/oskm_old_full_684253
    MODEL=$BASE/$MODEL_NAME
    CC=$BASE/$CC_NAME
    OUT=$FINAL/oskm
    mkdir -p "$OUT/biology"
    run_umap "$DATA" "$CC" "$OUT" "Old control versus old OSKM (full, NB)" slice_id 0
    "$PY" "$ROOT/bench_oskm_reprogramming/analyze_oskm_salient.py" \
      --data "$DATA" --young "$ROOT/bench_oskm_reprogramming/data/young_reference_full_330924.h5ad" \
      --emb "$CC" --out "$OUT/biology/rejuvenation" --max-umap-cells 0 --seed "$SEED"
    ;;
  lung_fibrosis)
    DATA=$ROOT/bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad
    BASE=$RESULTS/gse250346_unaffected_vs_all_fibrosis
    MODEL=$BASE/$MODEL_NAME
    CC=$BASE/$CC_NAME
    OUT=$FINAL/lung_fibrosis
    mkdir -p "$OUT/biology"
    run_umap "$DATA" "$CC" "$OUT" "GSE250346 lung fibrosis" sample
    "$PY" "$ROOT/bench_lung_fibrosis_gse250346/analyze_fibrosis_cc.py" \
      --data "$DATA" --cc-dir "$CC" --haruka-dir "$MODEL" \
      --out-dir "$OUT/biology/fibrosis_severity" --seed "$SEED"
    LANDMARKS=$OUT/biology/fibrosis_severity/moscot_landmarks.h5ad
    "$PY" "$ROOT/bench_real_discovery/analyze_domain_celltype_enrichment.py" \
      --data "$DATA" --cc-dir "$CC" --out-dir "$OUT/biology/domain_annotation" \
      --sample-key slice_id --cell-type-key cell_type --seed "$SEED"
    "$MOSCOT_PY" "$ROOT/bench_lung_fibrosis_gse250346/run_moscot_alignment.py" \
      --landmarks "$LANDMARKS" --out-dir "$OUT/biology/moscot_alignment"
    "$MOSCOT_PY" "$ROOT/bench_lung_fibrosis_gse250346/score_moscot_alignment.py" \
      --landmarks "$LANDMARKS" \
      --aligned "$OUT/biology/moscot_alignment/moscot_aligned_landmarks.csv.gz" \
      --out-dir "$OUT/biology/moscot_alignment"
    "$MOSCOT_PY" "$ROOT/bench_lung_fibrosis_gse250346/run_moscot_state_transitions.py" \
      --landmarks "$LANDMARKS" --out-dir "$OUT/biology/moscot_state_transitions"
    "$MOSCOT_PY" "$ROOT/bench_lung_fibrosis_gse250346/run_moscot_salient_marginal.py" \
      --landmarks "$LANDMARKS" --out-dir "$OUT/biology/moscot_salient_marginal" --device gpu
    "$MOSCOT_PY" "$ROOT/bench_lung_fibrosis_gse250346/analyze_initial_state_fate.py" \
      --landmarks "$LANDMARKS" \
      --data "$DATA" --haruka-dir "$MODEL" \
      --out-dir "$OUT/biology/initial_state_fate" --device gpu
    ;;
  *)
    echo "Unknown task: $TASK" >&2
    exit 2
    ;;
esac
