#!/usr/bin/env bash
set -euo pipefail

ROOT=${HARUKA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}
PY=${HARUKA_PY:-python}
RUNNER=$ROOT/bench_kras_xenium/run_haruka_separate_batch_heads.py
CC=$ROOT/bench_kras_xenium/rebuild_condition_only_salient_cc.py
PLOT=$ROOT/bench_real_discovery/plot_sample_batch_umaps.py
OUT=${HARUKA_RESULTS_ROOT:-$ROOT/results_real_compgated_nb_final}
GROUP=${1:?usage: run_all_compgated_samplebatch_sepheads.sh GROUP}

common=(
  --out_root "$OUT" --seed 0 --max_epochs 150
  --n_latent 32 --n_hidden 256 --n_layers 1 --decoder_layers 2
  --k_niche 20 --neighbor_embedding_dim 16
  --composition_loss_weight 1 --neighbor_mean_loss_weight 1
  --neighbor_expression_mode composition_gated --spatial_weight 100
  --background_wasserstein_penalty 0 --zs_independence_penalty 10
  --decoder_batch_dim 8 --batch_injection_mode decoder_only
  --augmentation_mode sparse --num_workers 4 --direct_only
)

run_one() {
  local study=$1 data=$2 batch_size=$3 sample_key=$4 batch_key=$5
  local condition_key=$6 target_label=$7 cell_type_key=$8
  local likelihood=$9 input_mode=${10} assay=${11} n_hvg=${12}
  local tag model cc_dir figure_dir
  tag=$(basename "$data" .h5ad)
  model=$OUT/$tag/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0
  cc_dir=$OUT/$tag/haruka_z_s_cellcharter_cc3_condition_only
  figure_dir=$OUT/$tag/sample_batch_diagnostic
  echo "===== TRAIN $study ====="
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" "$RUNNER" \
    --data "$data" --batch_size "$batch_size" --sample_key "$sample_key" \
    --batch_key "$batch_key" --condition_key "$condition_key" \
    --target_label "$target_label" --cell_type_key "$cell_type_key" \
    --gene_likelihood "$likelihood" --neighbor_input_mode "$input_mode" \
    --assay_type "$assay" --n_top_genes "$n_hvg" "${common[@]}"
  if [ ! -f "$model/salient_spatial.npy" ]; then   # fit unpacked from a fitted h5ad without neighbour-mean arrays
    echo "===== skip CC + sample UMAP diagnostics for $study (no salient_spatial.npy) ====="; return 0
  fi
  echo "===== CONDITION-ONLY CC $study ====="
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" "$CC" \
    --data "$data" --input-dir "$model" --output-dir "$cc_dir" \
    --condition-key "$condition_key" --condition-value "$target_label" \
    --slice-key "$sample_key"
  echo "===== SAMPLE UMAP $study ====="
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" "$PLOT" \
    --data "$data" --cc-dir "$cc_dir" --out-dir "$figure_dir" \
    --study "$study" --sample-key "$batch_key"
}

case "$GROUP" in
  large_kras)
    run_one "KRAS Xenium" "$ROOT/data/mouselung_kras_xenium_clean_noC3.h5ad" 32768 slice_id slice_id condition_label KRASi_treated celltype nb normalize sequencing 2000
    ;;
  large_lung)
    run_one "GSE250346 lung fibrosis" "$ROOT/bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad" 32768 slice_id sample condition fibrosis cell_type nb normalize sequencing 0
    ;;
  medium)
    run_one "SPARC-seq" "$ROOT/bench_spacseq/data/spac_lung_functional_guides_k8_hvg3000.h5ad" 2048 slice_id slice_id condition disease cell_type nb normalize sequencing 0
    run_one "STARmap AD" "$ROOT/bench_real_discovery/data/ad.h5ad" 4096 slice_id slice_id condition disease cell_type normal pretransformed sequencing 0
    run_one "CODEX AD" "$ROOT/bench_codex_cns_2026/data/processed/codex_cns_myeloid_primary_no_aml_no_combat.h5ad" 4096 donor_id donor_id condition disease paper_initial_leiden normal pretransformed protein 0
    run_one "OSKM" "$ROOT/bench_oskm_reprogramming/data/oskm_old_full_684253.h5ad" 32768 slice_id slice_id condition oskm cell_type nb normalize sequencing 0
    ;;
  discovery)
    run_one "MERFISH aging" "$ROOT/bench_aging/data/merfish_aging_100k.h5ad" 8192 graph_id slice_id condition disease cell_type nb normalize sequencing 0
    run_one "DKD" "$ROOT/bench_real_discovery/data/dkd.h5ad" 8192 graph_id sample condition disease cell_type nb normalize sequencing 2000
    run_one "Colitis" "$ROOT/bench_real_discovery/data/colitis.h5ad" 8192 graph_id sample condition disease cell_type normal pretransformed sequencing 0
    for section in 1 2 3 4; do
      run_one "PEtracer M1-$section" "$ROOT/bench_petracer_lineage/data_m1t1_pairs_22k/petracer_M1T1_M1-${section}_paired_22k.h5ad" 4096 graph_id graph_id condition disease gt_background_celltype nb normalize sequencing 0
    done
    ;;
  petracer_joint)
    run_one "PEtracer joint four replicates" "$ROOT/bench_petracer_lineage/data_m1t1_joint/petracer_M1T1_four_condition_110k.h5ad" 8192 graph_id graph_id condition disease gt_background_celltype nb normalize sequencing 0
    ;;
  *)
    echo "Unknown group: $GROUP" >&2
    exit 2
    ;;
esac
