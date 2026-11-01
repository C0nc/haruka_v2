#!/usr/bin/env bash
# Controlled A/B: identical runner + flags, only the normalization differs.
set -euo pipefail
ROOT=/home/yancui/haruka_v2
PY=/home/yancui/.conda/envs/haruka2/bin/python
RUNNER=$ROOT/bench_kras_xenium/run_haruka_separate_batch_heads.py
CC=$ROOT/bench_kras_xenium/rebuild_condition_only_salient_cc.py
PLOT=$ROOT/bench_real_discovery/plot_sample_batch_umaps.py
OUT=$ROOT/results_codex_channelnorm_ab
mkdir -p "$OUT"

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
  local study=$1 data=$2
  local tag model cc_dir fig
  tag=$(basename "$data" .h5ad)
  model=$OUT/$tag/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0
  cc_dir=$OUT/$tag/haruka_z_s_cellcharter_cc3_condition_only
  fig=$OUT/$tag/sample_batch_diagnostic
  echo "===== TRAIN $study ====="
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" "$RUNNER" \
    --data "$data" --batch_size 4096 --sample_key donor_id --batch_key donor_id \
    --condition_key condition --target_label disease --cell_type_key paper_initial_leiden \
    --gene_likelihood normal --neighbor_input_mode pretransformed \
    --assay_type protein --n_top_genes 0 "${common[@]}"
  echo "===== CONDITION-ONLY CC $study ====="
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" "$CC" \
    --data "$data" --input-dir "$model" --output-dir "$cc_dir" \
    --condition-key condition --condition-value disease --slice-key donor_id
  echo "===== SAMPLE UMAP $study ====="
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" "$PLOT" \
    --data "$data" --cc-dir "$cc_dir" --out-dir "$fig" \
    --study "$study" --sample-key donor_id
}

P=$ROOT/bench_codex_cns_2026/data/processed
run_one "CODEX AD baseline (log1p+global z)" "$P/codex_cns_myeloid_primary_no_aml_no_combat.h5ad"
run_one "CODEX AD channel-norm (p25/p95)"    "$P/codex_cns_myeloid_primary_no_aml_channelnorm.h5ad"
echo "ALL DONE"
