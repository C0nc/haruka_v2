#!/usr/bin/env bash
# Haruka default (composition-gated, lambda=10, decoder-only donor batch) on a whole-slice CODEX variant.
# usage: run_wholeslice_fit.sh <variant: all7_no_oligo|full3> <gpu>
set -euo pipefail
V=$1; export CUDA_VISIBLE_DEVICES=$2
ROOT=${HARUKA_ROOT:-$(d=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); while [ ! -f "$d/notebooks/repro.py" ]; do d=$(dirname "$d"); done; echo "$d")}
PY=${HARUKA_PY:-python}
RUNNER=$ROOT/bench_kras_xenium/run_haruka_separate_batch_heads.py
CC=$ROOT/bench_kras_xenium/rebuild_condition_only_salient_cc.py
PLOT=$ROOT/bench_real_discovery/plot_sample_batch_umaps.py
OUT=$ROOT/results_codex_wholeslice
DATA=$ROOT/bench_codex_cns_2026/wholeslice/codex_wholeslice_$V.h5ad
mkdir -p "$OUT"
tag=codex_wholeslice_$V
model=$OUT/$tag/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0
cc_dir=$OUT/$tag/haruka_z_s_cellcharter_cc3_condition_only
T=(env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1)
echo "===== TRAIN $V $(date) ====="
"${T[@]}" "$PY" "$RUNNER" --data "$DATA" --batch_size 4096 --sample_key donor_id --batch_key donor_id \
  --condition_key condition --target_label disease --cell_type_key cell_type \
  --gene_likelihood normal --neighbor_input_mode pretransformed --assay_type protein --n_top_genes 0 \
  --out_root "$OUT" --seed 0 --max_epochs 150 \
  --n_latent 32 --n_hidden 256 --n_layers 1 --decoder_layers 2 \
  --k_niche 20 --neighbor_embedding_dim 16 \
  --composition_loss_weight 1 --neighbor_mean_loss_weight 1 \
  --neighbor_expression_mode composition_gated --spatial_weight 100 \
  --background_wasserstein_penalty 0 --zs_independence_penalty 10 \
  --decoder_batch_dim 8 --batch_injection_mode decoder_only \
  --augmentation_mode sparse --num_workers 4 --direct_only
echo "===== CONDITION-ONLY CC $V $(date) ====="
"${T[@]}" "$PY" "$CC" --data "$DATA" --input-dir "$model" --output-dir "$cc_dir" \
  --condition-key condition --condition-value disease --slice-key donor_id
echo "===== SAMPLE UMAP $V $(date) ====="
"${T[@]}" "$PY" "$PLOT" --data "$DATA" --cc-dir "$cc_dir" --out-dir "$OUT/$tag/sample_batch_diagnostic" \
  --study "CODEX AD whole-slice $V" --sample-key donor_id
echo "DONE $V $(date)"
