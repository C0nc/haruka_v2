#!/usr/bin/env bash
# Haruka default on the best-quality SPAC subset; halo bins are neighbours only.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${1:-5}
ROOT=/home/yancui/haruka_v2; PY=/home/yancui/.conda/envs/haruka2/bin/python
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" "$ROOT/bench_kras_xenium/run_haruka_separate_batch_heads.py" \
  --data "$ROOT/bench_spacseq/data/spac_lung_best_quality_5targets_halo.h5ad" --batch_size 2048 \
  --sample_key slice_id --batch_key slice_id --condition_key condition --target_label disease --exclude_label halo \
  --cell_type_key cell_type --gene_likelihood nb --neighbor_input_mode normalize --assay_type sequencing --n_top_genes 0 \
  --out_root "$ROOT/results_spac_best_quality" --seed 0 --max_epochs 150 \
  --n_latent 32 --n_hidden 256 --n_layers 1 --decoder_layers 2 --k_niche 20 --neighbor_embedding_dim 16 \
  --composition_loss_weight 1 --neighbor_mean_loss_weight 1 --neighbor_expression_mode composition_gated --spatial_weight 100 \
  --background_wasserstein_penalty 0 --zs_independence_penalty 10 --decoder_batch_dim 8 --batch_injection_mode decoder_only \
  --augmentation_mode sparse --num_workers 4 --direct_only
echo "DONE $(date)"
