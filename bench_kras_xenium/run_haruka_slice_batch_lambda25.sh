#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMBA_NUM_THREADS=8

DATA=/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad
OUT=/home/yancui/haruka_v2/bench_kras_xenium/results_slice_batch_lambda25
SCRIPT=/home/yancui/haruka_v2/bench_mouse_brain_scviva/run_haruka_shared_decoder.py

exec conda run --no-capture-output -n haruka2 python "$SCRIPT" \
  --data "$DATA" \
  --out_root "$OUT" \
  --seed 0 \
  --max_epochs 150 \
  --batch_size 8192 \
  --n_latent 32 \
  --n_hidden 256 \
  --n_layers 1 \
  --decoder_layers 2 \
  --k_niche 20 \
  --neighbor_embedding_dim 16 \
  --composition_loss_weight 1 \
  --neighbor_mean_loss_weight 1 \
  --spatial_weight 100 \
  --background_wasserstein_penalty 0 \
  --zs_independence_penalty 25 \
  --batch_key slice_id \
  --batch_injection_mode decoder_only \
  --decoder_batch_dim 8 \
  --sample_key slice_id \
  --condition_key condition_label \
  --target_label KRASi_treated \
  --cell_type_key celltype \
  --gene_likelihood poisson \
  --neighbor_input_mode normalize \
  --augmentation_mode dense \
  --num_workers 4 \
  --direct_only
