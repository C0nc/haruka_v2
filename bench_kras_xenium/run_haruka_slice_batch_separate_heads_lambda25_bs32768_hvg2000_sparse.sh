#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMBA_NUM_THREADS=8

exec conda run --no-capture-output -n haruka2 python \
  /home/yancui/haruka_v2/bench_kras_xenium/run_haruka_separate_batch_heads.py \
  --data /home/yancui/mouselung_kras_xenium_clean_noC3.h5ad \
  --out_root /home/yancui/haruka_v2/bench_kras_xenium/results_slice_batch_separate_heads_lambda25_bs32768_hvg2000_sparse \
  --seed 0 \
  --n_top_genes 2000 \
  --max_epochs 150 \
  --batch_size 32768 \
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
  --augmentation_mode sparse \
  --num_workers 4 \
  --direct_only
