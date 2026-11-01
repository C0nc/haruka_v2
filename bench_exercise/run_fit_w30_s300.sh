#!/usr/bin/env bash
# Exercise (old control 4 mice vs old exercise 4 mice), batch key = run A/B (2+2 mice each), sample = mouse section.
# z-s independence 30, spatial weight 300; other settings as the OSKM pair-batch fit. Saves model + 3 shuffled-s nulls.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${1:-4}
ROOT=/home/yancui/haruka_v2; PY=/home/yancui/.conda/envs/haruka2/bin/python
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" "$ROOT/bench_kras_xenium/run_haruka_separate_batch_heads.py" \
  --data /project/zhihuanglab/yancui/haruka_exercise/data/exercise_old_full.h5ad --batch_size 32768 \
  --sample_key slice_id --batch_key run_batch --condition_key condition --target_label exercise --cell_type_key cell_type \
  --gene_likelihood nb --neighbor_input_mode normalize --assay_type sequencing --n_top_genes 0 \
  --out_root /project/zhihuanglab/yancui/haruka_exercise/results_w30_s300 --seed 0 --max_epochs 150 \
  --n_latent 32 --n_hidden 256 --n_layers 1 --decoder_layers 2 --k_niche 20 --neighbor_embedding_dim 16 \
  --composition_loss_weight 1 --neighbor_mean_loss_weight 1 --neighbor_expression_mode composition_gated --spatial_weight 300 \
  --background_wasserstein_penalty 0 --zs_independence_penalty 30 --decoder_batch_dim 8 --batch_injection_mode decoder_only \
  --augmentation_mode sparse --num_workers 4 --direct_only --shuffled_s_null 3
echo "DONE $(date)"
