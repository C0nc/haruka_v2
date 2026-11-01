#!/usr/bin/env bash
# HD MM-1128 regularisation / batch-key grid. Usage: run_fit_hd1128_grid.sh <tag> <batch_key> <zs_lambda> <oneway|twoway> <gpu>
# Everything else as run_fit_hd1128_oneway.sh (mask OFF, spatial 100, 150 epochs, seed 0, 3 shuffled-s nulls).
# batch_key slice_id = donor (one section per donor; donor is NESTED in condition, so the decoder batch term can absorb HD).
set -euo pipefail
TAG=$1; BK=$2; LAM=$3; DIR=$4; export CUDA_VISIBLE_DEVICES=$5
SYM=""; [ "$DIR" = twoway ] && SYM="--symmetric_zs_independence"
ROOT=/home/yancui/haruka_v2; PY=/home/yancui/.conda/envs/haruka2/bin/python; D=/project/zhihuanglab/yancui/haruka_hd
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMPY_MADVISE_HUGEPAGE=0 "$PY" "$ROOT/bench_kras_xenium/run_haruka_separate_batch_heads.py" \
  --data "$D/hd1128_haruka.h5ad" --batch_size 32768 \
  --sample_key slice_id --batch_key "$BK" --condition_key condition --target_label disease --cell_type_key cell_type \
  --gene_likelihood nb --neighbor_input_mode normalize --assay_type sequencing --n_top_genes 0 \
  --out_root "$D/results_1128_$TAG" --seed 0 --max_epochs 150 \
  --n_latent 32 --n_hidden 256 --n_layers 1 --decoder_layers 2 --k_niche 20 --neighbor_embedding_dim 16 \
  --composition_loss_weight 1 --neighbor_mean_loss_weight 1 --neighbor_expression_mode composition_gated --spatial_weight 100 \
  --background_wasserstein_penalty 0 --zs_independence_penalty "$LAM" $SYM \
  --decoder_batch_dim 8 --batch_injection_mode decoder_only \
  --augmentation_mode sparse --num_workers 4 --direct_only --shuffled_s_null 3
echo "DONE $(date)"
