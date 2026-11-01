#!/usr/bin/env bash
# HD MERFISH (human frontal cortex, MM-188): control vs HD. Usage: run_fit_hd.sh <balanced|all> <gpu>
# batch key = experimental batch (balanced: June/July/Sep each 2 control + 2 HD; all: + HD-only Aug batch).
# Unannotated cells are neighbours only (--exclude_label context).
# Changes vs the earlier fits: z/s independence in BOTH directions (weight 10); salient-mean mask OFF (it collapsed s in the first HD fits).
# (encoder s mean pushed to ~0 on control cells). Spatial weight 100.
# Saves model + 3 shuffled-s nulls.
set -euo pipefail
V=${1:?balanced|all}; export CUDA_VISIBLE_DEVICES=${2:-6}
ROOT=/home/yancui/haruka_v2; PY=/home/yancui/.conda/envs/haruka2/bin/python; D=/project/zhihuanglab/yancui/haruka_hd
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMPY_MADVISE_HUGEPAGE=0 "$PY" "$ROOT/bench_kras_xenium/run_haruka_separate_batch_heads.py" \
  --data "$D/hd_fit_${V}.h5ad" --batch_size 32768 \
  --sample_key slice_id --batch_key batch --condition_key condition --target_label disease --exclude_label context --cell_type_key cell_type \
  --gene_likelihood nb --neighbor_input_mode normalize --assay_type sequencing --n_top_genes 0 \
  --out_root "$D/results_${V}_sym" --seed 0 --max_epochs 150 \
  --n_latent 32 --n_hidden 256 --n_layers 1 --decoder_layers 2 --k_niche 20 --neighbor_embedding_dim 16 \
  --composition_loss_weight 1 --neighbor_mean_loss_weight 1 --neighbor_expression_mode composition_gated --spatial_weight 100 \
  --background_wasserstein_penalty 0 --zs_independence_penalty 10 --symmetric_zs_independence \
  --decoder_batch_dim 8 --batch_injection_mode decoder_only \
  --augmentation_mode sparse --num_workers 4 --direct_only --shuffled_s_null 3
echo "DONE $(date)"
