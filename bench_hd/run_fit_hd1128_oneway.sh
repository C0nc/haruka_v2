#!/usr/bin/env bash
# HD MERFISH MM-1128 (4 control / 5 HD donors, 1,128 genes; panel with per-cell CAG / inclusion / gammaH2AX GT). Usage: run_fit_hd1128.sh <gpu>
# batch key = experimental batch (Aug 2C/3H, Sep 2C/2H: both arms in each batch). All cells annotated, no context cells.
# z/s independence ONE-WAY (z <- s only, weight 10; same as all pre-HD fits); salient-mean mask OFF. Spatial weight 100.
# Saves model + 3 shuffled-s nulls.
set -euo pipefail
V=1128; export CUDA_VISIBLE_DEVICES=${1:-6}
ROOT=/home/yancui/haruka_v2; PY=/home/yancui/.conda/envs/haruka2/bin/python; D=/project/zhihuanglab/yancui/haruka_hd
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMPY_MADVISE_HUGEPAGE=0 "$PY" "$ROOT/bench_kras_xenium/run_haruka_separate_batch_heads.py" \
  --data "$D/hd1128_haruka.h5ad" --batch_size 32768 \
  --sample_key slice_id --batch_key batch --condition_key condition --target_label disease --cell_type_key cell_type \
  --gene_likelihood nb --neighbor_input_mode normalize --assay_type sequencing --n_top_genes 0 \
  --out_root "$D/results_1128_oneway" --seed 0 --max_epochs 150 \
  --n_latent 32 --n_hidden 256 --n_layers 1 --decoder_layers 2 --k_niche 20 --neighbor_embedding_dim 16 \
  --composition_loss_weight 1 --neighbor_mean_loss_weight 1 --neighbor_expression_mode composition_gated --spatial_weight 100 \
  --background_wasserstein_penalty 0 --zs_independence_penalty 10 \
  --decoder_batch_dim 8 --batch_injection_mode decoder_only \
  --augmentation_mode sparse --num_workers 4 --direct_only --shuffled_s_null 3
echo "DONE $(date)"
