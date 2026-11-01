#!/usr/bin/env bash
# Exercise (old control 4 mice vs old exercise 4 mice), batch key = run A/B (2+2 mice each), sample = mouse section.
# z-s independence 10, spatial weight 100; other settings as the OSKM pair-batch fit. Saves model + 3 shuffled-s nulls.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${1:-4}
ROOT=${HARUKA_ROOT:-$(d=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); while [ ! -f "$d/notebooks/repro.py" ]; do d=$(dirname "$d"); done; echo "$d")}; PY=${HARUKA_PY:-python}
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" "$ROOT/bench_kras_xenium/run_haruka_separate_batch_heads.py" \
  --data ${HARUKA_WORK:-/project/zhihuanglab/yancui}/haruka_exercise/data/exercise_old_full.h5ad --batch_size 32768 \
  --sample_key slice_id --batch_key run_batch --condition_key condition --target_label exercise --cell_type_key cell_type \
  --gene_likelihood nb --neighbor_input_mode normalize --assay_type sequencing --n_top_genes 0 \
  --out_root ${HARUKA_WORK:-/project/zhihuanglab/yancui}/haruka_exercise/results_w10_s100 --seed 0 --max_epochs 150 \
  --n_latent 32 --n_hidden 256 --n_layers 1 --decoder_layers 2 --k_niche 20 --neighbor_embedding_dim 16 \
  --composition_loss_weight 1 --neighbor_mean_loss_weight 1 --neighbor_expression_mode composition_gated --spatial_weight 100 \
  --background_wasserstein_penalty 0 --zs_independence_penalty 10 --decoder_batch_dim 8 --batch_injection_mode decoder_only \
  --augmentation_mode sparse --num_workers 4 --direct_only --shuffled_s_null 3
echo "DONE $(date)"
