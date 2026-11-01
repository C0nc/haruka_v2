#!/usr/bin/env bash
# Refit of the KRAS model with the same settings as results_real_compgated_nb_final (slice batch, 2000 HVG, NB),
# additionally saving the trained model and 3 shuffled-s null composition shifts (s permuted among treated cells
# within slice x cell type, z kept) for the "does baseline z predict the treatment effect" test.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${1:-4}
ROOT=${HARUKA_ROOT:-$(d=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); while [ ! -f "$d/notebooks/repro.py" ]; do d=$(dirname "$d"); done; echo "$d")}; PY=${HARUKA_PY:-python}
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" "$ROOT/bench_kras_xenium/run_haruka_separate_batch_heads.py" \
  --data "$ROOT/data/mouselung_kras_xenium_clean_noC3.h5ad" --batch_size 32768 \
  --sample_key slice_id --batch_key slice_id --condition_key condition_label --target_label KRASi_treated --cell_type_key celltype \
  --gene_likelihood nb --neighbor_input_mode normalize --assay_type sequencing --n_top_genes 2000 \
  --out_root "$ROOT/results_kras_te_null" --seed 0 --max_epochs 150 \
  --n_latent 32 --n_hidden 256 --n_layers 1 --decoder_layers 2 --k_niche 20 --neighbor_embedding_dim 16 \
  --composition_loss_weight 1 --neighbor_mean_loss_weight 1 --neighbor_expression_mode composition_gated --spatial_weight 100 \
  --background_wasserstein_penalty 0 --zs_independence_penalty 10 --decoder_batch_dim 8 --batch_injection_mode decoder_only \
  --augmentation_mode sparse --num_workers 4 --direct_only --shuffled_s_null 3
echo "DONE $(date)"
