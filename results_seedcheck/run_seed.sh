#!/usr/bin/env bash
# Seed robustness for the plaque-radial z-vs-s result: default Haruka, only --seed changes.
# usage: run_seed.sh <starmap|codex> <seed> <gpu>
set -euo pipefail
D=$1; SEED=$2; export CUDA_VISIBLE_DEVICES=$3
ROOT=${HARUKA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)}; PY=${HARUKA_PY:-python}
RUNNER=$ROOT/bench_kras_xenium/run_haruka_separate_batch_heads.py
common=(--seed "$SEED" --max_epochs 150 --n_latent 32 --n_hidden 256 --n_layers 1 --decoder_layers 2
  --k_niche 20 --neighbor_embedding_dim 16 --composition_loss_weight 1 --neighbor_mean_loss_weight 1
  --neighbor_expression_mode composition_gated --spatial_weight 100 --background_wasserstein_penalty 0
  --zs_independence_penalty 10 --decoder_batch_dim 8 --batch_injection_mode decoder_only
  --augmentation_mode sparse --num_workers 4 --direct_only --batch_size 4096 --condition_key condition
  --target_label disease --cell_type_key cell_type --gene_likelihood normal --neighbor_input_mode pretransformed --n_top_genes 0)
if [ "$D" = starmap ]; then
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" "$RUNNER" --data $ROOT/bench_real_discovery/data/ad.h5ad \
    --sample_key slice_id --batch_key slice_id --assay_type sequencing --out_root $ROOT/results_seedcheck/starmap "${common[@]}"
else
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" "$RUNNER" --data $ROOT/bench_codex_cns_2026/wholeslice/codex_wholeslice_all7_no_oligo.h5ad \
    --sample_key donor_id --batch_key donor_id --assay_type protein --out_root $ROOT/results_seedcheck/codex "${common[@]}"
fi
echo "DONE $D seed $SEED"
