#!/usr/bin/env bash
# All methods on one SYNTHETIC-COUNTS dataset, then score with the 2x2 matrix.
#   ./run_one_counts.sh <delta> <seed> <gpu> [out_root] [epochs]
#
# Same method set and same scorer as the real-tissue 2x2 -- only the data differ,
# so the two benchmarks are directly comparable. `score_2x2.py` needs no change:
# `build_sim_counts.py` writes the same four `gt_*` columns and the same
# `uns["sim2x2"]` schema.
set -uo pipefail
cd "$(dirname "$0")"
d="${1:?delta}"; seed="${2:-0}"; gpu="${3:-0}"; OUT="${4:-results_counts}"; EPOCHS="${5:-150}"; BATCH="${BATCH:-2048}"
HARUKA_VARIANTS="${HARUKA_VARIANTS:-both background vanilla both_simviloss}"

export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="$gpu"
PY_CC=/home/yancui/.conda/envs/cellcharter/bin/python
PY_HK=/home/yancui/.conda/envs/haruka2/bin/python
PY_OT=/home/yancui/.conda/envs/vio/bin/python

data="data/simcnt_d${d}_s${seed}.h5ad"
tag="d${d}_s${seed}"
[ -f "$data" ] || { echo "!! missing $data"; exit 1; }
mkdir -p logs "$OUT"
# EPOCHS/BATCH default to 150/2048, not the repo's 400/512. That config was tuned
# on a 600-cell lattice where 400 epochs = 800 total optimiser steps; at 72k cells
# the same setting is 56,400 steps per fit -- 70x more training for the same
# nominal epoch count, on a model with 32-dim latents and 300 genes that was
# running at 15-25% GPU utilisation. 150 epochs at batch 2048 is ~5,300 steps,
# still ~6x the original config's total, and about 10x faster.
echo "[$tag] counts-bench gpu=$gpu epochs=$EPOCHS batch=$BATCH start $(date +%T)"

$PY_HK run_haruka.py --data "$data" --seed "$seed" --max_epochs "$EPOCHS" --batch_size "$BATCH" \
    --variants $HARUKA_VARIANTS --out_root "$OUT" \
    > "logs/cnt_haruka_${tag}.log" 2>&1 || echo "[$tag] !! haruka failed"
$PY_CC run_baselines.py --data "$data" --seed "$seed" --max_epochs "$EPOCHS" --batch_size "$BATCH" \
    --methods pca cellcharter contrastivevi contrastivevi_cc --out_root "$OUT" \
    > "logs/cnt_base_${tag}.log" 2>&1 || echo "[$tag] !! baselines failed"
$PY_OT run_cinemaot.py --data "$data" --seed "$seed" --out_root "$OUT" \
    > "logs/cnt_cinemaot_${tag}.log" 2>&1 || echo "[$tag] !! cinemaot failed"
echo "[$tag] trained $(date +%T)"

# run_haruka/run_baselines strip only "sim_" from the basename, so the output tag
# keeps the "simcnt_" prefix. Glob both spellings and COUNT what was scored -- a
# silently-empty glob reporting success is what cost an entire earlier sweep.
n=0; pids=()
for dir in "$OUT/$tag"/*_seed${seed} "$OUT/simcnt_${tag}"/*_seed${seed}; do
  [ -f "$dir/meta.json" ] || continue
  m=$(basename "$dir" | sed "s/_seed${seed}\$//")
  $PY_CC score_2x2.py --data "$data" --emb_dir "$dir" --method "$m" --seed "$seed" \
      --out "$OUT/metrics_${tag}.jsonl" > "logs/cnt_score_${tag}_${m}.log" 2>&1 &
  pids+=($!); n=$((n+1))
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do wait -n; done
done
for p in "${pids[@]}"; do wait "$p" || echo "[$tag] !! a score job failed"; done
[ "$n" -ge 8 ] || echo "[$tag] !! WARNING only $n embedding dirs found"
echo "[$tag] ALL DONE $(date +%T) ($n scored)"
