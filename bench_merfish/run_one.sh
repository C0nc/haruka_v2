#!/usr/bin/env bash
# Run every method on ONE simulated dataset, then score them all.
#   ./run_one.sh <delta> <seed> <gpu> [out_root] [epochs]
#
# One conda env per method family, invoked as separate processes rather than
# threads: the envs are mutually incompatible (the patched contrastiveVI cannot
# coexist with the upstream one it is being compared against), and separate
# processes keep each method's OpenBLAS/torch thread state isolated.
set -uo pipefail
cd "$(dirname "$0")"

d="${1:?delta}"; seed="${2:-0}"; gpu="${3:-4}"; OUT="${4:-results}"; EPOCHS="${5:-400}"
HARUKA_VARIANTS="${HARUKA_VARIANTS:-vanilla background both}"

# This machine has 256 cores and OpenBLAS was built for a maximum of 128 threads;
# left uncapped, sklearn's PCA/GMM inside the scorer dies with
# "Program is Terminated. Because you tried to allocate too many memory regions".
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="$gpu"

PY_CC=/home/yancui/.conda/envs/cellcharter/bin/python
PY_HK=/home/yancui/.conda/envs/haruka2/bin/python
PY_OT=/home/yancui/.conda/envs/vio/bin/python

data="data/sim_d${d}_s${seed}.h5ad"
tag="d${d}_s${seed}"
[ -f "$data" ] || { echo "!! missing $data"; exit 1; }
mkdir -p logs "$OUT"

echo "[$tag] gpu=$gpu epochs=$EPOCHS  start $(date +%T)"

$PY_CC run_baselines.py --data "$data" --seed "$seed" --max_epochs "$EPOCHS" \
    --methods pca cellcharter contrastivevi contrastivevi_cc --out_root "$OUT" \
    > "logs/base_${tag}.log" 2>&1 || echo "[$tag] !! baselines failed"
echo "[$tag] baselines done $(date +%T)"

$PY_OT run_cinemaot.py --data "$data" --seed "$seed" --out_root "$OUT" \
    > "logs/cinemaot_${tag}.log" 2>&1 || echo "[$tag] !! cinemaot failed"
echo "[$tag] cinemaot done $(date +%T)"

$PY_HK run_haruka.py --data "$data" --seed "$seed" --max_epochs "$EPOCHS" \
    --variants $HARUKA_VARIANTS --out_root "$OUT" \
    > "logs/haruka_${tag}.log" 2>&1 || echo "[$tag] !! haruka failed"
echo "[$tag] haruka done $(date +%T)"

# Score whatever actually produced embeddings, a few at a time. A method that
# crashed is simply absent from metrics.jsonl rather than silently scored as zero.
# Scores go to a per-dataset file so parallel deltas never interleave writes.
pids=()
for dir in "$OUT/$tag"/*_seed${seed}; do
  [ -f "$dir/meta.json" ] || continue
  m=$(basename "$dir" | sed "s/_seed${seed}\$//")
  $PY_CC score.py --data "$data" --emb_dir "$dir" --method "$m" --seed "$seed" \
      --out "$OUT/metrics_${tag}.jsonl" > "logs/score_${tag}_${m}.log" 2>&1 &
  pids+=($!)
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do wait -n; done
done
for p in "${pids[@]}"; do wait "$p" || echo "[$tag] !! a score job failed"; done

echo "[$tag] ALL DONE $(date +%T)  -> $OUT/metrics_${tag}.jsonl"
