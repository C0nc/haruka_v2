#!/usr/bin/env bash
# Run every method on one 2x2 dataset, then score with the representation-by-truth
# matrix.  ./run_one_2x2.sh <delta> <seed> <gpu> [out_root] [epochs]
#
# `haruka_both` is the primary configuration -- it is the only one that emits all
# four latents and so the only one that can fill the whole matrix. `vanilla` and
# `background` are kept as ablations: they show what is lost by removing the
# spatial branch entirely, and by attaching it to the background encoder only.
set -uo pipefail
cd "$(dirname "$0")"

d="${1:?delta}"; seed="${2:-0}"; gpu="${3:-4}"; OUT="${4:-results_2x2}"; EPOCHS="${5:-400}"
HARUKA_VARIANTS="${HARUKA_VARIANTS:-both background vanilla}"

export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="$gpu"

PY_CC=/home/yancui/.conda/envs/cellcharter/bin/python
PY_HK=/home/yancui/.conda/envs/haruka2/bin/python
PY_OT=/home/yancui/.conda/envs/vio/bin/python

data="data/sim2x2_d${d}_s${seed}.h5ad"
tag="d${d}_s${seed}"
[ -f "$data" ] || { echo "!! missing $data"; exit 1; }
mkdir -p logs "$OUT"
echo "[$tag] gpu=$gpu epochs=$EPOCHS start $(date +%T)"

$PY_HK run_haruka.py --data "$data" --seed "$seed" --max_epochs "$EPOCHS" \
    --variants $HARUKA_VARIANTS --out_root "$OUT" \
    > "logs/x2_haruka_${tag}.log" 2>&1 || echo "[$tag] !! haruka failed"
echo "[$tag] haruka done $(date +%T)"

$PY_CC run_baselines.py --data "$data" --seed "$seed" --max_epochs "$EPOCHS" \
    --methods pca cellcharter contrastivevi contrastivevi_cc --out_root "$OUT" \
    > "logs/x2_base_${tag}.log" 2>&1 || echo "[$tag] !! baselines failed"
echo "[$tag] baselines done $(date +%T)"

$PY_OT run_cinemaot.py --data "$data" --seed "$seed" --out_root "$OUT" \
    > "logs/x2_cinemaot_${tag}.log" 2>&1 || echo "[$tag] !! cinemaot failed"
echo "[$tag] cinemaot done $(date +%T)"

pids=()
for dir in "$OUT/$tag"/*_seed${seed} "$OUT/sim2x2_$tag"/*_seed${seed}; do
  [ -f "$dir/meta.json" ] || continue
  m=$(basename "$dir" | sed "s/_seed${seed}\$//")
  $PY_CC score_2x2.py --data "$data" --emb_dir "$dir" --method "$m" --seed "$seed" \
      --out "$OUT/metrics_${tag}.jsonl" > "logs/x2_score_${tag}_${m}.log" 2>&1 &
  pids+=($!)
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do wait -n; done
done
for p in "${pids[@]}"; do wait "$p" || echo "[$tag] !! a score job failed"; done

echo "[$tag] ALL DONE $(date +%T) -> $OUT/metrics_${tag}.jsonl"
