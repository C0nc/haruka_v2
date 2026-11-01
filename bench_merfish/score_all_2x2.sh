#!/usr/bin/env bash
# Score every trained 2x2 embedding directory.
#
# The runners derive their output tag with `basename(data).replace("sim_","")`,
# which does not match `sim2x2_*`, so embeddings landed under
# `results_2x2/sim2x2_<tag>/` while run_one_2x2.sh's scoring loop globbed
# `results_2x2/<tag>/` and silently scored nothing. Training is unaffected --
# this recovers the scoring pass without retraining anything.
set -uo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
PY=/home/yancui/.conda/envs/cellcharter/bin/python
mkdir -p logs results_2x2

pids=()
for d in results_2x2/sim2x2_d*_s*; do
  tag=$(basename "$d" | sed 's/^sim2x2_//')
  seed="${tag##*_s}"
  data="data/sim2x2_${tag}.h5ad"
  [ -f "$data" ] || { echo "!! no data for $tag"; continue; }
  for dir in "$d"/*_seed${seed}; do
    [ -f "$dir/meta.json" ] || continue
    m=$(basename "$dir" | sed "s/_seed${seed}\$//")
    out="results_2x2/metrics_${tag}.jsonl"
    $PY score_2x2.py --data "$data" --emb_dir "$dir" --method "$m" --seed "$seed" \
        --out "$out" > "logs/x2_score_${tag}_${m}.log" 2>&1 &
    pids+=($!)
    while [ "$(jobs -rp | wc -l)" -ge 8 ]; do wait -n; done
  done
done
for p in "${pids[@]}"; do wait "$p" || true; done
echo "SCORING DONE $(date +%T): $(cat results_2x2/metrics_*.jsonl 2>/dev/null | wc -l) rows"
