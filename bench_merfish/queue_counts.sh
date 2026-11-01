#!/usr/bin/env bash
# The synthetic-counts 2x2 sweep: 12 datasets x 8 methods, 4 GPUs.
set -uo pipefail
cd "$(dirname "$0")"
GPUS=(3 5 6 7)
i=0
for seed in 0 1 2; do
  for d in 0 0.5 1 2; do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    nohup ./run_one_counts.sh "$d" "$seed" "$gpu" > "logs/cnt_driver_d${d}_s${seed}.log" 2>&1 &
    i=$((i+1))
    while [ "$(jobs -rp | wc -l)" -ge ${#GPUS[@]} ]; do wait -n; done
  done
done
wait
echo "COUNTS SWEEP ALL DONE $(date +%T)"
