#!/usr/bin/env bash
# All 12 datasets through the two SIMVI-loss variants, 4 at a time on free GPUs.
set -uo pipefail
cd "$(dirname "$0")"
GPUS=(0 2 4 6)
i=0
for seed in 0 1 2; do
  for d in 0 0.6 1.2 2.4; do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    nohup ./run_simvi_variants.sh "$d" "$seed" "$gpu" > "logs/simvi_driver_d${d}_s${seed}.log" 2>&1 &
    i=$((i+1))
    while [ "$(jobs -rp | wc -l)" -ge ${#GPUS[@]} ]; do wait -n; done
  done
done
wait
echo "SIMVI VARIANTS ALL DONE $(date +%T)"
