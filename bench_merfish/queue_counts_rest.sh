#!/usr/bin/env bash
# Seeds 1-2 of the synthetic-counts sweep, on the GPUs freed by stopping the
# SIMVI-loss ablation on real tissue (that ablation already answered its question
# consistently across 6 rows / 3 deltas: removing `spatial_invariance_penalty`
# does NOT fix the `both` latent inversion).
#
# Runs alongside the four seed-0 datasets already in flight on GPUs 3/5/6/7.
set -uo pipefail
cd "$(dirname "$0")"
GPUS=(0 2 4)
i=0
for seed in 1 2; do
  for d in 0 0.5 1 2; do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    nohup ./run_one_counts.sh "$d" "$seed" "$gpu" \
        > "logs/cnt_driver_d${d}_s${seed}.log" 2>&1 &
    echo "launched delta=$d seed=$seed gpu=$gpu"
    i=$((i + 1))
    while [ "$(jobs -rp | wc -l)" -ge ${#GPUS[@]} ]; do wait -n; done
  done
done
wait
echo "COUNTS SEEDS 1-2 DONE $(date +%T)"
