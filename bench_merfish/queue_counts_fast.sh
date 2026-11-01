#!/usr/bin/env bash
# The full synthetic-counts sweep at the corrected training budget.
#
#   150 epochs at batch 2048  (~5,300 optimiser steps per fit)
#   was: 400 epochs at batch 512 (~56,400 steps per fit)
#
# The repo's 400/512 was tuned on a 600-cell lattice, where 400 epochs is 800
# total steps. Epoch counts do not transfer across dataset sizes -- step counts
# do. At 72k cells the same nominal setting was 70x more training than the config
# was ever validated at, on tiny models (32-dim latents, 300 genes) that sat at
# 15-25% GPU utilisation while the CPU-side dataloader was the real bottleneck.
#
# All 12 datasets run concurrently, one per GPU across 7 cards (GPU 1 is occupied
# by another user), so this is one wave rather than three.
set -uo pipefail
cd "$(dirname "$0")"
GPUS=(0 2 3 4 5 6 7)
i=0
for seed in 0 1 2; do
  for d in 0 0.5 1 2; do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    nohup ./run_one_counts.sh "$d" "$seed" "$gpu" results_counts 150 \
        > "logs/cnt_driver_d${d}_s${seed}.log" 2>&1 &
    echo "launched delta=$d seed=$seed gpu=$gpu"
    i=$((i + 1))
    while [ "$(jobs -rp | wc -l)" -ge ${#GPUS[@]} ]; do wait -n; done
  done
done
wait
echo "COUNTS SWEEP (fast) ALL DONE $(date +%T)"
