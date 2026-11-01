#!/usr/bin/env bash
# Seeds 1 and 2 of the 2x2 sweep, launched once seed 0 clears.
#
# Multi-seed is not optional here. The Haruka-v2 README's own headline
# methodological finding is that this outcome is BIMODAL across seeds -- the
# background latent either separates architecture from condition or collapses,
# with little in between -- so a single run and a cross-seed mean are both
# misleading. `report_2x2.py` prints the seeds individually for that reason.
set -uo pipefail
cd "$(dirname "$0")"

GPUS=(0 2 4 5)
DELTAS="${DELTAS:-0 0.6 1.2 2.4}"

echo "[queue2x2] waiting for the seed-0 sweep"
while pgrep -f "run_one_2x2.sh [0-9.]* 0 " > /dev/null; do sleep 60; done
echo "[queue2x2] seed 0 clear at $(date +%T)"

i=0
for seed in 1 2; do
  for d in $DELTAS; do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    nohup ./run_one_2x2.sh "$d" "$seed" "$gpu" results_2x2 400 \
        > "logs/x2_driver_d${d}_s${seed}.log" 2>&1 &
    echo "[queue2x2] launched delta=$d seed=$seed gpu=$gpu"
    i=$((i + 1))
    while [ "$(jobs -rp | wc -l)" -ge ${#GPUS[@]} ]; do wait -n; done
  done
done
wait
echo "[queue2x2] ALL SEEDS DONE $(date +%T)"
