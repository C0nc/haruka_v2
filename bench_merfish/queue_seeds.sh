#!/usr/bin/env bash
# Wait for the seed-0 sweep to finish, then run seeds 1 and 2.
#
# WHY MULTI-SEED AT ALL. The Haruka-v2 README's central methodological finding is
# that this outcome is BIMODAL across seeds -- the background latent either cleanly
# separates architecture from condition or collapses, with little in between -- so
# a single run and a cross-seed mean are both misleading. Seeds 1 and 2 run a
# reduced delta set (null / moderate / strong) because the intermediate deltas
# only refine the detection curve, which one seed already establishes.
set -uo pipefail
cd "$(dirname "$0")"

GPUS=(4 0 2 3 6)

echo "[queue] waiting for seed-0 jobs to finish"
while pgrep -f "run_one.sh .* 0 " > /dev/null; do sleep 60; done
echo "[queue] seed 0 finished at $(date +%T)"

i=0
for seed in 1 2; do
  for d in 0 1.2 2.4; do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    nohup ./run_one.sh "$d" "$seed" "$gpu" results 400 \
        > "logs/driver_d${d}_s${seed}.log" 2>&1 &
    echo "[queue] launched delta=$d seed=$seed gpu=$gpu"
    i=$((i + 1))
    # Five GPUs, six jobs: block once the fleet is full so the last one does not
    # land on an already-busy card.
    while [ "$(jobs -rp | wc -l)" -ge ${#GPUS[@]} ]; do wait -n; done
  done
done
wait
echo "[queue] ALL SEEDS DONE $(date +%T)"
