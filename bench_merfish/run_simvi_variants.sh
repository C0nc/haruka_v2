#!/usr/bin/env bash
# Train + score the two SIMVI-faithful loss ablations of the 2x2.
#
#   both_simviloss       spatial_invariance_penalty = 0   (SIMVI's exact reg set:
#                                                          kl_spatial_weight + lam_mi)
#   both_simviloss_nome  that, plus me_weight = 0          (no COVET/ME head)
#
# The GAT input deliberately does NOT change: `_spatial_pool` already feeds it
# `encoder(log1p(neighbour counts))[:, -n_spatial:]`, which is exactly SIMVI's
# `q_m[:, -n_spatial:]`. Nothing raw reaches the GAT.
#
#   ./run_simvi_variants.sh <delta> <seed> <gpu>
set -uo pipefail
cd "$(dirname "$0")"
d="${1:?delta}"; seed="${2:-0}"; gpu="${3:-0}"; OUT="${4:-results_2x2}"; EPOCHS="${5:-400}"

export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="$gpu"
PY_HK=/home/yancui/.conda/envs/haruka2/bin/python
PY_CC=/home/yancui/.conda/envs/cellcharter/bin/python

data="data/sim2x2_d${d}_s${seed}.h5ad"
tag="d${d}_s${seed}"
[ -f "$data" ] || { echo "!! missing $data"; exit 1; }
mkdir -p logs "$OUT"
echo "[$tag] simvi-variants gpu=$gpu start $(date +%T)"

$PY_HK run_haruka.py --data "$data" --seed "$seed" --max_epochs "$EPOCHS" \
    --variants both_simviloss both_simviloss_nome --out_root "$OUT" \
    > "logs/simvi_${tag}.log" 2>&1 || { echo "[$tag] !! train failed"; exit 1; }
echo "[$tag] trained $(date +%T)"

# run_haruka derives its own tag as basename(data) minus "sim_", which leaves the
# "sim2x2_" prefix intact -- matching where every other 2x2 result already lives.
n=0
for m in both_simviloss both_simviloss_nome; do
  dir="$OUT/sim2x2_${tag}/haruka_${m}_seed${seed}"
  [ -f "$dir/meta.json" ] || { echo "[$tag] !! no embeddings for $m"; continue; }
  $PY_CC score_2x2.py --data "$data" --emb_dir "$dir" --method "haruka_$m" \
      --seed "$seed" --out "$OUT/metrics_${tag}.jsonl" \
      > "logs/simvi_score_${tag}_${m}.log" 2>&1 && n=$((n+1)) \
      || echo "[$tag] !! score failed: $m"
done
# Guard against the silent-empty-glob failure that cost a full sweep earlier.
[ "$n" -eq 2 ] || echo "[$tag] !! WARNING scored $n/2 variants"
echo "[$tag] ALL DONE $(date +%T) ($n/2 scored)"
