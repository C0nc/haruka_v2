#!/usr/bin/env bash
# Full benchmark sweep. One conda env per method family, invoked as separate
# processes rather than threads: the envs are mutually incompatible (the patched
# contrastiveVI cannot coexist with the upstream one it is being compared
# against), and separate processes also keep each method's OpenBLAS/torch thread
# state isolated.
#
#   ./run_all.sh                    # everything, seed 0
#   SEEDS="0 1 2" ./run_all.sh      # multi-seed
#
set -uo pipefail
cd "$(dirname "$0")"

# This machine has 256 cores and OpenBLAS was built for a maximum of 128 threads;
# left uncapped, sklearn's PCA/GMM inside the scorer dies with
# "Program is Terminated. Because you tried to allocate too many memory regions".
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

GPU="${GPU:-4}"
SEEDS="${SEEDS:-0}"
DELTAS="${DELTAS:-0 0.3 0.6 1.2 2.4}"
EPOCHS="${EPOCHS:-400}"
OUT="${OUT:-results}"
HARUKA_VARIANTS="${HARUKA_VARIANTS:-vanilla background both}"

PY_CC=/home/yancui/.conda/envs/cellcharter/bin/python
PY_HK=/home/yancui/.conda/envs/haruka2/bin/python
PY_OT=/home/yancui/.conda/envs/vio/bin/python

export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p logs "$OUT"

for seed in $SEEDS; do
for d in $DELTAS; do
  data="data/sim_d${d}_s${seed}.h5ad"
  tag="d${d}_s${seed}"
  [ -f "$data" ] || { echo "!! missing $data -- run build_sim.py first"; continue; }

  echo "######## $tag ########"
  $PY_CC run_baselines.py --data "$data" --seed "$seed" --max_epochs "$EPOCHS" \
      --methods pca cellcharter contrastivevi contrastivevi_cc \
      --out_root "$OUT" > "logs/base_${tag}.log" 2>&1 \
      || echo "!! baselines failed on $tag (see logs/base_${tag}.log)"

  $PY_OT run_cinemaot.py --data "$data" --seed "$seed" --out_root "$OUT" \
      > "logs/cinemaot_${tag}.log" 2>&1 \
      || echo "!! cinemaot failed on $tag (see logs/cinemaot_${tag}.log)"

  $PY_HK run_haruka.py --data "$data" --seed "$seed" --max_epochs "$EPOCHS" \
      --variants $HARUKA_VARIANTS --out_root "$OUT" \
      > "logs/haruka_${tag}.log" 2>&1 \
      || echo "!! haruka failed on $tag (see logs/haruka_${tag}.log)"

  # Score whatever actually produced embeddings. A method that crashed is simply
  # absent from metrics.jsonl rather than silently scored as zero.
  for dir in "$OUT/$tag"/*_seed${seed}; do
    [ -f "$dir/meta.json" ] || continue
    m=$(basename "$dir" | sed "s/_seed${seed}\$//")
    $PY_CC score.py --data "$data" --emb_dir "$dir" --method "$m" --seed "$seed" \
        --out "$OUT/metrics.jsonl" > "logs/score_${tag}_${m}.log" 2>&1 \
        || echo "!! score failed: $tag/$m"
  done
  echo "---- $tag scored ----"
done
done

echo "done -> $OUT/metrics.jsonl"
