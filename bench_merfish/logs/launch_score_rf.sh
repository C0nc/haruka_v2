#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
for K in 32 64; do
  python -u score_all_v3.py --data data/simv4_g0.8_r0.3_p0.15_d1_s0.h5ad \
    --results_root results_rf$K --rep_mode own --out results_rf$K/metrics_rf$K.csv
done
