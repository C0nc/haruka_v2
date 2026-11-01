#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
for R in results_syn results_k128; do
  python -u score_all_v3.py --data data/simv4_g0.8_r0.3_p0.15_d1_s0.h5ad \
    --results_root $R --rep_mode own --out $R/metrics.csv
done
