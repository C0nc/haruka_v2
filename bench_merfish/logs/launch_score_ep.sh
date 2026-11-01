#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
exec python -u score_all_v3.py --data data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad \
  --results_root results_easy_ep --rep_mode own --out results_easy_ep/metrics.csv
