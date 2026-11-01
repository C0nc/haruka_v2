#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
exec python -u score_all_v3.py --data data/simv3_d1_s0.h5ad --rep_mode own \
     --out results_v3/simv3_d1_s0/metrics_own_gmm.csv
