#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
# wait for the baseline runner to exit
while ps -eo cmd | grep -q "[r]un_baselines_2x2"; do sleep 30; done
echo "=== baselines done, scoring ==="
python -u score_all_v3.py --data data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad \
  --results_root results_easy_base --rep_mode own \
  --out results_easy_base/metrics_own_gmm.csv
echo "=== metrics figure ==="
python -u plot_metrics_easy.py --out results_easy_base/metrics_easy.png
echo "FINISH_ALL_DONE"
