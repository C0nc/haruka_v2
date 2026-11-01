#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate haruka2
cd /home/yancui/haruka_v2/bench_merfish
CUDA_VISIBLE_DEVICES=0 exec python -u run_haruka.py --data data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad \
  --out_root results_easy_ep500 --seed 0 --max_epochs 500 \
  --variants both_sal8_timebatch_ep500_fixed both_sal8_timebatch_ep500_sched
