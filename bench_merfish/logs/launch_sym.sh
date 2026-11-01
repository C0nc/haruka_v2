#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate haruka2
cd /home/yancui/haruka_v2/bench_merfish
while ps -eo cmd | grep -q "[r]un_haruka.py"; do sleep 60; done
CUDA_VISIBLE_DEVICES=0 python -u run_haruka.py --data data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad \
  --out_root results_priv --seed 0 --max_epochs 500 \
  --variants both_fully_symmetric
echo SYM_DONE
