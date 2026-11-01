#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate haruka2
cd /home/yancui/haruka_v2/bench_merfish
CUDA_VISIBLE_DEVICES=0 exec python -u run_haruka.py --data data/simv4_g0.8_r0.3_p0.15_c0.6_d1_s0.h5ad \
  --out_root results_v6 --seed 0 --variants both_sal8 both_mae60
