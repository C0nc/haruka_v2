#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate haruka2
cd /home/yancui/haruka_v2/bench_merfish
CUDA_VISIBLE_DEVICES=0 exec python -u run_haruka.py --data data/simv4_g0.8_r0.3_p0.15_d1_s0.h5ad \
  --out_root results_v5_sppen --seed 0 \
  --variants both_sppen1 both_sppen5 both_sal8_sppen5
