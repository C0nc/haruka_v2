#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate haruka2
cd /home/yancui/haruka_v2/bench_merfish
for S in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 python -u run_haruka.py --data data/simv4_g0.8_r0.3_p0.15_d1_s0.h5ad --out_root results_v5 \
    --seed $S --variants both_sp8 both_mae60
done
