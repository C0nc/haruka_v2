#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate haruka2
cd /home/yancui/haruka_v2/bench_merfish
export CUDA_VISIBLE_DEVICES=6
exec python -u run_haruka.py --data data/simv3_d1_s0.h5ad --out_root results_v3 \
     --variants both_sal8 both_sal4 both_sp8 both_wass2
