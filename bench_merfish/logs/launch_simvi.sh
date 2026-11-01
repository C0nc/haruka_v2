#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate simvi
cd /home/yancui/haruka_v2/bench_merfish
export CUDA_VISIBLE_DEVICES=5
exec python -u run_simvi.py --data data/simv3_d1_s0.h5ad --out_root results_v3
