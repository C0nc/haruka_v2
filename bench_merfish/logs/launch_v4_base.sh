#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
export CUDA_VISIBLE_DEVICES=7
exec python -u run_baselines_2x2.py --data data/simv4_g0.8_d1_s0.h5ad \
     --out_root results_v4 --methods pca cellcharter contrastivevi
