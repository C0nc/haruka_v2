#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate vio
cd /home/yancui/haruka_v2/bench_merfish
python -u run_cinemaot.py --data data/simv4_g0.8_d1_s0.h5ad --out_root results_v4
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
exec python -u cc_add_spatial.py --data data/simv4_g0.8_d1_s0.h5ad \
     --result_dir results_v4/simv4_g0.8_d1_s0/cinemaot_seed0
