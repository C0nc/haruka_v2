#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
rm -f data/simv4_g*_d1_s0.h5ad
exec python -u validate_v4.py --gates 0.8 1.2 1.6
