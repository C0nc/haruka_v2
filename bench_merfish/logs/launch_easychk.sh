#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
echo "=== CEILINGS on data_easy (doc claims salDOM 0.955) ==="
python -u measure_ceilings.py --data data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad
echo "=== effect geometry: is it actually non-linear? ==="
python -u diag_salient_geometry.py --data data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad
