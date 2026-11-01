#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
exec python -u plot_results_2x2.py \
  --data data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad \
  --results results_viz --rep_mode own --cluster_method gmm \
  --methods haruka500 cinemaot contrastivevi cellcharter \
  --out results_viz/recovered_easy.png
