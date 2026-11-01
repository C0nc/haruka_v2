#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate haruka2
cd /home/yancui/haruka_v2/bench_merfish
F=data_k4/simv4_g0.8_r0.8_p1_add_k4_d1_s0.h5ad
while ! grep -q "wrote quadrant_ceilings" logs/k4.log 2>/dev/null; do sleep 20; done
CUDA_VISIBLE_DEVICES=6 python -u run_haruka.py --data $F \
  --out_root results_k4 --seed 0 --max_epochs 500 \
  --variants both_sal8_timebatch_ep500_fixed
echo K4_MODEL_DONE
