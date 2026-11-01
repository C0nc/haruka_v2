#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate haruka2
cd /home/yancui/haruka_v2/bench_merfish
D=data_addsym/simv4_g0.8_r0.8_p1_add_d1_s0.h5ad
while [ ! -f "$D" ]; do sleep 20; done
sleep 20
# EXACT same model config as the data_easy 500-epoch run, so the generative
# decoder path is the only thing that differs between the two experiments.
CUDA_VISIBLE_DEVICES=5 python -u run_haruka.py --data $D \
  --out_root results_addsym --seed 0 --max_epochs 500 \
  --variants both_sal8_timebatch_ep500_fixed
echo ADDSYM_MODEL_DONE
