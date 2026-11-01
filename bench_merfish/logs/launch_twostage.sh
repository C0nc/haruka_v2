#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate haruka2
cd /home/yancui/haruka_v2/bench_merfish
# wait for the GPU-0 job in flight
while ps -eo cmd | grep -q "[r]un_haruka.py"; do sleep 60; done
D=data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad
for SWP in 1.0 5.0; do
  CUDA_VISIBLE_DEVICES=0 python -u run_haruka_twostage.py --data $D \
    --out_root results_twostage --stage1_epochs 350 --stage2_epochs 150 \
    --salient_within_penalty $SWP --tag twostage_swp$SWP
done
echo TWOSTAGE_ALL_DONE
