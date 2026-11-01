#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
mkdir -p data_k4
# Fully symmetric with the background arm:
#   cell_type ~ Categorical(p_ct[bg_domain]),  7 classes, additive path
#   sal_state ~ Categorical(p_ss[sal_domain]), 4 classes, additive path
python -u build_sim_v4.py --deltas 1.0 --seeds 0 --effect_mode additive \
    --n_salient_states 4 --delta_sp 1.0 --out_dir data_k4
F=data_k4/simv4_g0.8_r0.8_p1_add_k4_d1_s0.h5ad
ls -la $F
python -u measure_ceilings.py --data $F --write
