#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
# v6 = v5 generative process + composition shift between arms
python -u build_sim_v4.py --deltas 1.0 --seeds 0 --gate_strength 0.8 \
    --sal_rates 0.35 0.5 0.65 --delta_sp 0.15 --composition_shift 0.6
echo "=== CEILINGS: v5 (no shift) ==="
python -u measure_ceilings.py --data data/simv4_g0.8_r0.3_p0.15_d1_s0.h5ad --write
echo "=== CEILINGS: v6 (shift 0.6) -- all four must stay comparable ==="
python -u measure_ceilings.py --data data/simv4_g0.8_r0.3_p0.15_c0.6_d1_s0.h5ad --write
