#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
# v7 = WELL-SPECIFIED rung: our assumptions hold (additive effect, no composition
# shift) but the salient-spatial quadrant is still non-trivial (delta_sp 0.15).
python -u build_sim_v4.py --deltas 1.0 --seeds 0 --effect_mode additive \
    --sal_rates 0.35 0.5 0.65 --delta_sp 0.15
echo "=== v7 CEILINGS ==="
python -u measure_ceilings.py --data data/simv4_g0.8_r0.3_p0.15_add_d1_s0.h5ad --write
echo "=== v7 effect geometry (expect HIGH cosine = shared direction = our assumption) ==="
python -u diag_salient_geometry.py --data data/simv4_g0.8_r0.3_p0.15_add_d1_s0.h5ad
