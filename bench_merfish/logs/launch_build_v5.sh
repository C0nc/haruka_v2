#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
python -u build_sim_v4.py --deltas 1.0 --seeds 0 --gate_strength 0.8 --sal_rates 0.35 0.5 0.65
echo "=== OLD RATES (0.1/0.5/0.9) ==="
python -u validate_salsp.py --data data/simv4_g0.8_d1_s0.h5ad
echo "=== NEW RATES (0.35/0.5/0.65) ==="
python -u validate_salsp.py --data data/simv4_g0.8_r0.3_d1_s0.h5ad
