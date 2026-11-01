#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
mkdir -p data_addsym
# identical to data_easy EXCEPT the decoder path: salient enters additively,
# through the same shared linear map as the background, with no cell-type-specific
# gate. Everything else matched so the path is the only variable.
python -u build_sim_v4.py --deltas 1.0 --seeds 0 --effect_mode additive \
    --sal_rates 0.1 0.5 0.9 --delta_sp 1.0 --out_dir data_addsym
echo "=== CEILINGS (additive-symmetric) ==="
python -u measure_ceilings.py --data data_addsym/simv4_g0.8_r0.8_p1_add_d1_s0.h5ad --write
echo "=== effect geometry ==="
python -u diag_salient_geometry.py --data data_addsym/simv4_g0.8_r0.8_p1_add_d1_s0.h5ad
