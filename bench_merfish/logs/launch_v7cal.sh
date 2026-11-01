#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
for DSP in 0.08 0.05 0.03; do
  python -u build_sim_v4.py --deltas 1.0 --seeds 0 --effect_mode additive \
      --sal_rates 0.35 0.5 0.65 --delta_sp $DSP > /dev/null 2>&1
  F="data/simv4_g0.8_r0.3_p${DSP}_add_d1_s0.h5ad"
  echo "--- additive delta_sp=$DSP  ($F) ---"
  python -u measure_ceilings.py --data $F 2>/dev/null | grep -E "gt_"
done
echo "TARGET: gt_sal_domain ceiling ~0.64-0.70 to match v5 difficulty"
