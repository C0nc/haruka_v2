#!/usr/bin/env bash
set -euo pipefail

DATA=/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad
HARUKA=/home/yancui/haruka_v2/bench_kras_xenium/results_new_haruka/mouselung_kras_xenium_clean_noC3/haruka_shared_full_niche_seed0
CC=/home/yancui/haruka_v2/bench_kras_xenium/results_cellcharter/haruka_z_s_cc3
ANALYSIS=/home/yancui/haruka_v2/bench_kras_xenium/analysis/preprint_style
MACROPHAGE_PAGA=/home/yancui/haruka_v2/bench_kras_xenium/analysis/fig6_macrophage_paga

conda run --no-capture-output -n haruka2 python \
  /home/yancui/haruka_v2/bench_mouse_brain_scviva/aggregate_haruka_latent_cellcharter.py \
  --data "$DATA" --input_dir "$HARUKA" --output_dir "$CC"

conda run --no-capture-output -n haruka2 python \
  /home/yancui/haruka_v2/bench_kras_xenium/analyze_kras_preprint_style.py \
  --data "$DATA" --cc-dir "$CC" --haruka-dir "$HARUKA" \
  --out-dir "$ANALYSIS" --clusters 12 --plot-per-slice 3000 --fit-per-slice 15000

conda run --no-capture-output -n haruka2 python \
  /home/yancui/haruka_v2/bench_kras_xenium/analyze_kras_initial_state.py \
  --data "$DATA" --cc-dir "$CC" --out-dir "$ANALYSIS/initial_state" \
  --per-slice 6000 --neighbors 30 --salient-marginal-strength 0.75

# Figure 6 KRAS downstream: macrophage-state connectivity.
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
conda run --no-capture-output -n haruka2 python \
  /home/yancui/haruka_v2/bench_kras_xenium/analyze_kras_macrophage_paga.py \
  --data "$DATA" --cc-dir "$CC" --out-dir "$MACROPHAGE_PAGA"
