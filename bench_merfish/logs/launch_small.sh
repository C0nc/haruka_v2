#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
mkdir -p data_small
python -u build_sim_v4.py --deltas 1.0 --seeds 0 --gate_strength 0.8 \
    --sal_rates 0.35 0.5 0.65 --delta_sp 0.15 --n_side 48 --out_dir data_small
echo "=== CEILINGS on SMALL (must match full: salDOM~0.64, ctype~1.0) ==="
python -u - <<'PY'
import os
for v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v,"8")
import anndata as ad, numpy as np, scanpy as sc, scipy.sparse as sp
from common import dependence
from run_baselines_2x2 import _cc_aggregate
for f in ["data_small/simv4_g0.8_r0.3_p0.15_d1_s0.h5ad",
          "data/simv4_g0.8_r0.3_p0.15_d1_s0.h5ad"]:
    a=ad.read_h5ad(f); sc.pp.normalize_total(a,target_sum=1e4); sc.pp.log1p(a)
    X=a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    pooled=_cc_aggregate(a.copy(),X,"expr")
    t=(a.obs["condition"].astype(str)=="disease").to_numpy()
    g=a.obs["slice_id"].astype(str).to_numpy()
    dom=dependence(pooled[t],a.obs["gt_sal_domain"].astype(str).to_numpy()[t],g[t],True,0)
    st =dependence(X[t],a.obs["gt_sal_intrinsic"].astype(str).to_numpy()[t],g[t],True,0)
    ct =dependence(X,a.obs["gt_arch"].astype(str).to_numpy(),g,True,0)
    print(f"{'SMALL' if 'small' in f else 'FULL ':6s} n={a.n_obs:6d}  salDOM_ceil={dom:.3f}  state_ceil={st:.3f}  ctype_ceil={ct:.3f}")
PY
