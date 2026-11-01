#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
for DSP in 0.5 0.3 0.15; do
  python -u build_sim_v4.py --deltas 1.0 --seeds 0 --gate_strength 0.8 \
      --sal_rates 0.35 0.5 0.65 --delta_sp $DSP > /dev/null 2>&1
done
python -u - <<'PY'
import os,glob
for v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v,"8")
import anndata as ad, numpy as np, scanpy as sc, scipy.sparse as sp
from common import dependence
from run_baselines_2x2 import _cc_aggregate
print(f"{'delta_sp':>9} {'salDOM_ceil':>12} {'state_ceil':>11} {'ctype_ceil':>11}")
for f in ["data/simv4_g0.8_r0.3_p0.5_d1_s0.h5ad","data/simv4_g0.8_r0.3_p0.3_d1_s0.h5ad",
          "data/simv4_g0.8_r0.3_p0.15_d1_s0.h5ad"]:
    a=ad.read_h5ad(f); sc.pp.normalize_total(a,target_sum=1e4); sc.pp.log1p(a)
    X=a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    pooled=_cc_aggregate(a.copy(),X,"expr")
    t=(a.obs["condition"].astype(str)=="disease").to_numpy()
    g=a.obs["slice_id"].astype(str).to_numpy()
    dom=dependence(pooled[t],a.obs["gt_sal_domain"].astype(str).to_numpy()[t],g[t],True,0)
    st =dependence(X[t],a.obs["gt_sal_intrinsic"].astype(str).to_numpy()[t],g[t],True,0)
    ct =dependence(X,a.obs["gt_arch"].astype(str).to_numpy(),g,True,0)
    print(f"{a.uns['sim2x2']['delta_sp']:9.2f} {dom:12.3f} {st:11.3f} {ct:11.3f}")
PY
