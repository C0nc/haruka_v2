#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
python -u - <<'PY'
import numpy as np, anndata as ad, os
from sklearn.metrics import adjusted_rand_score as ari
from common import load_embeddings, embed, gmm_on_embedding, dependence
from score_2x2 import own_block_only
a=ad.read_h5ad("data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad")
t=(a.obs["condition"].astype(str)=="disease").to_numpy()
st=a.obs["gt_sal_intrinsic"].astype(str).to_numpy()[t]
dm=a.obs["gt_sal_domain"].astype(str).to_numpy()[t]
gp=a.obs["slice_id"].astype(str).to_numpy()[t]
nic=a.obs["gt_niche"].astype(str).to_numpy(); sl=a.obs["slice_id"].astype(str).to_numpy()
T="simv4_g0.8_r0.8_p1_d1_s0"
C=[("150ep baseline",f"results_easy/{T}/haruka_both_sal8_timebatch_seed0"),
   ("500ep fixed",f"results_easy_ep500/{T}/haruka_both_sal8_timebatch_ep500_fixed_seed0"),
   ("500ep swp+inv2",f"results_swpinv/{T}/haruka_both_sal8_timebatch_swp_inv2_seed0"),
   ("500ep swp+inv5",f"results_swpinv/{T}/haruka_both_sal8_timebatch_swp_inv5_seed0")]
print(f"{'config':18s} {'sal>state':>10} {'sal>salDOM':>11} {'sp>salDOM':>10} {'bg>niche':>9} {'ARI(state)':>11}")
for nm,d in C:
    if not os.path.exists(d): print(f"[skip] {nm}"); continue
    e=load_embeddings(d)
    R=own_block_only({k:e[k] for k in ("salient","spatial","background")}, e["meta"])
    S=R["salient"][t]
    print(f"{nm:18s} {dependence(S,st,gp,True,0):10.3f} {dependence(S,dm,gp,True,0):11.3f} "
          f"{dependence(R['spatial'][t],dm,gp,True,0):10.3f} "
          f"{dependence(R['background'],nic,sl,True,0):9.3f} "
          f"{ari(st,gmm_on_embedding(embed(S,seed=0),2,0)):11.3f}")
PY
