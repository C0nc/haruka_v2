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
CASES=[("GATED  (data_easy)","data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad",
        "results_easy_ep500/simv4_g0.8_r0.8_p1_d1_s0/haruka_both_sal8_timebatch_ep500_fixed_seed0"),
       ("ADDITIVE (symmetric)","data_addsym/simv4_g0.8_r0.8_p1_add_d1_s0.h5ad",
        "results_addsym/simv4_g0.8_r0.8_p1_add_d1_s0/haruka_both_sal8_timebatch_ep500_fixed_seed0")]
print(f"{'dataset':22s} {'sal>state':>10} {'sal>salDOM':>11} {'ARI(state)':>11} {'ARI(dom)':>9} {'bg>niche':>9}")
for nm,dp,ed in CASES:
    if not os.path.exists(ed): print(f"[skip] {ed}"); continue
    a=ad.read_h5ad(dp); t=(a.obs["condition"].astype(str)=="disease").to_numpy()
    st=a.obs["gt_sal_intrinsic"].astype(str).to_numpy()[t]
    dm=a.obs["gt_sal_domain"].astype(str).to_numpy()[t]
    gp=a.obs["slice_id"].astype(str).to_numpy()[t]
    e=load_embeddings(ed)
    R=own_block_only({k:e[k] for k in ("salient","background","spatial","salient_spatial")}, e["meta"])
    S=R["salient"][t]; Z=embed(S,seed=0)
    nic=a.obs["gt_niche"].astype(str).to_numpy()
    print(f"{nm:22s} {dependence(S,st,gp,True,0):10.3f} {dependence(S,dm,gp,True,0):11.3f} "
          f"{ari(st,gmm_on_embedding(Z,2,0)):11.3f} {ari(dm,gmm_on_embedding(Z,3,0)):9.3f} "
          f"{dependence(R['background'],nic,a.obs['slice_id'].astype(str).to_numpy(),True,0):9.3f}")
print("\nreference: background pair decouples at bg>niche = 0.017 on data_easy")
PY
