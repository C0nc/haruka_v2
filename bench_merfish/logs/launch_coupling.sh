#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
python -u - <<'PY'
import numpy as np, anndata as ad, os
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.mixture import GaussianMixture
from common import load_embeddings, embed, gmm_on_embedding, dependence
from score_2x2 import own_block_only

CASES=[("data_easy  q=(0.1,0.5,0.9)","data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad",
        "results_viz/simv4_g0.8_r0.8_p1_d1_s0/haruka500_seed0"),
       ("v5 stress  q=(0.35,0.5,0.65)","data/simv4_g0.8_r0.3_p0.15_d1_s0.h5ad",
        "results_v5/simv4_g0.8_r0.3_p0.15_d1_s0/haruka_both_sal8_seed0")]
for nm,dp,ed in CASES:
    if not os.path.exists(ed): print(f"[skip] {ed}"); continue
    a=ad.read_h5ad(dp); t=(a.obs["condition"].astype(str)=="disease").to_numpy()
    st=a.obs["gt_sal_intrinsic"].astype(str).to_numpy()[t]
    dm=a.obs["gt_sal_domain"].astype(str).to_numpy()[t]
    gp=a.obs["slice_id"].astype(str).to_numpy()[t]
    e=load_embeddings(ed); S=own_block_only({"salient":e["salient"]},e["meta"])["salient"][t]
    Z=embed(S,seed=0)
    # how correlated are state and domain in the GROUND TRUTH?
    q=np.array(a.uns["sim2x2"]["sal_rates"])
    from sklearn.linear_model import LogisticRegression
    import pandas as pd
    oh=pd.get_dummies(dm).to_numpy(float)
    auc_dom_pred_state = dependence(oh, st, gp, True, 0)
    print(f"\n=== {nm} ===  rates={q}")
    print(f"  GT coupling: dependence(domain -> state) = {auc_dom_pred_state:.3f}")
    print(f"  model salient: ARI(state,k=2)={ari(st,gmm_on_embedding(Z,2,0)):.3f}  "
          f"ARI(domain,k=3)={ari(dm,gmm_on_embedding(Z,3,0)):.3f}  "
          f"dep(state)={dependence(S,st,gp,True,0):.3f}")
PY
