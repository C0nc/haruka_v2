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

a=ad.read_h5ad("data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad")
t=(a.obs["condition"].astype(str)=="disease").to_numpy()
st_s=a.obs["gt_sal_intrinsic"].astype(str).to_numpy()[t]
st=(st_s=="1"); dm=a.obs["gt_sal_domain"].astype(str).to_numpy()[t]
gp=a.obs["slice_id"].astype(str).to_numpy()[t]
T="simv4_g0.8_r0.8_p1_d1_s0"
CASES=[("N(0,I)  prior (500ep)",f"results_easy_ep500/{T}/haruka_both_sal8_timebatch_ep500_fixed_seed0"),
       ("mixture K=2",         f"results_sprior/{T}/haruka_both_sal8_timebatch_sprior2_seed0"),
       ("mixture K=4",         f"results_sprior/{T}/haruka_both_sal8_timebatch_sprior4_seed0")]
print(f"{'config':24s} {'ARI(state)':>11} {'CohenD':>8} {'dep(state)':>11} {'ARI(dom,k3)':>12}")
for nm,d in CASES:
    if not os.path.exists(d): print(f"[skip] {d}"); continue
    e=load_embeddings(d); S=own_block_only({"salient":e["salient"]},e["meta"])["salient"][t]
    Z=embed(S,seed=0)
    w=S[st].mean(0)-S[~st].mean(0); w/=np.linalg.norm(w); p=S@w
    d_=abs(p[st].mean()-p[~st].mean())/p.std()
    print(f"{nm:24s} {ari(st_s,gmm_on_embedding(Z,2,0)):11.3f} {d_:8.2f} "
          f"{dependence(S,st_s,gp,True,0):11.3f} {ari(dm,gmm_on_embedding(Z,3,0)):12.3f}")
print("\nground-truth ceiling: ARI 0.999, Cohen d 1.92")
PY
