#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
python -u - <<'PY'
import numpy as np, anndata as ad
from sklearn.metrics import adjusted_rand_score as ari
from common import load_embeddings, embed, gmm_on_embedding
from score_2x2 import own_block_only
a=ad.read_h5ad("data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad")
t=(a.obs["condition"].astype(str)=="disease").to_numpy()
ct=a.obs["gt_arch"].astype(str).to_numpy()[t]
st=a.obs["gt_sal_intrinsic"].astype(str).to_numpy()[t]
T="simv4_g0.8_r0.8_p1_d1_s0"
for nm,d in [("N(0,I) 500ep",f"results_easy_ep500/{T}/haruka_both_sal8_timebatch_ep500_fixed_seed0"),
             ("mixture K=2",f"results_sprior/{T}/haruka_both_sal8_timebatch_sprior2_seed0")]:
    e=load_embeddings(d); S=own_block_only({"salient":e["salient"]},e["meta"])["salient"][t]
    glob=ari(st,gmm_on_embedding(embed(S,seed=0),2,0))
    lab=np.full(len(st),-1); nxt=0
    for c in sorted(set(ct)):
        m=ct==c
        if m.sum()<200: continue
        lab[m]=gmm_on_embedding(embed(S[m],seed=0),2,0)+nxt; nxt+=2
    ok=lab>=0
    # within-type ARI: score state against the per-type clustering, per type then average
    per=[ari(st[(ct==c)],lab[(ct==c)]) for c in sorted(set(ct)) if (ct==c).sum()>=200]
    print(f"{nm:16s} global ARI(k=2) = {glob:.3f}   |   within-cell-type ARI = "
          f"{np.mean(per):.3f}  (per type: {', '.join(f'{p:.2f}' for p in per)})")
PY
