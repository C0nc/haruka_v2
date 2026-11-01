#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
python -u - <<'PY'
import numpy as np, anndata as ad
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
from common import load_embeddings, embed, gmm_on_embedding
from score_2x2 import own_block_only
a=ad.read_h5ad("data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad")
t=(a.obs["condition"].astype(str)=="disease").to_numpy()
st=a.obs["gt_sal_intrinsic"].astype(str).to_numpy()[t]
dm=a.obs["gt_sal_domain"].astype(str).to_numpy()[t]
joint=np.char.add(np.char.add(st,"|"),dm)
e=load_embeddings("results_easy_ep500/simv4_g0.8_r0.8_p1_d1_s0/haruka_both_sal8_timebatch_ep500_fixed_seed0")
S=own_block_only({"salient":e["salient"]},e["meta"])["salient"][t]
Z=embed(S,seed=0)
print(f"{'k':>3} {'NMI(state)':>11} {'NMI(domain)':>12} {'NMI(joint)':>11} {'state purity':>13}")
for k in (2,3,4,6,8,12):
    lab=gmm_on_embedding(Z,k,0)
    # majority-state purity: if each cluster is labelled by its majority state,
    # what fraction of cells get the right state? (this is what you can DO in
    # real data: over-cluster, then annotate clusters)
    pur=np.mean([ (st[lab==c]==np.bincount(
        (st[lab==c]=="1").astype(int)).argmax().astype(str)).mean()*(lab==c).sum()
        for c in np.unique(lab)])
    correct=0
    for c in np.unique(lab):
        m=lab==c; maj = "1" if (st[m]=="1").mean()>=0.5 else "0"
        correct += (st[m]==maj).sum()
    print(f"{k:3d} {nmi(st,lab):11.3f} {nmi(dm,lab):12.3f} {nmi(joint,lab):11.3f} {correct/len(st):13.3f}")
print(f"\nbaseline: always-majority-state purity = {max((st=='1').mean(),(st=='0').mean()):.3f}")
PY
