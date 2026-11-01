#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
python -u - <<'PY'
import numpy as np, anndata as ad, scanpy as sc, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from common import load_embeddings, embed
from score_2x2 import own_block_only

a=ad.read_h5ad("data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad")
t=(a.obs["condition"].astype(str)=="disease").to_numpy()
ct=a.obs["gt_arch"].astype(str).to_numpy()[t]
st=(a.obs["gt_sal_intrinsic"].astype(str).to_numpy()[t]=="1")
b=a[t].copy(); sc.pp.normalize_total(b,target_sum=1e4); sc.pp.log1p(b)
X=b.X.toarray() if sp.issparse(b.X) else np.asarray(b.X)
e=load_embeddings("results_easy_ep500/simv4_g0.8_r0.8_p1_d1_s0/haruka_both_sal8_timebatch_ep500_fixed_seed0")
S=own_block_only({"salient":e["salient"]},e["meta"])["salient"][t]

# the TRUE gate weight norm per cell type
D_SS,D_SD,K_CT=4,6,7; dec=np.random.default_rng(40_000)
_=dec.normal(0,1.4,(7,10)); _=dec.normal(0,1.4,(3,6))
_=np.vstack([np.zeros((1,D_SS)),dec.normal(0,1.4,(1,D_SS))])
_=np.vstack([np.zeros((1,D_SD)),dec.normal(0,1.4,(2,D_SD))])
_=dec.dirichlet(np.full(7,0.5),size=3); _=dec.normal(0,1/np.sqrt(16),(16,128)); _=dec.normal(0,.1,128)
Wg=dec.normal(0,1/np.sqrt(D_SS+D_SD),(K_CT,D_SS+D_SD,128)); Wg=Wg-Wg.mean(0,keepdims=True)
gw={str(c): float(np.linalg.norm(Wg[c,:D_SS,:])) for c in range(K_CT)}

print(f"{'cell type':10s} {'n':>7} {'||Wg_state||':>13} {'AUROC raw expr':>15} {'AUROC salient':>14}")
Zs=embed(S,seed=0)
for c in sorted(set(ct)):
    m=ct==c
    if m.sum()<300 or st[m].mean() in (0,1): continue
    idx=np.random.default_rng(0).choice(np.where(m)[0], min(4000,m.sum()), replace=False)
    au_x=cross_val_score(LogisticRegression(max_iter=1000),X[idx],st[idx],cv=3,scoring="roc_auc").mean()
    au_s=cross_val_score(LogisticRegression(max_iter=1000),Zs[idx],st[idx],cv=3,scoring="roc_auc").mean()
    print(f"{c:10s} {m.sum():7d} {gw[c]:13.2f} {au_x:15.3f} {au_s:14.3f}")
PY
