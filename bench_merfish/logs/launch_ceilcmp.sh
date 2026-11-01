#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
python -u - <<'PY'
import numpy as np, anndata as ad, scanpy as sc, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from run_baselines_2x2 import _cc_aggregate
from common import embed

F="data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad"
a=ad.read_h5ad(F); sc.pp.normalize_total(a,target_sum=1e4); sc.pp.log1p(a)
X=a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
pooled=_cc_aggregate(a.copy(),X,"e")
t=(a.obs["condition"].astype(str)=="disease").to_numpy()
g=a.obs["slice_id"].astype(str).to_numpy()
rng=np.random.default_rng(0)

def dep(src,y,grp,model):
    cls=np.unique(y); out=[]
    for c in cls:
        yy=(y==c).astype(int)
        if yy.sum()<50 or (1-yy).sum()<50: continue
        pr=np.zeros(len(yy))
        for tr,te in GroupKFold(n_splits=min(4,len(np.unique(grp)))).split(src,yy,grp):
            s=StandardScaler().fit(src[tr]); m=model()
            m.fit(s.transform(src[tr]),yy[tr]); pr[te]=m.predict_proba(s.transform(src[te]))[:,1]
        out.append(roc_auc_score(yy,pr))
    au=float(np.mean(out)); return max(0.0,2*(au-0.5)), au

SPEC=[("gt_arch",False,None),("gt_niche",True,None),
      ("gt_sal_intrinsic",False,"t"),("gt_sal_domain",True,"t")]
print(f"{'factor':20s} {'LINEAR':>9} {'NONLINEAR':>10} {'gap':>8}")
for f,pool,mask in SPEC:
    src = pooled if pool else X
    m = t if mask else np.ones(len(X),bool)
    y=a.obs[f].astype(str).to_numpy()[m]
    idx=rng.choice(int(m.sum()), min(15000,int(m.sum())), replace=False)
    S=embed(src[m][idx],n_pcs=50,seed=0); yy=y[idx]; gg=g[m][idx]
    lin,_=dep(S,yy,gg, lambda: LogisticRegression(max_iter=2000,C=1.0))
    nl,_ =dep(S,yy,gg, lambda: HistGradientBoostingClassifier(max_iter=200))
    print(f"{f:20s} {lin:9.3f} {nl:10.3f} {nl-lin:8.3f}")
PY
