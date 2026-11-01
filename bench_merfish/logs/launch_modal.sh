#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
python -u - <<'PY'
import numpy as np, anndata as ad
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.mixture import GaussianMixture
from common import load_embeddings
from score_2x2 import own_block_only

a=ad.read_h5ad("data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad")
t=(a.obs["condition"].astype(str)=="disease").to_numpy()
st=(a.obs["gt_sal_intrinsic"].astype(str).to_numpy()[t]=="1")
e=load_embeddings("results_viz/simv4_g0.8_r0.8_p1_d1_s0/haruka500_seed0")
S=own_block_only({"salient":e["salient"]},e["meta"])["salient"][t]

w=S[st].mean(0)-S[~st].mean(0); w/=np.linalg.norm(w)
p=S@w
lab=GaussianMixture(2,random_state=0).fit_predict(p.reshape(-1,1))
print(f"1-D projection onto the state discriminant:")
print(f"   GMM k=2 on that 1-D axis -> ARI(state) = {ari(st,lab):.3f}")
print(f"   class means {p[st].mean():+.3f} vs {p[~st].mean():+.3f}, pooled sd {p.std():.3f}")
print(f"   separation (Cohen d) = {abs(p[st].mean()-p[~st].mean())/p.std():.2f}")
# bimodality of the projection
h,_=np.histogram(p,bins=60)
print(f"   histogram peaks (local maxima) = {int(((h[1:-1]>h[:-2])&(h[1:-1]>h[2:])).sum())}")
# and the TRUE latent for comparison
D_SS=4; dec=np.random.default_rng(40_000)
_=dec.normal(0,1.4,(7,10)); _=dec.normal(0,1.4,(3,6))
A=np.vstack([np.zeros((1,D_SS)),dec.normal(0,1.4,(1,D_SS))])
rng=np.random.default_rng(1); n=len(st)
s_true=A[st.astype(int)]+rng.normal(0,0.35,(n,D_SS))
wt=s_true[st].mean(0)-s_true[~st].mean(0); wt/=np.linalg.norm(wt); pt=s_true@wt
print(f"\nTRUE s for comparison: Cohen d = {abs(pt[st].mean()-pt[~st].mean())/pt.std():.2f}, "
      f"GMM ARI={ari(st,GaussianMixture(2,random_state=0).fit_predict(pt.reshape(-1,1))):.3f}")
PY
