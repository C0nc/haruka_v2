#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
python -u - <<'PY'
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.mixture import GaussianMixture
D_SS, D_SD, K_SS, K_SAD = 4, 6, 2, 3
seed=0; delta=1.0; delta_sp=1.0; rates=np.array([0.1,0.5,0.9])
dec=np.random.default_rng(40_000+seed)
_=dec.normal(0,1.4,(7,10)); _=dec.normal(0,1.4,(3,6))          # A_ct, A_bgd
A_ss=np.vstack([np.zeros((1,D_SS)), dec.normal(0,1.4,(K_SS-1,D_SS))])
A_sd=np.vstack([np.zeros((1,D_SD)), dec.normal(0,1.4,(K_SAD-1,D_SD))])
print("A_ss  (salient INTRINSIC embedding, %d-dim, %d states):"%(D_SS,K_SS))
for i,v in enumerate(A_ss): print(f"   state {i}: {np.round(v,2)}   ||.||={np.linalg.norm(v):.2f}")
print("A_sd  (salient SPATIAL embedding, %d-dim, %d domains):"%(D_SD,K_SAD))
for i,v in enumerate(A_sd): print(f"   domain {i}: {np.round(v,2)}   ||.||={np.linalg.norm(v):.2f}")
noise=0.35
print(f"\nper-dim noise sd = {noise};  s = delta*A_ss[state] + eps ; s_sp = delta*delta_sp*A_sd[domain] + eps")
print(f"  s   mode separation ||A_ss[1]-A_ss[0]||   = {np.linalg.norm(A_ss[1]-A_ss[0]):.2f}"
      f"   vs noise radius {noise*np.sqrt(D_SS):.2f}  -> SNR {np.linalg.norm(A_ss[1])/(noise*np.sqrt(D_SS)):.1f}")
for i in (1,2):
    print(f"  s_sp mode sep ||A_sd[{i}]-A_sd[0]||        = {np.linalg.norm(A_sd[i]):.2f}"
          f"   vs noise radius {noise*np.sqrt(D_SD):.2f}  -> SNR {np.linalg.norm(A_sd[i])/(noise*np.sqrt(D_SD)):.1f}")
# clusterability of the TRUE latents
rng=np.random.default_rng(1); n=36100
dom=rng.choice(K_SAD,size=n); st=(rng.random(n)<rates[dom]).astype(int)
s   = delta*A_ss[st]  + rng.normal(0,noise,(n,D_SS))
s_sp= delta*delta_sp*A_sd[dom] + rng.normal(0,noise,(n,D_SD))
for nm,X,y,k in [("TRUE s   -> state",s,st,2),("TRUE s_sp-> domain",s_sp,dom,3),
                 ("TRUE s   -> domain",s,dom,3)]:
    lab=GaussianMixture(k,covariance_type="full",random_state=0).fit_predict(X)
    print(f"  GMM k={k} on {nm:22s} ARI={ari(y,lab):.3f}")
print(f"\nP(state=1 | domain) = {rates}  <- the coupling")
PY
