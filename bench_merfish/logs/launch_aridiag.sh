#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
python -u - <<'PY'
import numpy as np, anndata as ad, json
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
from common import load_embeddings, embed, gmm_on_embedding
from score_2x2 import own_block_only

a=ad.read_h5ad("data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad")
t=(a.obs["condition"].astype(str)=="disease").to_numpy()
d="results_viz/simv4_g0.8_r0.8_p1_d1_s0/haruka500_seed0"
e=load_embeddings(d); reps=own_block_only({k:e[k] for k in ("salient",)}, e["meta"])
S=reps["salient"][t]
print("salient latent (own block), treated cells:", S.shape)

Z=embed(S,seed=0)
for k in (2,3):
    lab=gmm_on_embedding(Z,k,0)
    print(f"\n--- GMM k={k} on salient latent ---")
    for f in ["gt_sal_intrinsic","gt_sal_domain","gt_arch","gt_niche"]:
        y=a.obs[f].astype(str).to_numpy()[t]
        print(f"   vs {f:20s} ARI={ari(y,lab):.3f}  NMI={nmi(y,lab):.3f}")

# how much variance does the state direction carry?
y=(a.obs["gt_sal_intrinsic"].astype(str).to_numpy()[t]=="1")
mu=S[y].mean(0)-S[~y].mean(0)
mu/= (np.linalg.norm(mu)+1e-12)
proj=S@mu
tot=S.var(0).sum()
print(f"\nvariance along the state-discriminant direction: {proj.var():.4f}")
print(f"total variance of salient latent:                {tot:.4f}")
print(f"fraction:                                        {proj.var()/tot:.3%}")
PY
