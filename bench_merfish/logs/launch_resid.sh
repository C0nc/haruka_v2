#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cellcharter
cd /home/yancui/haruka_v2/bench_merfish
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
python -u - <<'PY'
import numpy as np, anndata as ad
from sklearn.metrics import adjusted_rand_score as ari
from common import load_embeddings, embed, gmm_on_embedding, dependence
from score_2x2 import own_block_only
from orthogonalize_haruka import residualize

a=ad.read_h5ad("data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad")
t=(a.obs["condition"].astype(str)=="disease").to_numpy()
st=a.obs["gt_sal_intrinsic"].astype(str).to_numpy()[t]
dm=a.obs["gt_sal_domain"].astype(str).to_numpy()[t]
gp=a.obs["slice_id"].astype(str).to_numpy()[t]

d="results_viz/simv4_g0.8_r0.8_p1_d1_s0/haruka500_seed0"
e=load_embeddings(d)
reps=own_block_only({k:e[k] for k in ("salient","salient_spatial")}, e["meta"])
S, SSP = reps["salient"][t], reps["salient_spatial"][t]

def report(nm, X):
    lab=gmm_on_embedding(embed(X,seed=0),2,0)
    print(f"{nm:34s} ARI(state)={ari(st,lab):.3f}  ARI(domain,k=3)="
          f"{ari(dm, gmm_on_embedding(embed(X,seed=0),3,0)):.3f}  "
          f"dep(state)={dependence(X,st,gp,True,0):.3f}")

report("salient  (as-is)", S)
for al in (1e-3, 1e-2, 1e-1, 1.0):
    report(f"salient - ridge(salient_spatial) a={al:g}", residualize(S, SSP, alpha=al))
PY
