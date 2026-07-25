"""Can the good mode be SELECTED without ground truth?

If final training loss (or ELBO) separates clean runs from lesion-contaminated ones, then
restart-and-select turns an 8/12 method into a reliable one at 3x compute. Ground truth is
used ONLY to label runs after the fact -- never to select.
"""
import sys; sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Haruka")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import run_salient_eval as ev
from run_haruka_spatial_example import make_synthetic
from haruka_spatial import HarukaSpatial
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
S = ev.SHARED

rows = []
for seed in range(16):
    out = make_synthetic(seed=seed); rna = out[0] if isinstance(out, tuple) else out
    rna.layers["counts"] = np.asarray(rna.X).copy()
    h = HarukaSpatial(rna, cov_gene_num=S["cov_num"], seed=seed, k_spatial=S["k_spatial"])
    h.construct_cov(); h.prepare_spatial_graph()
    h.setup_model(latent_dim=S["latent_dim"], gene_likelihood="poisson",
                  condition_key="condition", condition_label="disease", batch_key="slice_id",
                  n_spatial_latent=16, spatial_attach="background",
                  reg_to_use=S["reg_to_use"], lam_mi=S["lam_mi"],
                  kl_spatial_weight=S["kl_spatial_weight"])
    h.train(max_epochs=S["max_epochs"], batch_size=S["batch_size"], plan_kwargs=dict(lr=S["lr"]))
    hist = h.model.history
    key = next((k for k in hist if "train" in k and "loss" in k), None)
    loss = float(np.asarray(hist[key]).ravel()[-1]) if key else float("nan")
    h.extract_rep(n_layer=2); o = h.output_adata
    E = StandardScaler().fit_transform(o.obsm["X_cellcharter_background"])
    lab = KMeans(3, n_init=10, random_state=seed).fit_predict(E)
    l = ari(o.obs["layer"].astype(str), lab); le = ari(o.obs["lesion"].astype(str), lab)
    clean = bool(l > 0.8 and le < 0.15)
    rows.append((seed, loss, l, le, clean))
    print(f"seed {seed:2d}  loss {loss:12.2f}  bg_layer {l:.3f}  bg_lesion {le:.3f}  {'CLEAN' if clean else 'bad'}", flush=True)

L = np.array([r[1] for r in rows]); C = np.array([r[4] for r in rows])
print(f"\nclean {C.sum()}/{len(C)}")
if C.any() and (~C).any():
    print(f"mean loss  clean {L[C].mean():.2f}   bad {L[~C].mean():.2f}   gap {L[~C].mean()-L[C].mean():+.2f}")
    print(f"loss range clean [{L[C].min():.2f},{L[C].max():.2f}]  bad [{L[~C].min():.2f},{L[~C].max():.2f}]")
    sep = L[C].max() < L[~C].min()
    print(f"PERFECTLY SEPARABLE BY LOSS: {sep}")
    order = np.argsort(L)
    print(f"picking the single best-loss run would give: {'CLEAN' if C[order[0]] else 'BAD'}")
    print(f"best-of-3 by loss (3 disjoint triples): "
          f"{[bool(C[order[i]]) for i in range(0, min(9,len(order)), 3)]}")
