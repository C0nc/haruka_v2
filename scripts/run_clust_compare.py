"""KMeans vs PCA+GMM vs spatial Leiden on the SAME (unchanged) simulation, RNA only."""
import sys; sys.path.insert(0,"/mnt/HDD2B/Users/yancui/Haruka")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import run_salient_eval as ev
from run_haruka_spatial_example import make_synthetic
from haruka_spatial import HarukaSpatial
from haruka_leiden import spatial_leiden
from haruka_cluster import FastCluster
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
S = ev.SHARED

def prep(E, n_pcs=50, seed=0):
    X = StandardScaler().fit_transform(np.nan_to_num(E))
    if n_pcs and n_pcs < X.shape[1]:
        X = PCA(n_components=n_pcs, random_state=seed).fit_transform(X)
    return X

res = {}
for seed in range(6):
    out = make_synthetic(seed=seed); rna = out[0] if isinstance(out,tuple) else out
    rna.layers["counts"] = np.asarray(rna.X).copy()
    h = HarukaSpatial(rna, cov_gene_num=S["cov_num"], seed=seed, k_spatial=S["k_spatial"])
    h.construct_cov(); h.prepare_spatial_graph()
    h.setup_model(latent_dim=S["latent_dim"], gene_likelihood="poisson",
                  condition_key="condition", condition_label="disease", batch_key="slice_id",
                  n_spatial_latent=16, spatial_attach="background",
                  reg_to_use=S["reg_to_use"], lam_mi=S["lam_mi"],
                  kl_spatial_weight=S["kl_spatial_weight"])
    h.train(max_epochs=S["max_epochs"], batch_size=S["batch_size"], plan_kwargs=dict(lr=S["lr"]))
    h.extract_rep(n_layer=2); o = h.output_adata
    coords = (np.column_stack([o.obs["array_col"], o.obs["array_row"]]).astype(float)
              if "array_col" in o.obs else np.asarray(o.obsm["spatial"], float)[:, -2:])
    groups = o.obs["slice_id"].astype(str).values
    for rep, truth, k in (("X_cellcharter_background","layer",3), ("X_cellcharter_salient","lesion",2)):
        E = o.obsm[rep]; T = o.obs[truth].astype(str).values
        Xr = StandardScaler().fit_transform(np.nan_to_num(E)); Xp = prep(E, 50, seed)
        cand = {
            "kmeans_raw":  KMeans(k, n_init=10, random_state=seed).fit_predict(Xr),
            "kmeans_pca":  KMeans(k, n_init=10, random_state=seed).fit_predict(Xp),
            "gmm_pca_full": GaussianMixture(k, covariance_type="full", n_init=5,
                                            random_state=seed).fit_predict(Xp),
            "gmm_pca_diag": GaussianMixture(k, covariance_type="diag", n_init=5,
                                            random_state=seed).fit_predict(Xp),
        }
        lo, hi = 0.05, 3.0
        for _ in range(12):
            mid=(lo+hi)/2
            l = spatial_leiden(E, coords, groups, spatial_weight=0.5, resolution=mid, seed=seed)
            if l.max()+1 == k: break
            if l.max()+1 < k: lo = mid
            else: hi = mid
        cand["leiden_w0.5"] = l
        for name, lab in cand.items():
            res.setdefault((rep, name), []).append(ari(T, lab))

import itertools
for rep in ("X_cellcharter_background","X_cellcharter_salient"):
    meths = [m for (r,m) in res if r==rep]
    print(f"\n=== {rep} : PAIRED per-seed differences vs kmeans_raw ===")
    base = np.array(res[(rep,"kmeans_raw")])
    print(f"{'method':16s}{'mean ARI':>10s}{'mean diff':>12s}{'wins':>7s}{'per-seed diffs'}")
    for m in meths:
        v = np.array(res[(rep,m)]); d = v - base
        wins = int((d > 0.01).sum())
        print(f"{m:16s}{v.mean():10.3f}{d.mean():+12.3f}{wins:4d}/{len(d)}   "
              + " ".join(f"{x:+.2f}" for x in d))
