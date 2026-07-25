"""Feasibility of the C_both module on a REAL RNA dataset (SPAC Visium HD lung-met M001).

Not a biology claim -- SPAC has known confounds (guide=clone=location, depth-driven guide
detection). This asks the mechanical questions only:
  1. does spatial_attach='both' TRAIN on real 2k-HVG data at real scale (no NaN, finite loss)?
  2. does the decomposition BEHAVE -- background condition-invariant, salient captures the arm,
     spatial non-degenerate -- or does it collapse the way it did on the LPC sim?
Control arm = non-targeting bins, target arm = targeting bins, tumour bins only.
"""
import sys, time; sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Haruka")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc
from haruka_spatial import HarukaSpatial
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from haruka_cluster import FastCluster

DATA = "/mnt/HDD2B/Users/yancui/Haruka/data/spac_m001"
N_HVG = 2000
MAX_TARGET = 8000          # subsample the huge targeting arm for a tractable feasibility run

def load():
    a = sc.read_h5ad(f"{DATA}/filtered_gene_bc_matrix.h5")
    g = pd.read_csv(f"{DATA}/guide_assignment.csv", index_col=0).reindex(a.obs_names)
    tum = (g.cell_type == "Non-immune cell").values
    nt  = tum & g.clean.fillna(False).values & g.nt.fillna(False).values
    tg  = tum & g.clean.fillna(False).values & ~g.nt.fillna(True).values
    ntb = np.where(nt)[0]
    tgb = np.where(tg)[0]
    rng = np.random.default_rng(0)
    if len(tgb) > MAX_TARGET:
        tgb = rng.choice(tgb, MAX_TARGET, replace=False)
    keep = np.concatenate([ntb, tgb])
    ad = a[keep].copy()
    ad.obs["condition"] = np.where(np.isin(keep, ntb), "control", "disease")
    ad.obs["slice_id"] = "M001"
    # spatial: obsm is [in_tissue,row,col]; use row/col
    ad.obsm["spatial"] = np.column_stack([ad.obs["array_col"], ad.obs["array_row"]]).astype("float32")
    ad.layers["counts"] = np.asarray(ad.X.todense() if hasattr(ad.X,"todense") else ad.X, np.float32)
    sc.pp.highly_variable_genes(ad, n_top_genes=N_HVG, flavor="seurat_v3", layer="counts")
    ad = ad[:, ad.var.highly_variable].copy()
    ad.X = ad.layers["counts"] = np.asarray(ad.layers["counts"][:, ad.var.highly_variable.values]
                                            if ad.layers["counts"].shape[1]!=ad.n_vars
                                            else ad.layers["counts"], np.float32)
    return ad

def run(attach, ad, ns=16):
    h = HarukaSpatial(ad.copy(), cov_gene_num=16, seed=0, k_spatial=8)
    h.construct_cov(); h.prepare_spatial_graph()
    kw = dict(latent_dim=32, gene_likelihood="poisson", condition_key="condition",
              condition_label="disease", batch_key="slice_id", n_spatial_latent=ns,
              spatial_attach=attach, reg_to_use="mi", lam_mi=5.0, kl_spatial_weight=0.01,
              me_reduction="sum")
    if attach == "both": kw["spatial_invariance_penalty"] = 0.5
    h.setup_model(**kw)
    t=time.time(); h.train(max_epochs=200, batch_size=512, plan_kwargs=dict(lr=1e-3))
    h.extract_rep(n_layer=2); o = h.output_adata
    reps = {k:o.obsm[k] for k in ("background_rep","salient_rep","spatial_rep") if k in o.obsm}
    fin = {k: bool(np.isfinite(v).all()) for k,v in reps.items()}
    return h, o, reps, fin, time.time()-t

if __name__ == "__main__":
    ad = load()
    print(f"loaded real SPAC: {ad.n_obs} bins x {ad.n_vars} HVG | "
          f"control {int((ad.obs.condition=='control').sum())} disease {int((ad.obs.condition=='disease').sum())}",
          flush=True)
    for attach in ("background", "both"):
        h,o,reps,fin,secs = run(attach, ad)
        print(f"\n===== spatial_attach='{attach}' =====  trained {secs:.0f}s  finite={fin}", flush=True)
        for k,v in reps.items():
            print(f"  {k:16s} std {v.std():.3f}  rank {np.linalg.matrix_rank(v-v.mean(0),tol=1e-3)}/{v.shape[1]}")
        # feasibility behaviour: does background CLUSTER track the arm? (want NO) does salient? (want strong signal exists)
        # NB: output_adata is target-only, so arm ARI is on... target cells only -- can't test arm here.
        # Instead: probe arm from each FULL-data rep via get_latent on both arms.
        z = h.model.get_latent_representation(h.input_adata, representation_kind="background")
        s = h.model.get_latent_representation(h.input_adata, representation_kind="salient")
        arm = (h.input_adata.obs["condition"].values == "disease").astype(int)
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        for name,E in (("background",z),("salient",s)):
            Xs = StandardScaler().fit_transform(np.nan_to_num(E))
            acc = cross_val_score(LogisticRegression(max_iter=500,class_weight="balanced"),
                                  Xs, arm, cv=3, scoring="balanced_accuracy").mean()
            print(f"  arm-predictability from {name:11s}: {acc:.3f}   (background want ~0.5, salient want high)")
