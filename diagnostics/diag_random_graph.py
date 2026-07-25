"""Is `spatial_attach='background'` about SPATIAL structure, or just cross-cell coupling?

Same model, same everything, but the neighbour graph is randomised WITHIN slice: each cell
gets k random partners instead of its k nearest. If the stability gain survives, the effect
is generic cross-cell coupling and the spatial part is incidental. If it vanishes, the
coupling has to be spatial to work.
"""
import sys; sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Haruka")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import run_salient_eval as ev
from run_haruka_spatial_example import make_synthetic
from haruka_spatial import HarukaSpatial, NBR_IDX_KEY, NBR_MASK_KEY
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
S = ev.SHARED

def run(seed, randomize):
    out = make_synthetic(seed=seed); rna = out[0] if isinstance(out, tuple) else out
    rna.layers["counts"] = np.asarray(rna.X).copy()
    h = HarukaSpatial(rna, cov_gene_num=S["cov_num"], seed=seed, k_spatial=S["k_spatial"])
    h.construct_cov(); h.prepare_spatial_graph()
    if randomize:
        ad = h.input_adata
        idx = np.asarray(ad.obsm[NBR_IDX_KEY]).copy()
        sl = ad.obs["slice_id"].astype(str).values
        rng = np.random.default_rng(seed)
        for s_ in np.unique(sl):                      # keep it within-slice, as the real graph is
            rows = np.where(sl == s_)[0]
            for r in rows:
                idx[r] = rng.choice(rows[rows != r], size=idx.shape[1], replace=False)
        ad.obsm[NBR_IDX_KEY] = idx
        ad.obsm[NBR_MASK_KEY] = np.ones_like(idx, dtype="float32")
    h.setup_model(latent_dim=S["latent_dim"], gene_likelihood="poisson",
                  condition_key="condition", condition_label="disease", batch_key="slice_id",
                  n_spatial_latent=16, spatial_attach="background",
                  reg_to_use=S["reg_to_use"], lam_mi=S["lam_mi"],
                  kl_spatial_weight=S["kl_spatial_weight"])
    h.train(max_epochs=S["max_epochs"], batch_size=S["batch_size"], plan_kwargs=dict(lr=S["lr"]))
    h.extract_rep(n_layer=2); o = h.output_adata
    E = StandardScaler().fit_transform(o.obsm["X_cellcharter_background"])
    lab = KMeans(3, n_init=10, random_state=seed).fit_predict(E)
    return (ari(o.obs["layer"].astype(str), lab), ari(o.obs["lesion"].astype(str), lab))

for randomize in (False, True):
    ok = 0; rows = []
    for seed in range(12):
        l, le = run(seed, randomize); rows.append((l, le))
        ok += (l > 0.8 and le < 0.15)
    tag = "RANDOM graph" if randomize else "SPATIAL graph"
    L = np.array([r[0] for r in rows])
    print(f"{tag:14s}  mode fraction {ok}/12   bg_layer median {np.median(L):.3f}", flush=True)
