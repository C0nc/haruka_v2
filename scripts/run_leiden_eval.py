"""Does spatially-constrained Leiden beat KMeans for Haruka domain calling? (RNA only)"""
import sys; sys.path.insert(0,"/mnt/HDD2B/Users/yancui/Haruka")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import run_salient_eval as ev
from run_haruka_spatial_example import make_synthetic
from haruka_spatial import HarukaSpatial
from haruka_leiden import spatial_leiden
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
S = ev.SHARED
WEIGHTS = (0.0, 0.25, 0.5, 0.75)

def spatial_coherence(lab, coords, groups):
    """Fraction of spatial neighbours sharing a label -- higher = less speckled."""
    from sklearn.neighbors import NearestNeighbors
    tot = agree = 0
    for g in np.unique(groups):
        i = np.where(groups == g)[0]
        nn = NearestNeighbors(n_neighbors=min(9, len(i))).fit(coords[i])
        _, ind = nn.kneighbors(coords[i])
        agree += (lab[i][ind[:, 1:]] == lab[i][:, None]).sum(); tot += ind[:, 1:].size
    return agree / max(tot, 1)

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
    coords = np.column_stack([o.obs["array_col"], o.obs["array_row"]]).astype(float) \
             if "array_col" in o.obs else np.asarray(o.obsm["spatial"], float)[:, -2:]
    groups = o.obs["slice_id"].astype(str).values
    for rep, truth, k in (("X_cellcharter_background","layer",3), ("X_cellcharter_salient","lesion",2)):
        E = o.obsm[rep]; T = o.obs[truth].astype(str).values
        km = KMeans(k, n_init=10, random_state=seed).fit_predict(StandardScaler().fit_transform(E))
        res.setdefault((rep,"kmeans"), []).append((ari(T, km), spatial_coherence(km, coords, groups), k))
        for w in WEIGHTS:
            # match cluster count to KMeans by bisecting resolution
            lo, hi, lab = 0.05, 3.0, None
            for _ in range(12):
                mid = (lo+hi)/2
                l = spatial_leiden(E, coords, groups, spatial_weight=w, resolution=mid, seed=seed)
                nc = l.max()+1
                lab = l
                if nc == k: break
                if nc < k: lo = mid
                else: hi = mid
            res.setdefault((rep,f"leiden_w{w}"), []).append(
                (ari(T, lab), spatial_coherence(lab, coords, groups), lab.max()+1))

print(f"{'representation':30s}{'method':14s}{'ARI':>9s}{'coherence':>12s}{'n_clust':>9s}")
for (rep, meth), v in res.items():
    a = np.mean([x[0] for x in v]); c = np.mean([x[1] for x in v]); n = np.mean([x[2] for x in v])
    print(f"{rep:30s}{meth:14s}{a:9.3f}{c:12.3f}{n:9.1f}")
