"""What concretely differs inside the model between attach=background and attach=salient?

No hypothesis. Just instrument the quantities that could plausibly differ and look.
"""
import sys; sys.path.insert(0,"/mnt/HDD2B/Users/yancui/Haruka")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
import run_salient_eval as ev
from run_haruka_spatial_example import make_synthetic
from haruka_spatial import HarukaSpatial, NBR_IDX_KEY
S = ev.SHARED

def fit(attach, seed):
    out = make_synthetic(seed=seed); rna = out[0] if isinstance(out,tuple) else out
    rna.layers["counts"] = np.asarray(rna.X).copy()
    h = HarukaSpatial(rna, cov_gene_num=S["cov_num"], seed=seed, k_spatial=S["k_spatial"])
    h.construct_cov(); h.prepare_spatial_graph()
    h.setup_model(latent_dim=S["latent_dim"], gene_likelihood="poisson",
                  condition_key="condition", condition_label="disease", batch_key="slice_id",
                  n_spatial_latent=16, spatial_attach=attach,
                  reg_to_use=S["reg_to_use"], lam_mi=S["lam_mi"],
                  kl_spatial_weight=S["kl_spatial_weight"])
    h.train(max_epochs=S["max_epochs"], batch_size=S["batch_size"], plan_kwargs=dict(lr=S["lr"]))
    return h

for attach in ("background","salient"):
    rows = []
    for seed in (0,1,2):
        h = fit(attach, seed); h.extract_rep(n_layer=2); o = h.output_adata
        ad = h.input_adata
        idx = np.asarray(ad.obsm[NBR_IDX_KEY])
        z, s, sp = o.obsm["background_rep"], o.obsm["salient_rep"], o.obsm["spatial_rep"]
        src = z if attach == "background" else s          # what the GAT reads
        # 1. how much does the GAT's INPUT vary between neighbours vs globally?
        #    (if neighbours are near-identical in that space, the pooled output is near-constant)
        n = min(len(src), idx.shape[0])
        nb = idx[:n]
        d_nb = np.mean([np.linalg.norm(src[i] - src[nb[i][nb[i] < n]], axis=1).mean()
                        for i in range(n) if (nb[i] < n).any()])
        d_glob = np.linalg.norm(src[np.random.default_rng(0).permutation(n)] - src, axis=1).mean()
        # 2. how much of the spatial latent's variance is between-cell vs a constant offset?
        rows.append(dict(nbr_dist=d_nb, glob_dist=d_glob, ratio=d_nb/max(d_glob,1e-9),
                         src_std=float(src.std()), sp_std=float(sp.std()),
                         sp_rank=int(np.linalg.matrix_rank(sp - sp.mean(0), tol=1e-3))))
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(f"\n== attach='{attach}' ==  (3 seeds)")
    print(f"  GAT input: neighbour distance {m['nbr_dist']:.4f} vs global {m['glob_dist']:.4f}"
          f"   ratio {m['ratio']:.3f}   <- low ratio = neighbours look alike in this space")
    print(f"  GAT input std {m['src_std']:.4f}    spatial-latent std {m['sp_std']:.4f}")
    print(f"  effective rank of spatial latent (of 16): {m['sp_rank']:.1f}")
