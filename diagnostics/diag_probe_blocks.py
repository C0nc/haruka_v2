import sys; sys.path.insert(0,"/mnt/HDD2B/Users/yancui/Haruka")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import run_salient_eval as ev
from run_haruka_spatial_example import make_synthetic
from haruka_spatial import HarukaSpatial
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
S = ev.SHARED

def probe(X, y, seed=0):
    X = StandardScaler().fit_transform(np.nan_to_num(X)); y = np.asarray(y).astype(str)
    sc = [balanced_accuracy_score(y[te], KNeighborsClassifier(15).fit(X[tr],y[tr]).predict(X[te]))
          for tr,te in StratifiedKFold(3, shuffle=True, random_state=seed).split(X,y)]
    return float(np.mean(sc))

for attach in ("background","salient"):
    accs = {k: [] for k in ("z_layer","s_layer","sp_layer","z_lesion","sp_niche")}
    for seed in (0,1,2):
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
        h.extract_rep(n_layer=2); o = h.output_adata
        z,s,sp = o.obsm["background_rep"], o.obsm["salient_rep"], o.obsm["spatial_rep"]
        L,Le,Ni = o.obs["layer"].values, o.obs["lesion"].values, o.obs["niche"].values
        accs["z_layer"].append(probe(z,L));   accs["s_layer"].append(probe(s,L))
        accs["sp_layer"].append(probe(sp,L)); accs["z_lesion"].append(probe(z,Le))
        accs["sp_niche"].append(probe(sp,Ni))
    print(f"\n== attach='{attach}' == (3 seeds, mean)  chance: layer .333  lesion/niche .500")
    for k,v in accs.items(): print(f"   {k:10s} {np.mean(v):.3f}")
