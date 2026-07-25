import sys; sys.path.insert(0,"/mnt/HDD2B/Users/yancui/Haruka")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
import run_salient_eval as ev
from run_haruka_spatial_example import make_synthetic
from haruka_spatial import HarukaSpatial
S = ev.SHARED

def fit(attach, seed=0):
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

def decoder_block_norms(mod, widths, names):
    # first Linear of DecoderSCVI consumes the concatenated latent; column-norms per block
    W = None
    for m in mod.decoder.modules():
        if isinstance(m, torch.nn.Linear):
            W = m.weight.detach().cpu().numpy(); break
    out, off = {}, 0
    for w, n in zip(widths, names):
        out[n] = float(np.linalg.norm(W[:, off:off+w]) / np.sqrt(w))  # per-dim norm
        off += w
    return out

for attach in ("background", "salient"):
    h = fit(attach)
    mod = h.model.module
    with torch.no_grad():
        h.extract_rep(n_layer=2)
    o = h.output_adata
    z  = o.obsm["background_rep"]; s = o.obsm["salient_rep"]; sp = o.obsm["spatial_rep"]
    print(f"\n===== spatial_attach='{attach}' =====")
    print(f"  latent per-dim std   z {z.std():.3f}   s {s.std():.3f}   z_sp {sp.std():.3f}")
    nz = decoder_block_norms(mod, [S['latent_dim'], S['latent_dim'], 16], ["z","s","z_sp"])
    tot = sum(nz.values())
    print("  decoder per-dim weight norm (share of total):")
    for k,v in nz.items(): print(f"     {k:5s} {v:7.4f}   {v/tot*100:5.1f}%")
