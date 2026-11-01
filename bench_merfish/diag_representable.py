"""Can the GAT operator EXPRESS the true mapping, given the true latents?

THE DECISION THIS ANSWERS. The data comes from our own hypothesised generative
process, so a failure of `spatial_attach="both"` to fit it is exactly one of:

    (A) ARCHITECTURE  the GAT cannot represent the true neighbour->spatial map
    (B) TRAINING      it can represent it, but optimisation does not find it
    (C) MISMATCH      the simulation still violates what the architecture assumes

(A) and (C) are settleable WITHOUT training the model, by handing the GAT the
TRUE latents and fitting only the GAT to the TRUE target. If that fit succeeds,
the operator is expressive enough and the generative process is compatible --
so any remaining failure is (B). If it fails, the ceiling is structural and no
amount of training or tuning will reach it.

Two mappings are tested, one per pair:

    neighbours' z_true   -> z_sp_true      (background: arch -> niche)
    neighbours' s_true   -> s_sp_true      (salient:    sal_intr -> sal_domain)

and three operators of increasing power, to localise WHERE any ceiling sits:

    kernel-mean + linear   what a single attention layer can do at best if the
                           attention weights were uniform
    NeighborGATv2          the actual operator, trained directly on the target
    neighbour-mean + MLP   an unconstrained non-linear read of the same
                           neighbourhood, i.e. the information ceiling

Reading the result:
  * MLP high, GAT low      -> the information is in the neighbourhood but the GAT
                              cannot extract it. ARCHITECTURE (A).
  * MLP low                -> the target is not a function of the neighbourhood
                              the model sees. MISMATCH (C).
  * GAT high               -> operator is fine; failure in the full model is
                              TRAINING (B).

    python diag_representable.py --data data/simcnt_d1_s0.h5ad
"""

from __future__ import annotations

import argparse
import sys

import anndata as ad
import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "/home/yancui/haruka_v2/contrastivevi_patch")
from _module import NeighborGATv2  # noqa: E402


def knn(xy, sl, k):
    idx = np.zeros((len(xy), k), dtype=int)
    for s in np.unique(sl):
        m = np.where(sl == s)[0]
        _, i = cKDTree(xy[m]).query(xy[m], k=min(k, len(m) - 1) + 1)
        idx[m] = m[i[:, 1:k + 1]]          # self excluded, back to global indices
    return idx


def fit_gat(h_self, h_nbr, target, groups, n_in, n_out, epochs=400, seed=0,
            heads=4, hidden=0, width=None):
    """Train ONLY a NeighborGATv2 (+ optional MLP head) to predict the target.

    `width` widens the GAT's own output before the head, `hidden` adds a non-linear
    read on top. Together these separate three possible causes of the GAT falling
    short of the MLP ceiling: too few heads, too narrow an output, or the fact that
    a single attention layer is linear after pooling and the true mapping is not
    (it residualises against the cell's own class, then discretises).
    """
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    uniq = np.unique(groups)
    te_mask = groups == uniq[-1]
    tr_mask = ~te_mask
    w = width or n_out
    gat = NeighborGATv2(n_in, w, heads=heads).to(dev)
    head = (torch.nn.Sequential(torch.nn.Linear(w, hidden), torch.nn.ReLU(),
                                torch.nn.Linear(hidden, n_out)).to(dev)
            if hidden else torch.nn.Linear(w, n_out).to(dev))
    opt = torch.optim.Adam(list(gat.parameters()) + list(head.parameters()), lr=1e-2)
    HS = torch.tensor(h_self, dtype=torch.float32, device=dev)
    HN = torch.tensor(h_nbr, dtype=torch.float32, device=dev)
    Y = torch.tensor(target, dtype=torch.float32, device=dev)
    tr = torch.tensor(np.where(tr_mask)[0], device=dev)
    for _ in range(epochs):
        opt.zero_grad()
        sel = tr[torch.randint(0, len(tr), (4096,), device=dev)]
        loss = torch.nn.functional.mse_loss(head(gat(HS[sel], HN[sel], None)), Y[sel])
        loss.backward()
        opt.step()
    with torch.no_grad():
        te = torch.tensor(np.where(te_mask)[0], device=dev)
        pred = head(gat(HS[te], HN[te], None)).cpu().numpy()
    y = target[te_mask]
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def cv_r2(X, Y, groups, model):
    uniq = np.unique(groups)
    te = groups == uniq[-1]
    model.fit(X[~te], Y[~te])
    p = model.predict(X[te])
    y = Y[te]
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/simcnt_d1_s0.h5ad")
    ap.add_argument("--k", type=int, default=60)
    args = ap.parse_args()

    a = ad.read_h5ad(args.data)
    obs = a.obs
    xy = np.asarray(a.obsm["spatial"], float)
    sl = obs["slice_id"].astype(str).to_numpy()
    tr = (obs["condition"].astype(str) == "disease").to_numpy()
    idx = knn(xy, sl, args.k)

    # Rebuild the TRUE latents from the labels + codebooks the simulator used.
    import build_sim_counts as B
    seed = int(a.uns["sim2x2"]["seed"])
    dec = np.random.default_rng(40_000 + seed)
    A_arch = dec.normal(0, 1.4, (B.K_ARCH, B.D_ARCH))
    dec.normal(0, 1.0, B.K_ARCH)                    # niche_proj
    dec.normal(0, 1.0, B.K_SINT)                    # sdom_proj
    A_niche = dec.normal(0, 1.4, (B.K_NICHE, B.D_NICHE))
    A_sint = np.vstack([np.zeros((1, B.D_SINT)),
                        dec.normal(0, 1.4, (B.K_SINT - 1, B.D_SINT))])
    A_sdom = np.vstack([np.zeros((1, B.D_SDOM)),
                        dec.normal(0, 1.4, (B.K_SDOM - 1, B.D_SDOM))])

    pairs = [
        ("BACKGROUND  arch -> niche",
         A_arch[obs["gt_arch"].astype(int)], A_niche[obs["gt_niche"].astype(int)],
         np.ones(len(sl), bool)),
        ("SALIENT     sal_intr -> sal_domain",
         A_sint[obs["gt_sal_intrinsic"].astype(int)],
         A_sdom[obs["gt_sal_domain"].astype(int)], tr),
    ]

    print(f"{'pair':<30}{'GAT h4 w=out':>14}{'GAT h8':>9}{'GAT w=64':>10}"
          f"{'GAT w64+MLP':>13}{'MLP ceiling':>13}")
    for name, parent, child, mask in pairs:
        nbr = parent[idx]                       # (n, k, d_parent)
        nbrmean = nbr.mean(1)
        g = sl[mask]
        ni, no = parent.shape[1], child.shape[1]
        a = fit_gat(parent[mask], nbr[mask], child[mask], g, ni, no)
        b = fit_gat(parent[mask], nbr[mask], child[mask], g, ni, no, heads=8)
        c = fit_gat(parent[mask], nbr[mask], child[mask], g, ni, no, width=64)
        d = fit_gat(parent[mask], nbr[mask], child[mask], g, ni, no, width=64, hidden=128)
        mlp = cv_r2(np.hstack([nbrmean[mask], parent[mask]]), child[mask], g,
                    MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=300,
                                 random_state=0))
        print(f"{name.split()[0]:<30}{a:>14.3f}{b:>9.3f}{c:>10.3f}{d:>13.3f}{mlp:>13.3f}")

    print("\nMLP high + GAT low  -> ARCHITECTURE ceiling (A)")
    print("MLP low             -> generative process not a function of the "
          "neighbourhood the model sees (C)")
    print("GAT high            -> operator fine; remaining failure is TRAINING (B)")


if __name__ == "__main__":
    main()
