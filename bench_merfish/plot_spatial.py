"""Show, on the tissue, what each method's salient representation actually found.

The metric tables say how much a method recovered; this says what it recovered,
which is the part that catches a method scoring well for the wrong reason -- a
"salient" partition that is visibly a cortical-layer map, or a domain call that
tracks one animal and not the others.

Rows are methods, columns are the four treated slices. The first row is ground
truth. Colour is the PCA-50 + full-cov GMM label on the salient representation,
matched to the truth labels by Hungarian assignment so the colours are comparable
across panels rather than arbitrary per-panel cluster ids.

    python plot_spatial.py --data data/sim_d2.4_s0.h5ad --results results/d2.4_s0 \
        --out results/spatial_d2.4_s0.png
"""

from __future__ import annotations

import argparse
import glob
import os

import anndata as ad
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from common import embed, gmm_on_embedding, load_embeddings


def match_labels(pred, truth):
    """Relabel `pred` to maximise agreement with `truth` (Hungarian on the
    contingency table). Without this every panel gets its own arbitrary cluster
    numbering and the figure is unreadable."""
    pu, tu = np.unique(pred), np.unique(truth)
    C = np.zeros((len(pu), len(tu)))
    for i, p in enumerate(pu):
        for j, t in enumerate(tu):
            C[i, j] = np.sum((pred == p) & (truth == t))
    r, c = linear_sum_assignment(-C)
    mapping = {pu[i]: tu[j] for i, j in zip(r, c)}
    return np.array([mapping.get(p, -1) for p in pred])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", required=True, help="results/<tag> directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_cells", type=int, default=4000, help="per slice, for legibility")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    adata = ad.read_h5ad(args.data)
    obs = adata.obs
    treated = (obs["condition"].astype(str) == "disease").to_numpy()
    truth = obs["gt_domain"].astype(int).to_numpy()
    xy = np.asarray(adata.obsm["spatial"])
    slices = sorted(obs.loc[treated, "slice_id"].astype(str).unique())
    k = int(adata.uns["sim"]["n_domains"]) + 1

    dirs = sorted(glob.glob(os.path.join(args.results, f"*_seed{args.seed}")))
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "salient.npy"))]
    if not dirs:
        raise SystemExit(f"no embeddings under {args.results}")

    names = ["ground truth"] + [os.path.basename(d).replace(f"_seed{args.seed}", "")
                                for d in dirs]
    labels = [truth[treated]]
    for d in dirs:
        emb = load_embeddings(d)
        sal = emb["salient"]
        if "obs_names" in emb:
            pos = pd.Index(adata.obs_names).get_indexer(pd.Index(emb["obs_names"].astype(str)))
            full = np.zeros((adata.n_obs, sal.shape[1]), dtype=np.float32)
            full[pos] = sal
            sal = full
        lab = gmm_on_embedding(embed(sal[treated], seed=args.seed), k, args.seed)
        labels.append(match_labels(lab, truth[treated]))

    sl_of = obs.loc[treated, "slice_id"].astype(str).to_numpy()
    xy_t = xy[treated]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(len(names), len(slices),
                             figsize=(3.0 * len(slices), 2.9 * len(names)),
                             squeeze=False)
    cmap = plt.get_cmap("tab10")
    for r, (name, lab) in enumerate(zip(names, labels)):
        for c, sl in enumerate(slices):
            ax = axes[r][c]
            m = np.where(sl_of == sl)[0]
            if len(m) > args.max_cells:
                m = rng.choice(m, args.max_cells, replace=False)
            ax.scatter(xy_t[m, 0], xy_t[m, 1], c=[cmap(int(v) % 10) for v in lab[m]],
                       s=1.2, linewidths=0)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            if r == 0:
                ax.set_title(sl.replace("_disease", ""), fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=8)
    fig.suptitle(f"salient partition vs implanted domains — {os.path.basename(args.data)}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
