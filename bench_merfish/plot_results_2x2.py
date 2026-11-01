"""Show what each Haruka latent actually recovered, beside the factor it owns.

One row per quadrant of the 2x2. Left column is ground truth; each further column
is one model's GMM partition of the latent RESPONSIBLE for that quadrant, on the
same treated slice. Labels are Hungarian-matched to the truth so colours mean the
same thing across panels -- without that every panel gets its own arbitrary
cluster numbering and the figure is unreadable.

The ARI for each panel is printed in its title, so the picture and the number in
the scoring matrix can be checked against each other. A panel that looks right but
scores ~0 is the detection-versus-clusterability gap; a panel that scores well but
looks like a different factor is the model mixing quadrants.

    python plot_results_2x2.py --data data/simcnt_d1_s0.h5ad \
        --results results_fixed --methods haruka_both haruka_background
"""

from __future__ import annotations

import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from common import embed, gmm_on_embedding, kmeans_on_embedding, load_embeddings
from score_2x2 import own_block_only

# quadrant -> (representation that owns it, truth column, pretty label)
QUADRANTS = [
    ("background", "gt_arch", "background x intrinsic\nbackground -> gt_arch"),
    ("spatial", "gt_niche", "background x spatial\nspatial -> gt_niche"),
    ("salient", "gt_sal_intrinsic", "salient x intrinsic\nsalient -> gt_sal_intrinsic"),
    ("salient_spatial", "gt_sal_domain", "salient x spatial\nsalient_spatial -> gt_sal_domain"),
]


def match(pred, truth):
    """Relabel `pred` to maximise agreement with `truth` (Hungarian)."""
    pu, tu = np.unique(pred), np.unique(truth)
    C = np.zeros((len(pu), len(tu)))
    for i, p in enumerate(pu):
        for j, t in enumerate(tu):
            C[i, j] = np.sum((pred == p) & (truth == t))
    r, c = linear_sum_assignment(-C)
    m = {pu[i]: tu[j] for i, j in zip(r, c)}
    return np.array([m.get(p, -1) for p in pred])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--methods", nargs="+", default=["haruka_both"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_cells", type=int, default=6000)
    ap.add_argument("--rep_mode", choices=["own", "aggregated"], default="own",
                    help="'own' (default) plots the RAW per-cell latent, which is what "
                         "the generative process defines and what SIMVI returns")
    ap.add_argument("--cluster_method", choices=["gmm", "kmeans"], default="gmm",
                    help="GMM uses PCA-50; kmeans uses the standardized raw GAT readout")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    adata = ad.read_h5ad(args.data)
    obs = adata.obs
    treated = (obs["condition"].astype(str) == "disease").to_numpy()
    xy = np.asarray(adata.obsm["spatial"])
    sl = obs["slice_id"].astype(str).to_numpy()
    # one treated slice: the salient quadrants are only defined there
    slice_name = sorted(np.unique(sl[treated]))[0]

    tag = os.path.basename(args.data).replace("sim_", "").replace(".h5ad", "")
    embs = {}
    for m in args.methods:
        for root in (os.path.join(args.results, tag), args.results):
            d = os.path.join(root, f"{m}_seed{args.seed}")
            if os.path.exists(os.path.join(d, "meta.json")):
                e = load_embeddings(d)
                reps = {k: e[k] for k in
                        ("background", "salient", "spatial", "salient_spatial") if k in e}
                # RAW per-cell latent by default -- the object the generative process
                # defines. The cellcharter 3-hop concatenation is a post-hoc spatial
                # denoiser and corresponds to nothing in the generative model.
                if args.rep_mode == "own":
                    reps = own_block_only(reps, e["meta"])
                embs[m] = reps
                break
        else:
            print(f"[plot] no embeddings for {m}")

    ncol = 1 + len(embs)
    fig, axes = plt.subplots(len(QUADRANTS), ncol,
                             figsize=(3.3 * ncol, 3.1 * len(QUADRANTS)), squeeze=False)
    cmap = plt.get_cmap("tab10")
    rng = np.random.default_rng(0)

    for r, (rep, truth_col, title) in enumerate(QUADRANTS):
        truth_all = obs[truth_col].astype(str).to_numpy()
        k = len(np.unique(truth_all))
        show = np.where((sl == slice_name))[0]
        if len(show) > args.max_cells:
            show = rng.choice(show, args.max_cells, replace=False)
        codes = {v: i for i, v in enumerate(sorted(set(truth_all)))}

        ax = axes[r][0]
        ax.scatter(xy[show, 0], xy[show, 1],
                   c=[cmap(codes[v] % 10) for v in truth_all[show]], s=2.0, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_ylabel(title, fontsize=8)
        if r == 0:
            ax.set_title("ground truth", fontsize=9)

        for c, (m, e) in enumerate(embs.items(), start=1):
            ax = axes[r][c]
            if rep not in e:
                ax.text(0.5, 0.5, f"{m}\nhas no `{rep}`", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="0.45")
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(m, fontsize=9)
                continue
            # cluster on the SAME cells the scorer uses: treated arm for salient
            # quadrants, all cells otherwise
            mask = treated if truth_col in ("gt_sal_intrinsic", "gt_sal_domain") else np.ones(len(sl), bool)
            lab = np.full(len(sl), -1)
            if args.cluster_method == "kmeans":
                pred = kmeans_on_embedding(e[rep][mask], k, args.seed)
            else:
                pred = gmm_on_embedding(embed(e[rep][mask], seed=args.seed), k, args.seed)
            lab[mask] = match(pred, truth_all[mask])
            ari = adjusted_rand_score(truth_all[mask], lab[mask])
            sel = show[lab[show] >= 0]
            ax.scatter(xy[sel, 0], xy[sel, 1],
                       c=[cmap(codes.get(str(v), 0) % 10) for v in lab[sel]],
                       s=2.0, linewidths=0)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            ax.set_title(f"{m if r == 0 else ''}\nARI {ari:.3f}", fontsize=8)

    sim = adata.uns["sim2x2"]
    fig.suptitle(f"recovered vs truth — {os.path.basename(args.data)} "
                 f"(delta={sim['delta']}, seed={sim['seed']}, slice {slice_name})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or os.path.join(args.results, f"recovered_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
