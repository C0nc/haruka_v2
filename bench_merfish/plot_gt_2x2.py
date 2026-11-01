"""Render the four ground-truth factors on the tissue, one row per quadrant.

The point of the figure is to make three claims checkable by eye that are
otherwise only assertions in `self_check_2x2.py`:

  * `gt_arch` and `gt_niche` appear in BOTH arms -- they are background factors.
  * `gt_sal_intrinsic` and `gt_sal_domain` appear ONLY in the treated arm; on
    control slices they are the counterfactual labels, drawn faint, and no effect
    was applied to those cells.
  * `gt_sal_domain` is a contiguous region while `gt_sal_intrinsic` is spatial
    salt-and-pepper. That contrast is the whole reason the intrinsic factor exists:
    it is the one neighbourhood smoothing cannot help with.

Columns pair a control and a treated slice from the same timepoint, so the
control/treated difference is visible within a row rather than across the figure.

    python plot_gt_2x2.py --data data/sim2x2_d2.4_s0.h5ad --out results_2x2/gt_2x2.png
"""

from __future__ import annotations

import argparse

import anndata as ad
import numpy as np

FACTORS = [
    ("gt_arch", "background x intrinsic\ngt_arch (real anatomy, 7 subregions)", False),
    ("gt_niche", "background x spatial\ngt_niche (ligand balance, terciles)", False),
    ("gt_sal_intrinsic", "salient x intrinsic\ngt_sal_intrinsic (i.i.d. in space)", True),
    ("gt_sal_domain", "salient x spatial\ngt_sal_domain (implanted domains)", True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sim2x2_d2.4_s0.h5ad")
    ap.add_argument("--out", default="results_2x2/gt_2x2.png")
    ap.add_argument("--max_cells", type=int, default=6000)
    ap.add_argument("--point", type=float, default=2.2)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    adata = ad.read_h5ad(args.data)
    obs = adata.obs
    xy = np.asarray(adata.obsm["spatial"])
    sl = obs["slice_id"].astype(str).to_numpy()
    treated = (obs["condition"].astype(str) == "disease").to_numpy()

    # one control/treated pair from each of two timepoints
    tps = list(dict.fromkeys(obs["timepoint"].astype(str)))
    cols = []
    for tp in tps[:2]:
        for cond in ("control", "disease"):
            m = (obs["timepoint"].astype(str) == tp) & (obs["condition"].astype(str) == cond)
            if m.any():
                cols.append((tp, cond, np.unique(sl[m.to_numpy()])[0]))

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(FACTORS), len(cols),
                             figsize=(3.1 * len(cols), 3.0 * len(FACTORS)), squeeze=False)
    cmap = plt.get_cmap("tab10")

    for r, (key, title, salient_only) in enumerate(FACTORS):
        lab = obs[key].astype(str).to_numpy()
        codes = {v: i for i, v in enumerate(sorted(set(lab)))}
        for c, (tp, cond, slice_name) in enumerate(cols):
            ax = axes[r][c]
            idx = np.where(sl == slice_name)[0]
            if len(idx) > args.max_cells:
                idx = rng.choice(idx, args.max_cells, replace=False)
            colors = [cmap(codes[v] % 10) for v in lab[idx]]
            # On control slices a salient factor is COUNTERFACTUAL -- the label
            # exists so the false-positive probe is defined, but no effect was
            # applied. Drawn faint so the figure cannot be misread as showing a
            # perturbation there.
            alpha = 0.22 if (salient_only and cond == "control") else 0.9
            ax.scatter(xy[idx, 0], xy[idx, 1], c=colors, s=args.point,
                       linewidths=0, alpha=alpha)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            if r == 0:
                ax.set_title(f"{tp}\n{cond}", fontsize=9)
            if c == 0:
                ax.set_ylabel(title, fontsize=8)
            if salient_only and cond == "control":
                ax.text(0.5, 0.02, "counterfactual (no effect applied)",
                        transform=ax.transAxes, ha="center", fontsize=6, color="0.35")

    sim = adata.uns["sim2x2"]
    fig.suptitle(
        f"2x2 ground truth — {args.data.split('/')[-1]}  (delta={sim['delta']}, "
        f"seed={sim['seed']}; {adata.n_obs} cells, {adata.n_vars} genes)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[plot_gt] wrote {args.out}")

    for key, _, sal in FACTORS:
        v = obs[key].astype(str)
        sub = v[treated] if sal else v
        print(f"  {key:18s} classes={sorted(set(v))} "
              f"counts={dict(sub.value_counts())}")


if __name__ == "__main__":
    main()
