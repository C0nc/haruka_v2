"""Score representations against Haruka's 2x2 as a REPRESENTATION-BY-TRUTH matrix.

The single-number question ("did it find the domain?") was the wrong shape. With
`spatial_attach="both"` the model emits four latents that partition variation two
ways at once, and the claim under test is an ASSIGNMENT: each latent should carry
its own quadrant and not the other three.

                       intrinsic                spatial
    background   |     `background` -> gt_arch  `spatial` -> gt_niche
    salient      |     `salient` -> gt_sal_intrinsic
                                                `salient_spatial` -> gt_sal_domain

So the headline output is a 4x4 matrix of grouped, clusterer-free dependence;
ARI(GMM(representation), truth) is retained as a secondary readout. What matters is

    DIAGONAL   -- each latent recovers the factor it is responsible for
    OFF-DIAG   -- and does not also carry the other three

and the summary statistic is `assignment_margin` = mean(diagonal) - mean(off-diagonal).
A model that puts everything in one latent scores a high diagonal AND a high
off-diagonal and gets no credit, which is the failure the single-number version
could not see.

Methods with fewer representations fill what they can: contrastiveVI has two
(background, salient) and can address at most the intrinsic column; CellCharter
and PCA have one. Missing cells are NaN, not zero -- absence of a latent is not a
score of zero, it is a smaller claim.

The secondary ARI path sends every cell through the same PCA-50 + full-covariance
GMM as before, with k set to the true class count of that truth.
"""

from __future__ import annotations

import argparse
import json
import os

import anndata as ad
import numpy as np
import pandas as pd

from common import (ari, dependence, embed, gmm_on_embedding, kmeans_on_embedding, leiden_labels,
                    load_embeddings, nmi, probe_auroc_pre)

#: representation slot -> the quadrant it is responsible for
ASSIGNMENT = {
    "background": "gt_arch",
    "spatial": "gt_niche",
    "salient": "gt_sal_intrinsic",
    "salient_spatial": "gt_sal_domain",
}
TRUTHS = ["gt_arch", "gt_niche", "gt_sal_intrinsic", "gt_sal_domain"]
#: v4/v5 stores all four truths as categorical generative states. Older scorer
#: comments incorrectly described the two domain states as terciles of continuous
#: columns that the simulator never writes. The old fallback already scored them
#: categorically; make that actual contract explicit without changing any number.
DISCRETE = {name: True for name in TRUTHS}
CONTINUOUS_COL = {}
#: the salient-side truths only exist in the treated arm
TREATED_ONLY = {"gt_sal_intrinsic", "gt_sal_domain"}


def own_block_only(reps: dict, meta: dict) -> dict:
    """Slice each representation down to the cell's OWN latent, dropping the
    neighbourhood-aggregated blocks.

    WHY THIS MODE EXISTS. `HarukaSpatial.extract_rep` ends in
    `cc.gr.aggregate_neighbors(n_layers=3)`, which concatenates each latent with
    its 1-, 2- and 3-hop neighbourhood means. Three quarters of every saved
    representation is therefore smoothed copies of the latent rather than the
    latent -- and that is spatial smoothing applied a SECOND time, on top of the
    GAT that already pools neighbours inside the model.

    SIMVI does no such thing: `get_latent_representation` returns raw `q_m` /
    `qgat_m`, and its own clustering is `sc.pp.neighbors(use_rep='simvi_z')` +
    leiden on that raw latent. `edge_index` is used inside the model, never as
    post-hoc smoothing.

    Symptom this mode is meant to test: on the synthetic build, `salient` recovered
    `gt_sal_intrinsic` at probe AUROC 0.985 but GMM ARI 0.000, and the three failing
    quadrants all produced the SAME coarse blobby partition regardless of which
    latent or which model they came from -- the signature of the aggregated blocks
    dominating the variance.

    Block width is derived from `n_layer` in the run metadata rather than assumed,
    so this stays correct if the aggregation depth ever changes.
    """
    n_layer = int(meta.get("n_layer", 3))
    n_blocks = n_layer + 1
    out = {}
    for k, v in reps.items():
        if v.shape[1] % n_blocks:
            raise ValueError(
                f"{k}: width {v.shape[1]} is not divisible by n_layer+1={n_blocks}; "
                "cannot identify the own-latent block")
        out[k] = v[:, : v.shape[1] // n_blocks]
    return out


def score(data_path: str, emb_dir: str, method: str, seed: int = 0,
          rep_mode: str = "own", cluster_method: str = "gmm",
          resolution: float = 0.6, compute_ari: bool = True,
          compute_oracle: bool = True, rep_names: list[str] | None = None) -> dict:
    adata = ad.read_h5ad(data_path)
    emb = load_embeddings(emb_dir)
    meta = emb["meta"]

    if "obs_names" in emb:
        pos = pd.Index(adata.obs_names).get_indexer(pd.Index(emb["obs_names"].astype(str)))
        if (pos < 0).any():
            raise ValueError(f"{emb_dir}: {int((pos < 0).sum())} embedding rows not in data")
    else:
        pos = np.arange(adata.n_obs)

    obs = adata.obs.iloc[pos]
    treated = (obs["condition"].astype(str) == "disease").to_numpy()
    slice_id = obs["slice_id"].astype(str).to_numpy()
    cond = obs["condition"].astype(str).to_numpy()

    reps = {k: emb[k] for k in ("background", "salient", "spatial", "salient_spatial")
            if k in emb}
    if rep_names is not None:
        reps = {k: reps[k] for k in rep_names if k in reps}
    if rep_mode == "own":
        reps = own_block_only(reps, meta)
    sim = adata.uns["sim2x2"]

    row = dict(
        method=method, data=os.path.basename(data_path), emb_dir=emb_dir,
        delta=float(sim["delta"]), sim_seed=int(sim["seed"]),
        n_cells=int(len(pos)), score_seed=int(seed),
        n_reps=len(reps), reps=sorted(reps), rep_mode=rep_mode,
        cluster_method=cluster_method,
        resolution=(resolution if cluster_method == "leiden" else None),
        rep_widths={k: int(v.shape[1]) for k, v in reps.items()},
        runtime_s=meta.get("runtime_s"),
        ari_computed=bool(compute_ari), oracle_computed=bool(compute_oracle),
    )

    cache = {}

    def Z(name, mask_key, arr, mask):
        key = (name, mask_key)
        if key not in cache:
            cache[key] = embed(arr if mask is None else arr[mask], seed=seed)
        return cache[key]

    def labels(name, mask_key, arr, mask, k):
        key = (name, mask_key, k if cluster_method == "gmm" else "leiden")
        if key not in cache:
            if cluster_method == "leiden":
                # SIMVI's read-out: kNN graph on the RAW latent, no PCA, no
                # Gaussian assumption. Resolution-based, so k is not imposed.
                sub = arr if mask is None else arr[mask]
                cache[key] = leiden_labels(sub, resolution=resolution, seed=seed)
            elif cluster_method == "kmeans":
                sub = arr if mask is None else arr[mask]
                cache[key] = kmeans_on_embedding(sub, k, seed)
            else:
                cache[key] = gmm_on_embedding(Z(name, mask_key, arr, mask), k, seed)
        return cache[key]

    # --- the matrix ---------------------------------------------------------
    diag, offdiag = [], []
    for rname, arr in reps.items():
        for tname in TRUTHS:
            mask = treated if tname in TREATED_ONLY else None
            mask_key = "t" if tname in TREATED_ONLY else "all"
            truth = obs[tname].astype(str).to_numpy()
            truth = truth[mask] if mask is not None else truth
            if compute_ari:
                k = len(np.unique(truth))
                lab = labels(rname, mask_key, arr, mask, k)
                row[f"ARI__{rname}__{tname}"] = ari(lab, truth)

            # PRIMARY metric: clusterer-free dependence, comparable across factor
            # types. ARI above is retained but is only well posed for the discrete
            # factors.
            disc = DISCRETE.get(tname, True)
            ycol = CONTINUOUS_COL.get(tname)
            if not disc and ycol in obs.columns:
                yv = obs[ycol].to_numpy()
            else:
                disc, yv = True, truth if disc else truth
            yv = yv[mask] if (mask is not None and len(yv) == len(obs)) else yv
            grp = slice_id[mask] if mask is not None else slice_id
            sub = arr if mask is None else arr[mask]
            v = dependence(sub, yv, grp, discrete=disc, seed=seed)
            row[f"DEP__{rname}__{tname}"] = v
            if ASSIGNMENT.get(rname) == tname:
                diag.append(v)
            else:
                offdiag.append(v)

        # A linear probe on the slot's own quadrant, as the sensitive read-out --
        # a latent can carry a factor without any GMM split landing on it.
        tname = ASSIGNMENT.get(rname)
        if tname:
            mask = treated if tname in TREATED_ONLY else None
            truth = obs[tname].astype(str).to_numpy()
            truth = truth[mask] if mask is not None else truth
            grp = slice_id[mask] if mask is not None else slice_id
            major = pd.Series(truth).value_counts().idxmax()
            row[f"AUROC__{rname}__{tname}"] = probe_auroc_pre(
                Z(rname, "t" if mask is not None else "all", arr, mask),
                truth != major, grp)

    if diag:
        row["diag_mean"] = float(np.mean(diag))

        # CEILING-NORMALISED DIAGONAL. The four quadrants do NOT have the same
        # attainable maximum: with the v5 generative process an oracle reaches
        # ~1.00 on cell type and background domain, 0.775 on the salient state, and
        # 0.640 on the salient domain (the last two are deliberately subtle). A raw
        # mean over the four therefore rewards a method for being good at the EASY
        # quadrants and quietly penalises one that is good at the hard ones -- a
        # bias in the metric, not in the methods.
        #
        # `diag_frac_ceiling` reports each quadrant as a fraction of what is
        # actually achievable and then averages, so 1.0 means "saturated every
        # quadrant". Ceilings are supplied per dataset via `uns['sim2x2']
        # ['quadrant_ceilings']` (written by measure_ceilings.py); without them the
        # column is omitted rather than silently falling back to 1.0, which would
        # reproduce the very bias this exists to remove.
        ceil = (adata.uns.get("sim2x2", {}) or {}).get("quadrant_ceilings")
        if ceil:
            fr = []
            for rname, tname in ASSIGNMENT.items():
                v = row.get(f"DEP__{rname}__{tname}")
                c = ceil.get(tname)
                if v is not None and c:
                    fr.append(min(1.0, float(v) / float(c)))
            if fr:
                row["diag_frac_ceiling"] = float(np.mean(fr))
        row["offdiag_mean"] = float(np.mean(offdiag)) if offdiag else float("nan")
        # The headline. High only if each latent owns its quadrant AND declines the
        # other three; a model that dumps everything into one latent cannot win it.
        # Now computed on the DEPENDENCE scale, not ARI.
        row["assignment_margin"] = row["diag_mean"] - row["offdiag_mean"]

        # ORACLE FLOOR. Zero off-diagonal is NOT the target: the ground-truth
        # factors are not independent of each other. `cell_type` is drawn at a
        # rate set by `bg_domain`, and `sal_state` at a rate set by `sal_domain`,
        # so a latent that recovers its own factor PERFECTLY and nothing else
        # still shows dependence on the neighbouring factor. The floor measures
        # exactly that: replace each latent with a one-hot encoding of the factor
        # it is supposed to own -- a perfect, leak-free representation -- and run
        # the identical off-diagonal cells through the identical metric.
        #
        # `excess_offdiag` is the number to read. It is what the model leaks OVER
        # AND ABOVE the structural coupling built into the generative process, so
        # 0 means a perfect decomposition and the raw `offdiag_mean` alone would
        # overstate the leak by the floor.
        if compute_oracle:
            oracle = []
            for rname in reps:
                src = ASSIGNMENT.get(rname)
                if src is None:
                    continue
                onehot = pd.get_dummies(obs[src].astype(str)).to_numpy(dtype=np.float32)
                for tname in TRUTHS:
                    if tname == src:
                        continue
                    mask = treated if (tname in TREATED_ONLY or src in TREATED_ONLY) else None
                    disc = DISCRETE.get(tname, True)
                    ycol = CONTINUOUS_COL.get(tname)
                    yv = (obs[ycol].to_numpy() if (not disc and ycol in obs.columns)
                          else obs[tname].astype(str).to_numpy())
                    disc = disc or ycol not in obs.columns
                    sub = onehot if mask is None else onehot[mask]
                    yy = yv if mask is None else yv[mask]
                    grp = slice_id if mask is None else slice_id[mask]
                    oracle.append(dependence(sub, yy, grp, discrete=disc, seed=seed))
            if oracle:
                row["oracle_offdiag"] = float(np.mean(oracle))
                row["excess_offdiag"] = row["offdiag_mean"] - row["oracle_offdiag"]

    # --- the leak tests that do not belong in the matrix ---------------------
    if compute_ari and "background" in reps and len(np.unique(cond)) > 1:
        bg = reps["background"]
        row["bg_ARI_condition"] = ari(labels("background", "all", bg, None, 2), cond)
        row["bg_ARI_slice"] = ari(
            labels("background", "all", bg, None, len(np.unique(slice_id))), slice_id)

    # A cell's own row is independent of its own niche by construction (self is
    # excluded from the kernel). So `gt_niche` recovered by an INTRINSIC latent is
    # not a success, it is evidence the spatial branch leaked into it -- reported
    # explicitly because the matrix alone reads it as a mere off-diagonal.
    if "background" in reps:
        row["niche_in_intrinsic_leak"] = row.get("ARI__background__gt_niche")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rep_mode", choices=["own", "aggregated"], default="own",
                    help="'own' (default) scores the RAW per-cell latent, which is the "
                         "object the generative process defines and what SIMVI returns; "
                         "'aggregated' scores the cellcharter 3-hop concatenation, which "
                         "measures the latent PLUS a post-hoc spatial denoiser")
    ap.add_argument("--cluster_method", choices=["gmm", "leiden", "kmeans"], default="gmm",
                    help="'gmm' = PCA-50 + full-cov GaussianMixture (this benchmark's "
                         "fixed protocol); 'leiden' = SIMVI's own read-out, kNN graph "
                         "on the raw latent then Leiden at `--resolution`")
    ap.add_argument("--resolution", type=float, default=0.6,
                    help="Leiden resolution; 0.6 is SIMVI's value")
    ap.add_argument("--skip_ari", action="store_true",
                    help="skip secondary GMM/Leiden ARI calculations")
    ap.add_argument("--skip_oracle", action="store_true",
                    help="skip repeated oracle-floor calculation (margin is unchanged)")
    ap.add_argument("--reps", nargs="+", choices=list(ASSIGNMENT), default=None,
                    help="score only selected representation slots")
    ap.add_argument("--out", default="results_2x2/metrics.jsonl")
    args = ap.parse_args()

    row = score(args.data, args.emb_dir, args.method, args.seed, args.rep_mode,
                args.cluster_method, args.resolution,
                compute_ari=not args.skip_ari, compute_oracle=not args.skip_oracle,
                rep_names=args.reps)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
