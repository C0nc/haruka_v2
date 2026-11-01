"""Score one method's saved representations against the implanted ground truth.

Every number below comes out of the same PCA-50 + full-covariance-GMM in
`common.py`, so differences between methods are differences between
representations and nothing else.

WHAT IS BEING ASKED. Three questions, and a method has to answer all three -- a
method that wins one by losing another has not solved the problem:

  1. DOES THE SALIENT REPRESENTATION FIND THE IMPLANTED DOMAIN?
     `sal_ARI_domain` / `sal_NMI_domain` (GMM with k = 1 + n_domains, treated
     cells) and `sal_AUROC_domain` (leave-one-animal-out linear probe). The probe
     is the sensitive read-out: a representation can carry the domain linearly
     while no GMM split lands on it.

  2. IS THE BACKGROUND REPRESENTATION CLEAN?
     `bg_ARI_anat` (should be HIGH -- it should recover the 7 anatomical
     subregions), `bg_ARI_condition` (should be ~0 -- condition must not leak
     in), `bg_ARI_slice` (should be low -- animal identity is nuisance), and
     `bg_AUROC_domain` (should be ~0.5 -- the injected effect belongs in salient).

  3. IS THE SALIENT REPRESENTATION SPECIFIC, OR IS IT JUST ANATOMY / JUST NOISE?
     `sal_ARI_anat` is the contamination number: a "salient" latent that is
     really a second copy of the tissue map scores high here and should be
     discounted no matter what it does on question 1. And the delta=0 build is
     the false-positive arm -- every question-1 number computed there is a
     measurement of how much structure the method invents.

SINGLE-REPRESENTATION METHODS. PCA+GMM and CellCharter produce one embedding, not
a background/salient pair. They are scored by putting that single embedding into
BOTH slots. That is the honest comparison: it is precisely their claim that one
spatial-domain map answers the question, and the numbers show what that costs.
"""

from __future__ import annotations

import argparse
import json
import os

import anndata as ad
import numpy as np
import pandas as pd

from common import (ari, embed, gmm_on_embedding, load_embeddings, nmi,
                    probe_auroc_pre)

ANAT_KEY = "subregion"


def score(data_path: str, emb_dir: str, method: str, seed: int = 0) -> dict:
    adata = ad.read_h5ad(data_path)
    emb = load_embeddings(emb_dir)
    meta = emb["meta"]

    obs = adata.obs
    if "obs_names" in emb:
        # Runners may reorder or subset (CINEMA-OT works per timepoint). Align on
        # names rather than trusting positions.
        pos = pd.Index(adata.obs_names).get_indexer(pd.Index(emb["obs_names"].astype(str)))
        if (pos < 0).any():
            raise ValueError(f"{emb_dir}: {int((pos < 0).sum())} embedding rows not in data")
    else:
        pos = np.arange(adata.n_obs)

    obs = obs.iloc[pos]
    treated = (obs["condition"].astype(str) == "disease").to_numpy()
    domain = obs["gt_domain"].astype(int).to_numpy()
    anat = obs[ANAT_KEY].astype(str).to_numpy()
    slice_id = obs["slice_id"].astype(str).to_numpy()
    cond = obs["condition"].astype(str).to_numpy()

    n_dom = int(adata.uns["sim"]["n_domains"])
    k_dom = n_dom + 1
    k_anat = len(np.unique(anat))

    bg = emb.get("background")
    sal = emb.get("salient")
    row = dict(
        method=method, data=os.path.basename(data_path), emb_dir=emb_dir,
        delta=float(adata.uns["sim"]["delta"]), sim_seed=int(adata.uns["sim"]["seed"]),
        delta_mode=str(adata.uns["sim"].get("delta_mode", "constant")),
        domain_anatomy_nmi=float(adata.uns["sim"]["domain_anatomy_nmi"]),
        n_cells=int(len(pos)), score_seed=int(seed),
        single_rep=bool(meta.get("single_rep", False)),
        runtime_s=meta.get("runtime_s"),
    )

    # The PCA-50 embedding of a 72k x 300 matrix is not cheap and each GMM refits
    # it; compute one embedding per (representation, cell subset) and reuse.
    cache = {}

    def Z(arr, mask, key):
        if key not in cache:
            cache[key] = embed(arr if mask is None else arr[mask], seed=seed)
        return cache[key]

    def clusters(arr, mask, key, k):
        ck = (key, k)
        if ck not in cache:
            cache[ck] = gmm_on_embedding(Z(arr, mask, key), k, seed)
        return cache[ck]

    # --- question 1 + 3: the salient side, on treated cells -----------------
    if sal is not None and treated.sum() > 0:
        lab_dom = clusters(sal, treated, "sal_t", k_dom)
        row["sal_ARI_domain"] = ari(lab_dom, domain[treated])
        row["sal_NMI_domain"] = nmi(lab_dom, domain[treated])
        row["sal_ARI_anat"] = ari(clusters(sal, treated, "sal_t", k_anat), anat[treated])
        row["sal_AUROC_domain"] = probe_auroc_pre(
            Z(sal, treated, "sal_t"), domain[treated] > 0, slice_id[treated])
        # Per-domain detection: an average over two programs can hide one being
        # missed entirely.
        for d in range(1, n_dom + 1):
            row[f"sal_AUROC_d{d}"] = probe_auroc_pre(
                Z(sal, treated, "sal_t"), domain[treated] == d, slice_id[treated])
        # The control arm carries the COUNTERFACTUAL domain label -- the same
        # geometry with no effect applied. Any AUROC above 0.5 here is the method
        # reading tissue position, not treatment response. Note this is vacuous for
        # models that zero the salient latent on controls by construction
        # (Haruka, contrastiveVI, and our CINEMA-OT wrapper all do); the delta=0
        # build is the false-positive test that applies to every method.
        if (~treated).sum() > 0 and np.ptp(sal[~treated]) > 0:
            row["sal_AUROC_ctrl_fp"] = probe_auroc_pre(
                Z(sal, ~treated, "sal_c"), domain[~treated] > 0, slice_id[~treated])

    # --- question 2: the background side ------------------------------------
    if bg is not None:
        row["bg_ARI_anat"] = ari(clusters(bg, treated, "bg_t", k_anat), anat[treated])
        row["bg_ARI_domain"] = ari(clusters(bg, treated, "bg_t", k_dom), domain[treated])
        row["bg_AUROC_domain"] = probe_auroc_pre(
            Z(bg, treated, "bg_t"), domain[treated] > 0, slice_id[treated])
        if len(np.unique(cond)) > 1:
            row["bg_ARI_condition"] = ari(clusters(bg, None, "bg_all", 2), cond)
            row["bg_ARI_anat_all"] = ari(clusters(bg, None, "bg_all", k_anat), anat)
            row["bg_ARI_slice"] = ari(
                clusters(bg, None, "bg_all", len(np.unique(slice_id))), slice_id)

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/metrics.jsonl")
    args = ap.parse_args()

    row = score(args.data, args.emb_dir, args.method, args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
