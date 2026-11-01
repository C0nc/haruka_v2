"""Validate the BENCHMARK before validating any method on it.

`haruka_sim.py` has a `self_check` that fails loudly if the simulator does not
have the property the whole benchmark rests on. This is the same idea for the
MERFISH build. Four assertions; if any fails, every downstream number is
meaningless and should not be reported.

  A. THE SIGNAL EXISTS. An oracle probe -- logistic regression on the 30 genes
     that were actually perturbed -- must separate domain from non-domain cells
     in the TREATED arm. If it cannot, delta is below the noise floor of this
     tissue and a method that scores 0 is being blamed for the simulator.

  B. THE SIGNAL IS NOT THERE WITHOUT TREATMENT. The same oracle probe on the
     CONTROL arm, against the same counterfactual domain geometry, must sit at
     ~0.5. Anything above means the domain shape correlates with real biology and
     the "detection" being measured downstream is partly free.

  C. ANATOMY DOES NOT GIVE IT AWAY. A probe using only the anatomical subregion
     label must not predict domain membership. This is B's coarser sibling and is
     the specific failure that would flatter CellCharter and PCA.

  D. THE NULL ARM IS NULL. On the delta=0 build, A must also collapse to ~0.5.

  E. THE DOMAIN IS ACTUALLY A SPATIAL DOMAIN. Fraction of a cell's 12 nearest
     neighbours within its own slice that share its domain label. This is here
     because the stratified thresholding that fixes B and C pulls in the opposite
     direction -- forcing equal prevalence in every subregion x celltype stratum
     could in principle shred a contiguous blob into salt and pepper, and a
     rendered scatter at 1.2-pt markers cannot be trusted to tell the difference
     (it looked shredded; it is not). Two reference points are reported with it:
     a random label of the same prevalence, and the anatomical `subregion` label
     itself. The implanted domain must beat the random floor decisively, and in
     practice it also beats real anatomy.

Usage:  python self_check.py --data data/sim_d0.6_s0.h5ad data/sim_d0_s0.h5ad
"""

from __future__ import annotations

import argparse
import json

import anndata as ad
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder

from common import probe_auroc

TOL_NULL = 0.58     # an AUROC below this counts as "no signal"
TOL_SIGNAL = 0.62   # an AUROC above this counts as "signal present"


def _lognorm(adata):
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    lib = np.maximum(X.sum(1, keepdims=True), 1.0)
    return np.log1p(X / lib * 1e4)


def spatial_contiguity(labels, xy, slice_id, k: int = 12) -> float:
    """Mean fraction of a cell's k nearest same-slice neighbours sharing its label."""
    from sklearn.neighbors import NearestNeighbors
    out = []
    for s in np.unique(slice_id):
        m = np.where(slice_id == s)[0]
        _, idx = NearestNeighbors(n_neighbors=k + 1).fit(xy[m]).kneighbors(xy[m])
        out.append((labels[m][idx[:, 1:]] == labels[m][:, None]).mean())
    return float(np.mean(out))


def check(path: str, signal_delta: float = 1.2) -> dict:
    adata = ad.read_h5ad(path)
    W = np.asarray(adata.uns["program_W"])
    prog = np.abs(W).sum(0) > 0

    Xl = _lognorm(adata)
    obs = adata.obs
    treated = (obs["condition"].astype(str) == "disease").to_numpy()
    domain = obs["gt_domain"].astype(int).to_numpy()
    sl = obs["slice_id"].astype(str).to_numpy()
    delta = float(adata.uns["sim"]["delta"])

    anat_oh = OneHotEncoder(sparse_output=False).fit_transform(
        obs["subregion"].astype(str).to_numpy().reshape(-1, 1))

    res = dict(
        path=path, delta=delta,
        A_oracle_treated=probe_auroc(Xl[np.ix_(treated, prog)], domain[treated] > 0, sl[treated]),
        B_oracle_control=probe_auroc(Xl[np.ix_(~treated, prog)], domain[~treated] > 0, sl[~treated]),
        C_anatomy_only=probe_auroc(anat_oh[treated], domain[treated] > 0, sl[treated]),
        full_transcriptome_treated=probe_auroc(Xl[treated], domain[treated] > 0, sl[treated]),
        domain_anatomy_nmi=float(adata.uns["sim"]["domain_anatomy_nmi"]),
    )

    xy = np.asarray(adata.obsm["spatial"])
    rng = np.random.default_rng(0)
    res["E_contiguity_domain"] = spatial_contiguity(domain, xy, sl)
    res["E_contiguity_random_floor"] = spatial_contiguity(
        (rng.random(adata.n_obs) < (domain > 0).mean()).astype(int), xy, sl)
    res["E_contiguity_anatomy_ref"] = spatial_contiguity(
        obs["subregion"].astype(str).to_numpy(), xy, sl)

    # The incremental signal: how much the oracle gains in the treated arm over the
    # same probe on untreated cells with the same geometry. This, not the raw
    # treated AUROC, is what the injected effect actually contributes.
    res["incremental_signal"] = res["A_oracle_treated"] - res["B_oracle_control"]

    fails = []
    if delta >= signal_delta and not res["A_oracle_treated"] > TOL_SIGNAL:
        fails.append(f"A: oracle probe cannot see the implant in the treated arm "
                     f"(AUROC {res['A_oracle_treated']:.3f} <= {TOL_SIGNAL}) at delta={delta}, "
                     f"which was declared detectable; the effect size sweep is mis-scaled")
    if 0 < delta < signal_delta and res["A_oracle_treated"] <= TOL_SIGNAL:
        res["below_floor"] = True   # informational: this arm is a sensitivity probe
    if delta == 0 and not res["A_oracle_treated"] < TOL_NULL:
        fails.append(f"D: null arm is not null (AUROC {res['A_oracle_treated']:.3f})")
    if not res["B_oracle_control"] < TOL_NULL:
        fails.append(f"B: the domain geometry is visible in expression WITHOUT treatment "
                     f"(AUROC {res['B_oracle_control']:.3f}); detection downstream is partly free")
    if not res["C_anatomy_only"] < TOL_NULL:
        fails.append(f"C: anatomy alone predicts the domain (AUROC {res['C_anatomy_only']:.3f}); "
                     f"a non-contrastive anatomy clusterer gets a head start")
    margin = res["E_contiguity_domain"] - res["E_contiguity_random_floor"]
    if margin < 0.15:
        fails.append(f"E: the implanted domain is not spatially contiguous "
                     f"(k-NN agreement {res['E_contiguity_domain']:.3f} vs random floor "
                     f"{res['E_contiguity_random_floor']:.3f}); the stratified thresholding "
                     f"has shredded it and this is no longer a spatial-domain benchmark")
    res["fails"] = fails
    res["ok"] = not fails
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--signal_delta", type=float, default=1.2,
                    help="deltas at or above this MUST be detectable by the oracle probe; "
                         "smaller ones are sensitivity arms and are only reported")
    args = ap.parse_args()
    bad = 0
    for p in args.data:
        r = check(p, signal_delta=args.signal_delta)
        print(json.dumps(r, indent=2))
        if not r["ok"]:
            bad += 1
            for f in r["fails"]:
                print(f"  !! {f}")
    print(f"\n[self_check] {len(args.data) - bad}/{len(args.data)} builds passed")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
