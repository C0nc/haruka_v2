#!/usr/bin/env python3
"""Re-audit (2026-10-16) of the simulated PEtracer lineage benchmark.

Read-only on existing data/results. Measures:
  1. that the TRUE kNN-20 cell-type composition is identical between arms;
  2. the size and structure of Haruka's predicted composition shift dp
     (salient_composition_shift.npy), which should be ~0 by construction;
  3. whether dp / neighbour-mean shift / s carry the planted lineage signal,
     against a coordinates-only baseline (clades are spatially contiguous);
  4. cell-level linear-probe recovery (5-fold CV balanced accuracy).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/yancui/haruka_v2")
OUT = ROOT / "bench_petracer_lineage/reaudit_2026-10-16"
K = 20

FITS = []
for i in range(1, 5):
    FITS.append(dict(
        name=f"M1-{i} pair (compgated lambda10, no batch)",
        data=ROOT / f"bench_petracer_lineage/data_m1t1_pairs_22k/petracer_M1T1_M1-{i}_paired_22k.h5ad",
        fit=ROOT / f"bench_petracer_lineage/results_compgated_lambda10/petracer_M1T1_M1-{i}_paired_22k/haruka_shared_full_niche_compgated_seed0",
    ))
FITS.append(dict(
    name="joint four replicates (sepheads, batch=graph_id)",
    data=ROOT / "bench_petracer_lineage/data_m1t1_joint/petracer_M1T1_four_condition_110k.h5ad",
    fit=ROOT / "results_real_compgated_lambda10_samplebatch_sepheads/petracer_M1T1_four_condition_110k/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0",
))


def knn_composition(coords, graphs, labels, cats):
    comp = np.zeros((len(labels), len(cats)), np.float32)
    nbr = np.full((len(labels), K), -1, np.int64)
    code = pd.Categorical(labels, categories=cats).codes
    for g in np.unique(graphs):
        idx = np.flatnonzero(graphs == g)
        tree = cKDTree(coords[idx])
        _, nn = tree.query(coords[idx], k=K + 1)
        nn = idx[nn[:, 1:]]
        nbr[idx] = nn
        for j in range(K):
            np.add.at(comp, (idx, code[nn[:, j]]), 1.0)
    return comp / K, nbr


def probe(x, y, seed=0):
    y = np.asarray(y).astype(str)
    keep = pd.Series(y).map(pd.Series(y).value_counts()) >= 10
    x, y = x[keep.to_numpy()], y[keep.to_numpy()]
    if len(np.unique(y)) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    pred = cross_val_predict(clf, x, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return float(balanced_accuracy_score(y, pred))


def main():
    rows, probes = [], []
    for spec in FITS:
        print("==", spec["name"], flush=True)
        a = ad.read_h5ad(spec["data"])
        names = np.load(spec["fit"] / "obs_names.npy", allow_pickle=True).astype(str)
        pos = pd.Index(a.obs_names.astype(str)).get_indexer(names)
        assert (pos >= 0).all()
        obs = a.obs.iloc[pos]
        coords = np.asarray(a.obsm["spatial"])[pos]
        meta = json.load(open(spec["fit"] / "meta.json"))
        cats = meta["cell_type_categories"]
        graphs = obs["graph_id"].astype(str).to_numpy()
        labels = obs["gt_background_celltype"].astype(str).to_numpy()
        comp_obs, nbr = knn_composition(coords, graphs, labels, cats)
        cond = obs["condition"].astype(str).to_numpy() == "disease"
        elig = cond & obs["lineage_effect_eligible"].astype(bool).to_numpy()

        # 1. TRUE composition difference between arms for exactly paired cells
        if "pair_id" in obs and (~cond).any():
            pid = obs["pair_id"].astype(str).to_numpy()
            sec = obs["section_id"].astype(str).to_numpy()
            key = pd.Series(np.arange(len(obs)), index=pd.MultiIndex.from_arrays([sec, pid]))
            ctrl = np.flatnonzero(~cond)
            ctrl_map = pd.Series(ctrl, index=pd.MultiIndex.from_arrays([sec[ctrl], pid[ctrl]]))
            ci = np.flatnonzero(cond)
            m = ctrl_map.reindex(pd.MultiIndex.from_arrays([sec[ci], pid[ci]])).to_numpy()
            ok = ~pd.isna(m)
            m = m[ok].astype(int)
            true_dp_l1 = np.abs(comp_obs[ci[ok]] - comp_obs[m]).sum(1)
            n_paired = int(ok.sum())
            true_dp_max = float(true_dp_l1.max()) if n_paired else float("nan")
        else:
            n_paired, true_dp_max = 0, float("nan")

        dp = np.load(spec["fit"] / "salient_composition_shift.npy")
        pbg = np.load(spec["fit"] / "background_composition.npy")
        pcd = np.load(spec["fit"] / "condition_composition.npy")
        nms = np.load(spec["fit"] / "salient_neighbor_mean_shift.npy")
        s = np.load(spec["fit"] / "salient.npy")
        z = np.load(spec["fit"] / "background.npy")
        l1 = np.abs(dp).sum(1)
        # fraction of the 20 nearest neighbours that received the planted effect
        frac_pert = elig[nbr].mean(1)
        recon_bg = np.abs(pbg - comp_obs).sum(1)
        recon_cd = np.abs(pcd - comp_obs).sum(1)
        # scale reference: L1 distance between random condition-cell compositions
        rng = np.random.default_rng(0)
        ci = np.flatnonzero(cond)
        rp = rng.permutation(ci)
        rand_l1 = np.abs(comp_obs[ci] - comp_obs[rp]).sum(1)
        mean_vec = dp[cond].mean(0)
        top = np.argsort(-np.abs(mean_vec))[:3]
        nonel = cond & ~elig
        near = cond & (frac_pert > 0.25)
        far = cond & (frac_pert == 0)
        auc_l1 = roc_auc_score(elig[cond], l1[cond])
        nm_l1 = np.abs(nms).sum(1)
        row = dict(
            fit=spec["name"],
            n_condition=int(cond.sum()), n_lineage=int(elig.sum()),
            n_exactly_paired=n_paired, true_dp_L1_max=true_dp_max,
            dp_L1_mean_condition=float(l1[cond].mean()),
            dp_L1_median_condition=float(np.median(l1[cond])),
            dp_L1_p95_condition=float(np.quantile(l1[cond], 0.95)),
            dp_L1_mean_lineage=float(l1[elig].mean()),
            dp_L1_mean_nonlineage=float(l1[nonel].mean()),
            dp_L1_mean_nbr_gt25pct_perturbed=float(l1[near].mean()) if near.any() else float("nan"),
            dp_L1_mean_nbr_0pct_perturbed=float(l1[far].mean()) if far.any() else float("nan"),
            dp_L1_AUROC_lineage_vs_not=float(auc_l1),
            spearman_dpL1_vs_frac_perturbed_nbrs=float(pd.Series(l1[cond]).corr(pd.Series(frac_pert[cond]), method="spearman")),
            spearman_nmshiftL1_vs_frac_perturbed_nbrs=float(pd.Series(nm_l1[cond]).corr(pd.Series(frac_pert[cond]), method="spearman")),
            mean_dp_vector_L1=float(np.abs(mean_vec).sum()),
            top_mean_dp_types="; ".join(f"{cats[t]} {mean_vec[t]:+.4f}" for t in top),
            ref_random_pair_comp_L1=float(rand_l1.mean()),
            recon_L1_D_z0_vs_observed=float(recon_bg[cond].mean()),
            recon_L1_D_zs_vs_observed=float(recon_cd[cond].mean()),
            s_norm_lineage=float(np.linalg.norm(s[elig], axis=1).mean()),
            s_norm_nonlineage=float(np.linalg.norm(s[nonel], axis=1).mean()),
            s_norm_control=float(np.linalg.norm(s[~cond], axis=1).mean()) if (~cond).any() else float("nan"),
        )
        rows.append(row)
        print(json.dumps(row, indent=1), flush=True)

        # 3/4. linear-probe recovery of family/clade on lineage cells
        if spec["name"].startswith("joint"):
            secs = obs["section_id"].astype(str).to_numpy()
        reps = {
            "dp (composition shift, 18)": dp,
            "neighbour-mean shift (288)": nms,
            "s latent (32)": s,
            "z latent (32)": z,
            "xy coordinates only (2)": coords,
            "observed kNN composition (18)": comp_obs,
        }
        for truth in ("gt_lineage_family", "gt_lineage_clade"):
            y = obs[truth].astype(str).to_numpy()
            for rname, x in reps.items():
                if spec["name"].startswith("joint"):
                    # within-section probes, averaged (coordinates not comparable across sections)
                    vals = [probe(x[elig & (secs == sct)], y[elig & (secs == sct)]) for sct in np.unique(secs[elig])]
                    val = float(np.nanmean(vals))
                else:
                    val = probe(x[elig], y[elig])
                probes.append(dict(fit=spec["name"], truth=truth, representation=rname, balanced_accuracy_5fold=val))
                print(truth, rname, round(val, 3), flush=True)
        # chance level
        for truth in ("gt_lineage_family", "gt_lineage_clade"):
            y = obs[truth].astype(str).to_numpy()[elig]
            vc = pd.Series(y).value_counts()
            probes.append(dict(fit=spec["name"], truth=truth, representation="chance (1/n_classes)",
                               balanced_accuracy_5fold=1.0 / int((vc >= 10).sum())))

    pd.DataFrame(rows).to_csv(OUT / "dp_leakage_by_fit.csv", index=False)
    pd.DataFrame(probes).to_csv(OUT / "lineage_probe_by_representation.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
