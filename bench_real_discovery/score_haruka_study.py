"""Quantify and interpret the four Haruka slots on a prepared real study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SLOTS = ["background", "spatial", "salient", "salient_spatial"]
PROGRAMS = {
    "dkd": {
        "positive": ["HAVCR1", "VCAM1", "VIM", "LCN2", "CCL2", "SPP1", "COL1A1", "COL3A1", "FN1"],
        "negative": ["LRP2", "CUBN", "SLC34A1", "ALDOB"],
    },
    "colitis": {
        "positive": ["Il1b", "Cxcl10", "S100a8", "S100a9", "Tnf", "Ccl2", "Lcn2", "Clu", "Reg3b", "Reg3g"],
        "negative": ["Apoa1", "Krt20", "Slc26a3", "Car1"],
    },
    "ad": {
        "positive": ["Gfap", "Apoe", "Cst7", "Lpl", "Tyrobp", "Trem2", "C1qa", "C1qb", "C1qc", "Spp1", "Itgax"],
        "negative": ["P2ry12", "Tmem119", "Cx3cr1"],
    },
}


def cv_classification(z, y, groups, seed=0):
    y, groups = np.asarray(y).astype(str), np.asarray(groups).astype(str)
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan, 0
    vals = []
    for tr, te in GroupKFold(n_splits=n_splits).split(z, y, groups):
        common = np.intersect1d(np.unique(y[tr]), np.unique(y[te]))
        use = np.isin(y[te], common)
        if len(common) < 2 or use.sum() < 20:
            continue
        model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0, class_weight="balanced"))
        model.fit(z[tr], y[tr])
        vals.append(balanced_accuracy_score(y[te][use], model.predict(z[te][use])))
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (np.nan, np.nan, 0)


def cv_regression(z, y, groups):
    y, groups = np.asarray(y, float), np.asarray(groups).astype(str)
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        return np.nan, np.nan, 0
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(z, y, groups):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(z[tr], y[tr]); pred[te] = model.predict(z[te])
    ok = np.isfinite(pred)
    return float(r2_score(y[ok], pred[ok])), float(spearmanr(y[ok], pred[ok]).statistic), n_splits


def _lognorm_matrix(a: ad.AnnData, rows=None, cols=None):
    x = a.layers["counts"] if "counts" in a.layers else a.X
    if rows is not None:
        x = x[rows]
    if cols is not None:
        x = x[:, cols]
    x = x.toarray() if sp.issparse(x) else np.asarray(x)
    lib = np.maximum(x.sum(1, keepdims=True), 1.0)
    return np.log1p(x / lib * 1e4).astype(np.float32)


def program_scores(a: ad.AnnData, study: str):
    index = {str(g): i for i, g in enumerate(a.var_names)}
    pos = [index[g] for g in PROGRAMS[study]["positive"] if g in index]
    neg = [index[g] for g in PROGRAMS[study]["negative"] if g in index]
    used = [*pos, *neg]
    if not used:
        return np.zeros(a.n_obs, np.float32), np.zeros(a.n_obs, np.float32), [], []
    x = _lognorm_matrix(a, cols=used)
    own = x[:, :len(pos)].mean(1) if pos else np.zeros(a.n_obs)
    if neg:
        own -= x[:, len(pos):].mean(1)
    local = np.zeros(a.n_obs, dtype=np.float32)
    graph = a.obs.graph_id.astype(str).to_numpy()
    xy = np.asarray(a.obsm["spatial"])
    for value in np.unique(graph):
        idx = np.flatnonzero(graph == value)
        kk = min(9, len(idx))
        nbr = NearestNeighbors(n_neighbors=kk).fit(xy[idx]).kneighbors(return_distance=False)
        local[idx] = own[idx[nbr]].mean(1)
    return own.astype(np.float32), local, [a.var_names[i] for i in pos], [a.var_names[i] for i in neg]


def group_center(x: np.ndarray, obs: pd.DataFrame, keys=("graph_id", "cell_type")):
    group = obs[list(keys)].astype(str).agg("|".join, axis=1).to_numpy()
    out = x.copy()
    for value in np.unique(group):
        mask = group == value
        out[mask] -= out[mask].mean(0, keepdims=True)
    return out


def transfer_probe(z, y, graph, train_graph, test_graph):
    tr, te = graph == train_graph, graph == test_graph
    labels = np.intersect1d(np.unique(y[tr]), np.unique(y[te]))
    tr &= np.isin(y, labels); te &= np.isin(y, labels)
    if len(labels) < 2 or tr.sum() < 20 or te.sum() < 20:
        return np.nan
    model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0, class_weight="balanced"))
    model.fit(z[tr], y[tr])
    return float(balanced_accuracy_score(y[te], model.predict(z[te])))


def gene_discoveries(a, reps, study, out, seed=0, max_cells=15000):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(a.n_obs, min(max_cells, a.n_obs), replace=False))
    obs = a.obs.iloc[idx]
    x = _lognorm_matrix(a, rows=idx)
    graph = obs.graph_id.astype(str).to_numpy()
    xy = np.asarray(a.obsm["spatial"])[idx]
    local = np.zeros_like(x)
    for value in np.unique(graph):
        gi = np.flatnonzero(graph == value)
        kk = min(9, len(gi))
        nbr = NearestNeighbors(n_neighbors=kk).fit(xy[gi]).kneighbors(return_distance=False)
        local[gi] = x[gi[nbr]].mean(1)
    views = {
        "background": x,
        "spatial": local,
        "salient": x - local,
        "salient_spatial": local,
    }
    rows = []
    target = obs.condition.astype(str).to_numpy() == "disease"
    for slot in SLOTS:
        use = np.ones(len(idx), bool) if slot in ("background", "spatial") else target
        xx, zz = views[slot][use], reps[slot][idx][use]
        if slot in ("salient", "salient_spatial"):
            subobs = obs.iloc[np.flatnonzero(use)]
            xx = group_center(xx, subobs)
            zz = group_center(zz, subobs)
        xx = (xx - xx.mean(0)) / (xx.std(0) + 1e-6)
        zz = (zz - zz.mean(0)) / (zz.std(0) + 1e-6)
        corr = (zz.T @ xx) / max(len(xx) - 1, 1)
        dim = np.abs(corr).argmax(0)
        score = corr[dim, np.arange(corr.shape[1])]
        top = np.argsort(-np.abs(score))[:30]
        for rank, gene in enumerate(top, 1):
            rows.append({"slot": slot, "rank": rank, "gene": str(a.var_names[gene]),
                         "correlation": float(score[gene]), "latent_dim": int(dim[gene]),
                         "expression_view": {"background":"own", "spatial":"local",
                            "salient":"own_minus_local", "salient_spatial":"local_target"}[slot]})
    pd.DataFrame(rows).to_csv(out / "slot_gene_discoveries.csv", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--emb", required=True)
    p.add_argument("--study", choices=list(PROGRAMS), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-probe", type=int, default=40000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(args.data)
    names = np.load(Path(args.emb) / "obs_names.npy", allow_pickle=True).astype(str)
    order = pd.Series(np.arange(a.n_obs), index=a.obs_names.astype(str)).loc[names].to_numpy()
    a = a[order].copy()
    reps = {slot: np.load(Path(args.emb) / f"{slot}.npy").astype(np.float32) for slot in SLOTS}
    rng = np.random.default_rng(args.seed)
    groups_all = a.obs.specimen_id.astype(str).to_numpy()
    per_group = max(1, args.max_probe // len(np.unique(groups_all)))
    idx = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(groups_all == g), min(per_group, (groups_all == g).sum()), replace=False)
        for g in np.unique(groups_all)
    ]))
    obs = a.obs.iloc[idx]
    groups = obs.specimen_id.astype(str).to_numpy()
    target = obs.condition.astype(str).to_numpy() == "disease"
    own_score, local_score, pos_genes, neg_genes = program_scores(a, args.study)
    intrinsic_score = own_score - local_score

    factors = {
        "cell_type": obs.cell_type.astype(str).to_numpy(),
        "cell_subtype": obs.cell_subtype.astype(str).to_numpy(),
        "tissue_domain": obs.tissue_domain.astype(str).to_numpy(),
        "condition": obs.condition.astype(str).to_numpy(),
        "stage": obs.stage.astype(str).to_numpy(),
    }
    rows = []
    for slot, zfull in reps.items():
        z = zfull[idx]
        for factor, y in factors.items():
            use = target if factor == "stage" and len(np.unique(y[target])) > 1 else np.ones(len(idx), bool)
            mean, sd, folds = cv_classification(z[use], y[use], groups[use], args.seed)
            rows.append({"slot": slot, "factor": "target_stage" if factor == "stage" and use is target else factor,
                         "score": mean, "fold_sd": sd, "n_folds": folds,
                         "chance": 1.0 / len(np.unique(y[use])), "metric": "balanced_accuracy"})
        centered = group_center(z, obs)
        for factor, values in (("intrinsic_program", intrinsic_score[idx]), ("local_program", local_score[idx])):
            use = target
            r2, rho, folds = cv_regression(centered[use], values[use], groups[use])
            rows.append({"slot": slot, "factor": factor, "score": rho, "r2": r2,
                         "fold_sd": np.nan, "n_folds": folds, "chance": 0.0,
                         "metric": "cross_specimen_spearman"})
        for plaque in ("plaque_tau", "plaque_abeta", "plaque_any"):
            if plaque in obs and obs[plaque].nunique() > 1:
                y = obs[plaque].astype(str).to_numpy(); use = target
                mean, sd, folds = cv_classification(z[use], y[use], groups[use], args.seed)
                rows.append({"slot": slot, "factor": plaque, "score": mean, "fold_sd": sd,
                             "n_folds": folds, "chance": 0.5, "metric": "balanced_accuracy"})
    probes = pd.DataFrame(rows)
    probes.to_csv(out / "cross_specimen_probes.csv", index=False)

    transfer = []
    if args.study == "ad":
        graph = a.obs.graph_id.astype(str).to_numpy()
        for slot, z in reps.items():
            for factor in ("cell_type", "plaque_tau", "plaque_abeta", "plaque_any"):
                y = a.obs[factor].astype(str).to_numpy()
                train_graph = "disease_r1" if factor.startswith("plaque") else "control_r1"
                transfer.append({"slot": slot, "factor": factor,
                    "train_graph": train_graph, "test_graph": "disease_r2",
                    "balanced_accuracy": transfer_probe(z, y, graph, train_graph, "disease_r2")})
        pd.DataFrame(transfer).to_csv(out / "ad_replicate2_transfer.csv", index=False)

    wanted = ["cell_type", "tissue_domain", "intrinsic_program", "local_program"]
    matrix = probes[probes.factor.isin(wanted)].pivot(index="factor", columns="slot", values="score")
    matrix = matrix.reindex(index=wanted, columns=SLOTS)
    matrix.to_csv(out / "four_slot_factor_matrix.csv")
    expected = dict(zip(wanted, SLOTS))
    margins = []
    for factor, slot in expected.items():
        own = float(matrix.loc[factor, slot]); other = float(matrix.loc[factor].drop(slot).max())
        margins.append({"factor": factor, "expected_slot": slot, "score": own,
                        "best_other": other, "assignment_margin": own - other})
    pd.DataFrame(margins).to_csv(out / "assignment_margins.csv", index=False)

    cluster_rows = []
    cidx = rng.choice(idx, min(25000, len(idx)), replace=False)
    for slot, z in reps.items():
        zz = StandardScaler().fit_transform(z[cidx])
        for factor in ("cell_type", "tissue_domain", "stage"):
            y = a.obs.iloc[cidx][factor].astype(str).to_numpy(); k = len(np.unique(y))
            pred = MiniBatchKMeans(k, batch_size=2048, n_init=10, random_state=args.seed).fit_predict(zz)
            cluster_rows.append({"slot": slot, "factor": factor,
                                 "ari": adjusted_rand_score(y, pred), "n_clusters": k})
    pd.DataFrame(cluster_rows).to_csv(out / "clustering_ari.csv", index=False)
    gene_discoveries(a, reps, args.study, out, args.seed)

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax)
    ax.set_title(f"Haruka four-slot discovery: {args.study}")
    fig.tight_layout(); fig.savefig(out / "four_slot_factor_matrix.png", dpi=220); plt.close(fig)

    summary = {
        "study": args.study, "n_cells": int(a.n_obs), "n_probe": int(len(idx)),
        "training_supervision": "binary control/target arm only",
        "withheld_from_training": "stage, cell type/subtype, tissue domain, marker program, plaque labels",
        "evaluation": "grouped by biological specimen; AD includes disease-r1 to disease-r2 plaque transfer",
        "program_positive_genes_present": [str(x) for x in pos_genes],
        "program_negative_genes_present": [str(x) for x in neg_genes],
        "assignment_margins": margins,
        "ad_transfer": transfer,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(matrix.to_string(), flush=True)
    print(pd.DataFrame(margins).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
