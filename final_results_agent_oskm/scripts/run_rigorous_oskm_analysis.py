#!/usr/bin/env python3
"""Rigorous post-fit OSKM analysis for the completed full-cohort Haruka run.

Young controls are used only to define an external evaluation direction.  Haruka
is not retrained.  All spatial operations are bounded by slice_id and all
prediction assessments hold out one matched paired_batch / OSKM mouse at a time.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse, stats
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    normalized_mutual_info_score,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


PROGRAMS = {
    "reprogramming_stemness": ["Pou5f1", "Klf4", "Myc", "Sox2", "Tert", "Nr2e1", "Prom1"],
    "cell_cycle_dna_repair": ["Mki67", "Pcna", "Mcm2", "Top2a", "Cdk1", "Rrm2", "Pclaf", "H2afx", "Parp1", "Brca1", "Ercc1", "Atm"],
    "senescence_stress": ["Cdkn2a", "Cdkn1a", "Gdf15", "Sod2", "Jun", "Fos", "Btg2", "Trp53"],
    "interferon_inflammation": ["Il1a", "Il1b", "Tnf", "Il6", "Ifng", "Ifit1", "Ifitm1", "Ifitm3", "Stat1", "B2m", "H2-D1", "H2-K1", "C3", "Nfkb1", "Socs3"],
    "microglial_phagolysosomal": ["C1qa", "C1qb", "Cx3cr1", "Csf1r", "Itgam", "Lyz2", "Fcrls", "Hexb", "Apoe", "Gmfb", "Grn"],
    "astrocyte_reactivity": ["Gfap", "Clu", "C3", "Apoe", "Aqp4", "S100a6", "Sox9", "Slc1a3"],
    "vascular_activation": ["Pecam1", "Cldn5", "Flt1", "Flt4", "Apln", "Efnb2", "Kcnj8", "Acta2", "Vcam1"],
    "oligodendrocyte_myelin": ["Mog", "Opalin", "Cldn11", "Olig1", "Olig2", "Gpr17", "Enpp2", "Apod", "Ptprz1"],
    "neuronal_plasticity": ["Dcx", "Sox11", "Stmn2", "Tubb3", "Dlx1", "Dlx6os1", "Reln", "Btg2"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--young", required=True, type=Path)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--cc-dir", required=True, type=Path)
    p.add_argument("--domain-assignments", required=True, type=Path)
    p.add_argument("--existing-umap", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clusters", type=int, default=8)
    p.add_argument("--response-neighbors", type=int, default=15)
    return p.parse_args()


def log_normalize(matrix) -> sparse.csr_matrix:
    x = sparse.csr_matrix(matrix, dtype=np.float32).copy()
    library = np.asarray(x.sum(axis=1)).ravel()
    x = x.multiply((10_000.0 / np.maximum(library, 1.0))[:, None]).tocsr()
    x.data = np.log1p(x.data)
    return x


def safe_mean(x: sparse.csr_matrix, rows: np.ndarray) -> np.ndarray:
    return np.asarray(x[rows].mean(axis=0)).ravel().astype(np.float64)


def safe_var(x: sparse.csr_matrix, rows: np.ndarray, mean: np.ndarray) -> np.ndarray:
    second = np.asarray(x[rows].power(2).mean(axis=0)).ravel()
    return np.maximum(second - mean * mean, 0.0)


def projection(x: sparse.csr_matrix, rows: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    denom = float(direction @ direction)
    if denom <= 1e-12:
        return np.full(len(rows), np.nan, dtype=np.float32)
    return ((np.asarray(x[rows].dot(direction)).ravel() - float(origin @ direction)) / denom).astype(np.float32)


def response_scores(
    x: sparse.csr_matrix,
    xy: sparse.csr_matrix,
    obs: pd.DataFrame,
    young_obs: pd.DataFrame,
    genes: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matched-pair directions plus leave-pair-out and variance-scaled sensitivity."""
    condition = obs["condition"].astype(str).to_numpy()
    pair = obs["paired_batch"].astype(str).to_numpy()
    celltype = obs["cell_type"].astype(str).to_numpy()
    subregion = obs["subregion"].astype(str).to_numpy()
    strata = np.char.add(np.char.add(celltype, "::"), subregion)
    yct = young_obs["cell_type"].astype(str).to_numpy()
    ysr = young_obs["subregion"].astype(str).to_numpy()
    ystrata = np.char.add(np.char.add(yct, "::"), ysr)
    n = len(obs)
    out = pd.DataFrame(index=obs.index)
    for key in ["score_unscaled", "score_scaled", "score_loo_old"]:
        out[key] = np.full(n, np.nan, dtype=np.float32)
    out["direction_level"] = "unscored"
    directions: list[dict] = []

    def fill_level(group_name: str, ctype: str, region: str | None, level: str) -> None:
        yi = np.flatnonzero((yct == ctype) if region is None else (ystrata == group_name))
        if len(yi) < 20:
            return
        young_mean = safe_mean(xy, yi)
        young_var = safe_var(xy, yi, young_mean)
        for p in np.unique(pair):
            member = (celltype == ctype) if region is None else (strata == group_name)
            ci = np.flatnonzero(member & (pair == p) & (condition == "control"))
            ti = np.flatnonzero(member & (pair == p) & (condition == "oskm"))
            missing = ti[np.isnan(out["score_unscaled"].to_numpy()[ti])]
            if len(ci) < 20 or len(missing) < 5:
                continue
            rows = np.concatenate([ci, missing])
            old_mean = safe_mean(x, ci)
            old_var = safe_var(x, ci, old_mean)
            direction = young_mean - old_mean
            out.iloc[rows, out.columns.get_loc("score_unscaled")] = projection(x, rows, old_mean, direction)
            sd = np.sqrt(np.maximum(0.5 * (old_var + young_var), 0.0625))
            weighted_direction = direction / sd
            scaled_origin = old_mean / sd
            # Algebraically equivalent to projecting x/sd in scaled space.
            denom = float(weighted_direction @ weighted_direction)
            scaled = (np.asarray(x[rows].dot(weighted_direction / sd)).ravel() - float(scaled_origin @ weighted_direction)) / max(denom, 1e-12)
            out.iloc[rows, out.columns.get_loc("score_scaled")] = scaled.astype(np.float32)
            other_ci = np.flatnonzero(member & (pair != p) & (condition == "control"))
            if len(other_ci) >= 60:
                loo_mean = safe_mean(x, other_ci)
                loo_direction = young_mean - loo_mean
                out.iloc[rows, out.columns.get_loc("score_loo_old")] = projection(x, rows, loo_mean, loo_direction)
            out.iloc[rows, out.columns.get_loc("direction_level")] = level
            top = np.argsort(np.abs(direction))[-15:][::-1]
            for rank, gi in enumerate(top, 1):
                directions.append(
                    {
                        "paired_batch": p,
                        "stratum": group_name,
                        "level": level,
                        "rank_abs": rank,
                        "gene": str(genes[gi]),
                        "young_minus_old_logexpr": float(direction[gi]),
                        "scaled_loading": float(weighted_direction[gi]),
                        "n_old": int(len(ci)),
                        "n_young": int(len(yi)),
                    }
                )

    target_strata = np.unique(strata[condition == "oskm"])
    for group_name in target_strata:
        ctype, region = group_name.split("::", 1)
        fill_level(group_name, ctype, region, "cell_type_x_subregion")
    unresolved_types = np.unique(celltype[(condition == "oskm") & out["score_unscaled"].isna().to_numpy()])
    for ctype in unresolved_types:
        fill_level(f"celltype_fallback::{ctype}", ctype, None, "cell_type_fallback")
    return out, pd.DataFrame(directions)


def spatial_average(values: np.ndarray, spatial_xy: np.ndarray, samples: np.ndarray, k: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float32)
    for sample in np.unique(samples):
        rows = np.flatnonzero((samples == sample) & np.isfinite(values))
        if not len(rows):
            continue
        take = min(k, len(rows))
        _, local = cKDTree(spatial_xy[rows]).query(spatial_xy[rows], k=take, workers=1)
        if local.ndim == 1:
            local = local[:, None]
        result[rows] = np.nanmean(values[rows][local], axis=1)
    return result


def spatial_average_matrix(values: np.ndarray, spatial_xy: np.ndarray, samples: np.ndarray, k: int) -> np.ndarray:
    """Reuse one slice-bounded neighbor graph for several response variants."""
    result = np.full(values.shape, np.nan, dtype=np.float32)
    for sample in np.unique(samples):
        sample_rows = np.flatnonzero(samples == sample)
        if not len(sample_rows):
            continue
        take = min(k, len(sample_rows))
        _, local = cKDTree(spatial_xy[sample_rows]).query(spatial_xy[sample_rows], k=take, workers=1)
        if local.ndim == 1:
            local = local[:, None]
        for column in range(values.shape[1]):
            matrix = values[sample_rows, column]
            neighbor_values = matrix[local]
            count = np.isfinite(neighbor_values).sum(axis=1)
            total = np.nansum(neighbor_values, axis=1)
            valid = count > 0
            result[sample_rows[valid], column] = (total[valid] / count[valid]).astype(np.float32)
    return result


def calibrate_responders(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = []
    status_columns = []
    variants = {
        "main_q95": ("local_score_unscaled", 0.95),
        "main_q90": ("local_score_unscaled", 0.90),
        "main_q975": ("local_score_unscaled", 0.975),
        "main_q99": ("local_score_unscaled", 0.99),
        "raw_q95": ("score_unscaled", 0.95),
        "scaled_q95": ("local_score_scaled", 0.95),
        "loo_old_q95": ("local_score_loo_old", 0.95),
    }
    frame["calibration_stratum"] = (
        frame["paired_batch"].astype(str) + "::" + frame["cell_type"].astype(str) + "::" + frame["subregion"].astype(str)
    )
    is_control = frame["condition"].astype(str).eq("control")
    is_oskm = ~is_control
    for name, (column, quantile) in variants.items():
        status = np.full(len(frame), False, dtype=bool)
        for key, idx in frame.groupby("calibration_stratum", sort=True, observed=True).indices.items():
            idx = np.asarray(idx)
            ci = idx[is_control.to_numpy()[idx] & np.isfinite(frame[column].to_numpy()[idx])]
            ti = idx[is_oskm.to_numpy()[idx] & np.isfinite(frame[column].to_numpy()[idx])]
            if len(ci) < 20 or len(ti) < 5:
                continue
            threshold = float(np.quantile(frame[column].to_numpy()[ci], quantile))
            status[ti] = frame[column].to_numpy()[ti] > threshold
            if name == "main_q95":
                median = float(np.median(frame[column].to_numpy()[ci]))
                mad = float(stats.median_abs_deviation(frame[column].to_numpy()[ci], scale="normal"))
                frame.loc[frame.index[ti], "null_median"] = median
                frame.loc[frame.index[ti], "null_sd_robust"] = max(mad, 1e-4)
            thresholds.append(
                {
                    "variant": name,
                    "score_column": column,
                    "quantile": quantile,
                    "calibration_stratum": key,
                    "n_control_null": len(ci),
                    "n_oskm": len(ti),
                    "threshold": threshold,
                    "oskm_responder_rate": float(status[ti].mean()),
                }
            )
        frame[f"responder_{name}"] = status & is_oskm.to_numpy()
        status_columns.append(f"responder_{name}")
    frame["null_z_response"] = (frame["local_score_unscaled"] - frame["null_median"]) / frame["null_sd_robust"]
    return frame, pd.DataFrame(thresholds)


def balanced_rows(groups: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(groups) <= limit:
        return np.arange(len(groups))
    rng = np.random.default_rng(seed)
    values = np.unique(groups)
    quota = max(1, limit // len(values))
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(groups == v), min(quota, np.sum(groups == v)), replace=False) for v in values]))


def fit_domains(matrix: np.ndarray, samples: np.ndarray, n_clusters: int, seed: int, max_fit: int = 50_000) -> np.ndarray:
    rows = balanced_rows(samples, max_fit, seed)
    scaler = StandardScaler().fit(np.asarray(matrix[rows], dtype=np.float32))
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=20, batch_size=8192).fit(
        scaler.transform(np.asarray(matrix[rows], dtype=np.float32))
    )
    labels = np.empty(len(matrix), dtype=np.int16)
    for left in range(0, len(matrix), 100_000):
        right = min(left + 100_000, len(matrix))
        labels[left:right] = model.predict(scaler.transform(np.asarray(matrix[left:right], dtype=np.float32)))
    return labels


def cv_continuous(rep: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[dict, np.ndarray, pd.DataFrame]:
    valid = np.isfinite(y)
    x, yy, gg = rep[valid], y[valid].astype(float), groups[valid]
    pred = np.full(len(yy), np.nan)
    folds = []
    for held in np.unique(gg):
        train, test = gg != held, gg == held
        scaler = StandardScaler().fit(x[train])
        model = Ridge(alpha=100.0, solver="lsqr", tol=1e-4).fit(scaler.transform(x[train]), yy[train])
        pred[test] = model.predict(scaler.transform(x[test]))
        folds.append(
            {
                "held_out_pair": held,
                "n": int(test.sum()),
                "r2": float(r2_score(yy[test], pred[test])),
                "spearman": float(stats.spearmanr(yy[test], pred[test]).statistic),
                "alpha": 100.0,
            }
        )
    metric = {
        "n": int(len(yy)),
        "r2": float(r2_score(yy, pred)),
        "spearman": float(stats.spearmanr(yy, pred).statistic),
        "fold_r2_mean": float(np.mean([v["r2"] for v in folds])),
        "fold_spearman_mean": float(np.mean([v["spearman"] for v in folds])),
    }
    full_pred = np.full(len(y), np.nan)
    full_pred[np.flatnonzero(valid)] = pred
    return metric, full_pred, pd.DataFrame(folds)


def cv_binary(rep: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[dict, np.ndarray, pd.DataFrame]:
    valid = np.isfinite(y)
    x, yy, gg = rep[valid], y[valid].astype(int), groups[valid]
    pred = np.full(len(yy), np.nan)
    folds = []
    for held in np.unique(gg):
        train, test = gg != held, gg == held
        scaler = StandardScaler().fit(x[train])
        model = RidgeClassifier(alpha=100.0, class_weight="balanced", solver="lsqr", tol=1e-4).fit(scaler.transform(x[train]), yy[train])
        pred[test] = model.decision_function(scaler.transform(x[test]))
        folds.append(
            {
                "held_out_pair": held,
                "n": int(test.sum()),
                "prevalence": float(yy[test].mean()),
                "auroc": float(roc_auc_score(yy[test], pred[test])),
                "auprc": float(average_precision_score(yy[test], pred[test])),
            }
        )
    metric = {
        "n": int(len(yy)),
        "prevalence": float(yy.mean()),
        "auroc": float(roc_auc_score(yy, pred)),
        "auprc": float(average_precision_score(yy, pred)),
        "fold_auroc_mean": float(np.mean([v["auroc"] for v in folds])),
        "fold_auprc_mean": float(np.mean([v["auprc"] for v in folds])),
    }
    full_pred = np.full(len(y), np.nan)
    full_pred[np.flatnonzero(valid)] = pred
    return metric, full_pred, pd.DataFrame(folds)


def normalized_entropy(values: np.ndarray) -> float:
    count = pd.Series(values).value_counts().to_numpy(float)
    p = count / count.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0


def cluster_metrics(labels: np.ndarray, meta: pd.DataFrame) -> dict:
    sample = meta["slice_id"].astype(str).to_numpy()
    table = pd.crosstab(pd.Series(labels), pd.Series(sample))
    max_fraction = float(table.div(table.sum(1), axis=0).max(axis=1).max())
    min_entropy = min(normalized_entropy(sample[labels == c]) for c in np.unique(labels))
    return {
        "sample_nmi": float(normalized_mutual_info_score(sample, labels)),
        "cell_type_nmi": float(normalized_mutual_info_score(meta["cell_type"].astype(str), labels)),
        "subregion_nmi": float(normalized_mutual_info_score(meta["subregion"].astype(str), labels)),
        "responder_nmi": float(normalized_mutual_info_score(meta["responder_main_q95"].astype(str), labels)),
        "max_cluster_single_sample_fraction": max_fraction,
        "min_cluster_sample_entropy": float(min_entropy),
        "sample_dominated_any": bool(max_fraction > 0.50 or min_entropy < 0.75),
    }


def adjusted_domain_response(meta: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    work = meta.copy()
    work["domain"] = labels
    work["stratum"] = work["paired_batch"].astype(str) + "::" + work["cell_type"].astype(str) + "::" + work["subregion"].astype(str)
    y = work["responder_main_q95"].astype(int)
    margins = work.groupby("stratum", observed=True)["responder_main_q95"].agg(["sum", "size"])
    records = []
    for domain, block in work.groupby("domain", observed=True):
        observed = int(block["responder_main_q95"].sum())
        expected = 0.0
        variance = 0.0
        for key, n_domain in block.groupby("stratum", observed=True).size().items():
            K, N = float(margins.loc[key, "sum"]), float(margins.loc[key, "size"])
            p = K / N
            expected += n_domain * p
            if N > 1:
                variance += n_domain * p * (1 - p) * (N - n_domain) / (N - 1)
        z = (observed - expected) / math.sqrt(max(variance, 1e-12))
        sample_frac = block["slice_id"].astype(str).value_counts(normalize=True)
        pair_rates = block.groupby("paired_batch", observed=True)["responder_main_q95"].mean()
        global_pair_rates = work.groupby("paired_batch", observed=True)["responder_main_q95"].mean()
        records.append(
            {
                "scc_domain": int(domain),
                "n": int(len(block)),
                "responders": observed,
                "raw_responder_rate": float(observed / len(block)),
                "expected_responders_given_pair_celltype_subregion": float(expected),
                "standardized_observed_expected": float(observed / max(expected, 1e-12)),
                "conditional_z": float(z),
                "conditional_pvalue": float(2 * stats.norm.sf(abs(z))),
                "median_local_youngward": float(block["local_score_unscaled"].median()),
                "mean_pair_adjusted_heterogeneity": float(block["pair_adjusted_heterogeneity"].mean()),
                "max_slice_fraction": float(sample_frac.max()),
                "sample_entropy": normalized_entropy(block["slice_id"].astype(str).to_numpy()),
                "sample_dominated": bool(sample_frac.max() > 0.50 or normalized_entropy(block["slice_id"].astype(str).to_numpy()) < 0.75),
                "pair_rate_min": float(pair_rates.min()),
                "pair_rate_max": float(pair_rates.max()),
                "pairs_above_global_rate": int((pair_rates > global_pair_rates.reindex(pair_rates.index)).sum()),
            }
        )
    result = pd.DataFrame(records)
    result["conditional_fdr"] = multipletests(result["conditional_pvalue"], method="fdr_bh")[1]
    return result.sort_values("standardized_observed_expected", ascending=False)


def categorical_enrichment(labels: np.ndarray, values: np.ndarray, sample: np.ndarray, name: str) -> pd.DataFrame:
    rows = []
    cats = np.unique(values)
    for domain in np.unique(labels):
        for cat in cats:
            observed = expected = 0.0
            for s in np.unique(sample):
                si = sample == s
                focal = si & (labels == domain)
                observed += np.sum(focal & (values == cat))
                expected += focal.sum() * np.mean(values[si] == cat)
            rows.append({"domain": int(domain), name: cat, "observed": int(observed), "expected_slice_adjusted": expected, "log2_enrichment": float(np.log2((observed + 0.5) / (expected + 0.5)))})
    return pd.DataFrame(rows)


def expression_interpretation(
    x_oskm: sparse.csr_matrix,
    genes: np.ndarray,
    meta: pd.DataFrame,
    domain: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dense = x_oskm.toarray().astype(np.float32, copy=False)
    stratum = (meta["paired_batch"].astype(str) + "::" + meta["cell_type"].astype(str) + "::" + meta["subregion"].astype(str)).to_numpy()
    residual = dense.copy()
    for key in np.unique(stratum):
        rows = np.flatnonzero(stratum == key)
        residual[rows] -= residual[rows].mean(axis=0, keepdims=True)
    responder = meta["responder_main_q95"].to_numpy(bool)
    gene_rows = []
    for d in np.unique(domain):
        keep = domain == d
        effect = residual[keep].mean(axis=0) - residual[~keep].mean(axis=0)
        pair_effect_matrix = []
        for p in np.unique(meta["paired_batch"].astype(str)):
            pi = meta["paired_batch"].astype(str).to_numpy() == p
            if np.sum(pi & keep) >= 20 and np.sum(pi & ~keep) >= 20:
                pair_effect_matrix.append(residual[pi & keep].mean(axis=0) - residual[pi & ~keep].mean(axis=0))
        pair_effect_matrix = np.asarray(pair_effect_matrix)
        for gi, gene in enumerate(genes):
            pair_effect = pair_effect_matrix[:, gi] if pair_effect_matrix.size else np.array([])
            pvalue = float(stats.wilcoxon(pair_effect).pvalue) if len(pair_effect) >= 3 and np.any(np.asarray(pair_effect) != 0) else np.nan
            gene_rows.append({"scc_domain": int(d), "gene": str(gene), "adjusted_logexpr_effect": float(effect[gi]), "n_pairs": len(pair_effect), "pair_sign_consistency": float(max(np.mean(np.asarray(pair_effect) > 0), np.mean(np.asarray(pair_effect) < 0))) if len(pair_effect) else np.nan, "wilcoxon_pair_pvalue": pvalue})
    domain_genes = pd.DataFrame(gene_rows)
    domain_genes["pair_fdr_within_domain"] = domain_genes.groupby("scc_domain")["wilcoxon_pair_pvalue"].transform(lambda s: multipletests(s.fillna(1), method="fdr_bh")[1])

    responder_rows = []
    responder_pair_matrix = []
    for p in np.unique(meta["paired_batch"].astype(str)):
        pi = meta["paired_batch"].astype(str).to_numpy() == p
        if np.sum(pi & responder) >= 20 and np.sum(pi & ~responder) >= 20:
            responder_pair_matrix.append(residual[pi & responder].mean(axis=0) - residual[pi & ~responder].mean(axis=0))
    responder_pair_matrix = np.asarray(responder_pair_matrix)
    for gi, gene in enumerate(genes):
        pair_effect = responder_pair_matrix[:, gi] if responder_pair_matrix.size else np.array([])
        pvalue = float(stats.wilcoxon(pair_effect).pvalue) if len(pair_effect) >= 3 and np.any(np.asarray(pair_effect) != 0) else np.nan
        responder_rows.append({"gene": str(gene), "adjusted_responder_minus_nonresponder": float(residual[responder, gi].mean() - residual[~responder, gi].mean()), "n_pairs": len(pair_effect), "pair_sign_consistency": float(max(np.mean(np.asarray(pair_effect) > 0), np.mean(np.asarray(pair_effect) < 0))) if len(pair_effect) else np.nan, "wilcoxon_pair_pvalue": pvalue})
    responder_genes = pd.DataFrame(responder_rows)
    responder_genes["pair_fdr"] = multipletests(responder_genes["wilcoxon_pair_pvalue"].fillna(1), method="fdr_bh")[1]

    gene_index = {g: i for i, g in enumerate(genes)}
    sd = residual.std(axis=0)
    standardized = residual / np.maximum(sd, 1e-4)
    program_matrix = {}
    program_rows = []
    program_def = []
    for pname, members in PROGRAMS.items():
        measured = [g for g in members if g in gene_index]
        idx = [gene_index[g] for g in measured]
        score = standardized[:, idx].mean(axis=1)
        program_matrix[pname] = score
        program_def.append({"program": pname, "genes": ";".join(measured), "n_genes": len(measured)})
        for d in np.unique(domain):
            keep = domain == d
            program_rows.append({"program": pname, "scc_domain": int(d), "mean_adjusted_score": float(score[keep].mean()), "responder_minus_nonresponder": float(score[responder].mean() - score[~responder].mean()), "spearman_local_response": float(stats.spearmanr(score, meta["local_score_unscaled"]).statistic)})
        meta[f"program_{pname}"] = score.astype(np.float32)
    return domain_genes, responder_genes, pd.DataFrame(program_rows), pd.DataFrame(program_def)


def response_summaries(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_rows = []
    for label in ["cell_type", "subregion"]:
        for value, block in meta.groupby(label, observed=True):
            pair_rates = block.groupby("paired_batch", observed=True)["responder_main_q95"].mean()
            pair_score = block.groupby("paired_batch", observed=True)["local_score_unscaled"].median()
            label_rows.append({"label": label, "value": value, "n": len(block), "responder_rate": float(block["responder_main_q95"].mean()), "median_local_youngward": float(block["local_score_unscaled"].median()), "pair_rate_min": float(pair_rates.min()), "pair_rate_max": float(pair_rates.max()), "pair_score_min": float(pair_score.min()), "pair_score_max": float(pair_score.max()), "n_pairs": int(len(pair_rates))})
    sensitivity = []
    for col in [c for c in meta if c.startswith("responder_")]:
        sensitivity.append({"definition": col, "n_responders": int(meta[col].sum()), "responder_rate": float(meta[col].mean()), "agreement_with_main": float(np.mean(meta[col].to_numpy() == meta["responder_main_q95"].to_numpy()))})
    return pd.DataFrame(label_rows), pd.DataFrame(sensitivity)


def plot_umaps(existing: Path, meta: pd.DataFrame, clusters: dict[str, np.ndarray], out: Path) -> pd.DataFrame:
    coords = pd.read_csv(existing)
    if not np.array_equal(coords["cell_id"].astype(str).to_numpy(), meta.index.astype(str).to_numpy()):
        raise ValueError("Existing fixed-seed UMAP rows do not align to OSKM cells")
    columns = {
        "z": ("z_intrinsic_umap1", "z_intrinsic_umap2"),
        "s": ("s_intrinsic_umap1", "s_intrinsic_umap2"),
        "z_cc": ("zcc_background_umap1", "zcc_background_umap2"),
        "s_cc": ("scc_salient_umap1", "scc_salient_umap2"),
    }
    merged = coords.copy()
    for col in ["slice_id", "paired_batch", "cell_type", "subregion", "local_score_unscaled", "responder_main_q95"]:
        merged[col] = meta[col].to_numpy()
    for name, values in clusters.items():
        merged[f"{name}_cluster"] = values

    def categorical(fig_name: str, label: str, palette_name: str = "husl") -> None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)
        categories = pd.Categorical(merged[label].astype(str))
        palette = sns.color_palette(palette_name, len(categories.categories))
        for ax, (name, (xcol, ycol)) in zip(axes.ravel(), columns.items()):
            for i, cat in enumerate(categories.categories):
                keep = categories.codes == i
                ax.scatter(merged.loc[keep, xcol], merged.loc[keep, ycol], s=0.18, alpha=0.55, color=palette[i], rasterized=True, label=str(cat))
            ax.set_title(name)
            ax.set_xticks([]); ax.set_yticks([])
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center right", markerscale=8, fontsize=7, frameon=False)
        fig.suptitle(f"OSKM-only fixed-seed UMAPs colored by {label}")
        fig.subplots_adjust(right=0.86)
        fig.savefig(out / fig_name, dpi=220, bbox_inches="tight")
        plt.close(fig)

    categorical("umap_by_sample.png", "slice_id")
    categorical("umap_by_cell_type.png", "cell_type")
    categorical("umap_by_subregion.png", "subregion")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    values = merged["local_score_unscaled"].to_numpy(float)
    lo, hi = np.nanpercentile(values, [1, 99])
    for ax, (name, (xcol, ycol)) in zip(axes.ravel(), columns.items()):
        pts = ax.scatter(merged[xcol], merged[ycol], c=values, s=0.18, alpha=0.6, cmap="coolwarm", vmin=lo, vmax=hi, rasterized=True)
        ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(pts, ax=ax, fraction=0.04, pad=0.01)
    fig.suptitle("OSKM-only fixed-seed UMAPs: local youngward response (descriptive)")
    fig.savefig(out / "umap_by_response_score.png", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    for ax, (name, (xcol, ycol)) in zip(axes.ravel(), columns.items()):
        labels = merged[f"{name}_cluster"].to_numpy()
        ax.scatter(merged[xcol], merged[ycol], c=labels, s=0.18, alpha=0.6, cmap="tab10", rasterized=True)
        ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("OSKM-only deterministic representation clusters (descriptive)")
    fig.savefig(out / "umap_by_representation_cluster.png", dpi=220)
    plt.close(fig)
    return merged


def plot_summaries(rep_metrics: pd.DataFrame, domain_summary: pd.DataFrame, label_summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    view = rep_metrics[(rep_metrics["endpoint"] == "pair_adjusted_heterogeneity") & (rep_metrics["task"] == "continuous")]
    sns.barplot(data=view, x="representation", y="spearman", ax=axes[0], color="#4c78a8")
    sns.barplot(data=view, x="representation", y="r2", ax=axes[1], color="#72b7b2")
    axes[0].axhline(0, color="black", lw=0.7); axes[1].axhline(0, color="black", lw=0.7)
    axes[0].set_title("Leave-pair-out Spearman"); axes[1].set_title("Leave-pair-out R²")
    for ax in axes: ax.tick_params(axis="x", rotation=30)
    fig.savefig(out / "heldout_representation_comparison.png", dpi=220)
    plt.close(fig)

    ordered = domain_summary.sort_values("standardized_observed_expected", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    sns.barplot(data=ordered, x="scc_domain", y="standardized_observed_expected", order=ordered["scc_domain"], ax=axes[0], color="#59a14f")
    sns.barplot(data=ordered, x="scc_domain", y="mean_pair_adjusted_heterogeneity", order=ordered["scc_domain"], ax=axes[1], color="#e15759")
    axes[0].axhline(1, color="black", lw=0.8); axes[1].axhline(0, color="black", lw=0.8)
    axes[0].set_title("Responder enrichment adjusted for pair × cell type × subregion")
    axes[1].set_title("Pair/anatomy-adjusted response by s-CC domain")
    fig.savefig(out / "scc_domain_response_adjusted.png", dpi=220)
    plt.close(fig)

    cell = label_summary[label_summary["label"] == "cell_type"].sort_values("median_local_youngward")
    region = label_summary[label_summary["label"] == "subregion"].sort_values("median_local_youngward")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    sns.barplot(data=cell, y="value", x="median_local_youngward", ax=axes[0], color="#4c78a8")
    sns.barplot(data=region, y="value", x="median_local_youngward", ax=axes[1], color="#f28e2b")
    for ax in axes: ax.axvline(0, color="black", lw=0.7)
    axes[0].set_title("Cell types"); axes[1].set_title("Subregions")
    fig.suptitle("Matched-pair youngward response across all four OSKM pairs")
    fig.savefig(out / "response_by_celltype_subregion.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    tables = args.out / "tables"; figures = args.out / "figures"; audit = args.out / "audit"
    for path in [tables, figures, audit]: path.mkdir(parents=True, exist_ok=True)

    data = ad.read_h5ad(args.data)
    young = ad.read_h5ad(args.young)
    print("stage=loaded_data", flush=True)
    names = np.load(args.model_dir / "obs_names.npy", allow_pickle=True).astype(str)
    if not np.array_equal(names, data.obs_names.astype(str).to_numpy()):
        raise ValueError("Full model rows do not exactly align")
    source_rows = np.load(args.cc_dir / "source_rows.npy").astype(np.int64)
    cc_names = np.load(args.cc_dir / "obs_names.npy", allow_pickle=True).astype(str)
    oskm_rows = np.flatnonzero(data.obs["condition"].astype(str).to_numpy() == "oskm")
    if not np.array_equal(source_rows, oskm_rows) or not np.array_equal(cc_names, data.obs_names[source_rows].astype(str).to_numpy()):
        raise ValueError("Condition-only rows do not exactly align")
    if not np.array_equal(data.var_names.astype(str), young.var_names.astype(str)):
        raise ValueError("Young reference genes do not align")

    x = log_normalize(data.layers["counts"])
    xy = log_normalize(young.layers["counts"])
    scores, directions = response_scores(x, xy, data.obs, young.obs, data.var_names.astype(str).to_numpy())
    print("stage=built_response_directions", flush=True)
    spatial_xy = np.asarray(data.obsm["spatial"], dtype=np.float32)
    samples_all = data.obs["slice_id"].astype(str).to_numpy()
    response_columns = ["score_unscaled", "score_scaled", "score_loo_old"]
    local_matrix = spatial_average_matrix(scores[response_columns].to_numpy(float), spatial_xy, samples_all, args.response_neighbors)
    for column, source in enumerate(response_columns):
        scores[f"local_{source}"] = local_matrix[:, column]
    print("stage=smoothed_response_scores", flush=True)
    full = data.obs[["condition", "slice_id", "paired_batch", "mouse_id", "cell_type", "subregion"]].copy()
    full = pd.concat([full, scores], axis=1)
    full, threshold_table = calibrate_responders(full)
    target = full.iloc[oskm_rows].copy()
    target.index = cc_names
    target["anatomy_heterogeneity"] = target["local_score_unscaled"] - target.groupby(["cell_type", "subregion"], observed=True)["local_score_unscaled"].transform("mean")
    target["pair_adjusted_heterogeneity"] = target["local_score_unscaled"] - target.groupby(["paired_batch", "cell_type", "subregion"], observed=True)["local_score_unscaled"].transform("mean")
    print("stage=calibrated_responders", flush=True)

    reps = {
        "z": np.load(args.model_dir / "background.npy", mmap_mode="r")[oskm_rows].astype(np.float32),
        "s": np.load(args.model_dir / "salient.npy", mmap_mode="r")[oskm_rows].astype(np.float32),
        "z_cc": np.load(args.cc_dir / "spatial.npy", mmap_mode="r").astype(np.float32),
        "s_cc": np.load(args.cc_dir / "salient_spatial.npy", mmap_mode="r").astype(np.float32),
    }
    reps["z_plus_s"] = np.concatenate([reps["z"], reps["s"]], axis=1)
    reps["zcc_plus_scc"] = np.concatenate([reps["z_cc"], reps["s_cc"]], axis=1)
    pair_target = target["paired_batch"].astype(str).to_numpy()

    generic = pd.read_csv(args.domain_assignments)
    if not np.array_equal(generic["cell_id"].astype(str).to_numpy(), cc_names):
        raise ValueError("Generic domain assignments do not align")
    clusters = {"z_cc": generic["z_cc_domain"].to_numpy(np.int16), "s_cc": generic["s_cc_domain"].to_numpy(np.int16)}
    clusters["z"] = fit_domains(reps["z"], pair_target, args.clusters, args.seed)
    clusters["s"] = fit_domains(reps["s"], pair_target, args.clusters, args.seed)

    rep_metric_rows = []
    fold_tables = []
    prediction_table = pd.DataFrame(index=target.index)
    endpoints = {
        "local_youngward": target["local_score_unscaled"].to_numpy(float),
        "anatomy_heterogeneity": target["anatomy_heterogeneity"].to_numpy(float),
        "pair_adjusted_heterogeneity": target["pair_adjusted_heterogeneity"].to_numpy(float),
    }
    for name, rep in reps.items():
        for endpoint, y in endpoints.items():
            metric, pred, folds = cv_continuous(rep, y, pair_target)
            rep_metric_rows.append({"representation": name, "task": "continuous", "endpoint": endpoint, **metric})
            folds.insert(0, "endpoint", endpoint); folds.insert(0, "task", "continuous"); folds.insert(0, "representation", name); fold_tables.append(folds)
            prediction_table[f"pred_{endpoint}_{name}"] = pred
        metric, pred, folds = cv_binary(rep, target["responder_main_q95"].to_numpy(float), pair_target)
        rep_metric_rows.append({"representation": name, "task": "binary", "endpoint": "responder_q95", **metric})
        folds.insert(0, "endpoint", "responder_q95"); folds.insert(0, "task", "binary"); folds.insert(0, "representation", name); fold_tables.append(folds)
        prediction_table[f"pred_responder_{name}"] = pred
        print(f"stage=heldout_representation_done name={name}", flush=True)
    rep_metrics = pd.DataFrame(rep_metric_rows)

    cluster_rows = []
    for name in ["z", "s", "z_cc", "s_cc"]:
        target[f"cluster_{name}"] = clusters[name]
        cluster_rows.append({"representation": name, **cluster_metrics(clusters[name], target)})
    cluster_metric_table = pd.DataFrame(cluster_rows)

    # Direct z/s held-pair-out condition recovery: each test fold contains its matched control and OSKM slices.
    all_pair = data.obs["paired_batch"].astype(str).to_numpy()
    condition_y = (data.obs["condition"].astype(str).to_numpy() == "oskm").astype(int)
    condition_rows = []
    for name, filename in [("z", "background.npy"), ("s", "salient.npy")]:
        rep_all = np.load(args.model_dir / filename, mmap_mode="r")
        metric, _, folds = cv_binary(rep_all, condition_y.astype(float), all_pair)
        condition_rows.append({"representation": name, **metric})
        folds.to_csv(tables / f"condition_recovery_{name}_folds.csv", index=False)
    condition_recovery = pd.DataFrame(condition_rows)

    domain_summary = adjusted_domain_response(target, clusters["s_cc"])
    ct_enrichment = categorical_enrichment(clusters["s_cc"], target["cell_type"].astype(str).to_numpy(), target["slice_id"].astype(str).to_numpy(), "cell_type")
    sr_enrichment = categorical_enrichment(clusters["s_cc"], target["subregion"].astype(str).to_numpy(), target["slice_id"].astype(str).to_numpy(), "subregion")
    label_summary, sensitivity = response_summaries(target)

    domain_genes, responder_genes, program_table, program_def = expression_interpretation(
        x[oskm_rows], data.var_names.astype(str).to_numpy(), target, clusters["s_cc"]
    )
    print("stage=expression_interpretation_done", flush=True)

    # Stability of the salient domain solution on a fixed balanced 50k evaluation subset.
    eval_rows = balanced_rows(pair_target, 50_000, args.seed)
    stability = []
    for seed in [1, 2, 3, 4]:
        alt = fit_domains(reps["s_cc"], pair_target, args.clusters, seed)
        stability.append({"reference_seed": args.seed, "alternative_seed": seed, "adjusted_rand_index_50k": float(adjusted_rand_score(clusters["s_cc"][eval_rows], alt[eval_rows]))})

    umap_merged = plot_umaps(args.existing_umap, target, clusters, figures)
    plot_summaries(rep_metrics, domain_summary, label_summary, figures)
    print("stage=figures_done", flush=True)

    target_out = target.reset_index(names="cell_id")
    target_out.to_csv(tables / "oskm_cell_scores_and_domains.csv.gz", index=False, compression="gzip")
    prediction_table.reset_index(names="cell_id").to_csv(tables / "heldout_predictions.csv.gz", index=False, compression="gzip")
    umap_merged.to_csv(tables / "condition_only_fixed_seed_umap_coordinates.csv.gz", index=False, compression="gzip")
    threshold_table.to_csv(tables / "matched_control_null_thresholds.csv", index=False)
    directions.to_csv(tables / "youngward_direction_top_genes.csv", index=False)
    rep_metrics.to_csv(tables / "representation_heldout_metrics.csv", index=False)
    pd.concat(fold_tables, ignore_index=True).to_csv(tables / "representation_heldout_metrics_by_pair.csv", index=False)
    cluster_metric_table.to_csv(tables / "representation_cluster_associations.csv", index=False)
    condition_recovery.to_csv(tables / "direct_latent_condition_recovery.csv", index=False)
    domain_summary.to_csv(tables / "scc_domain_response_adjusted.csv", index=False)
    ct_enrichment.to_csv(tables / "scc_domain_celltype_slice_adjusted_enrichment.csv", index=False)
    sr_enrichment.to_csv(tables / "scc_domain_subregion_slice_adjusted_enrichment.csv", index=False)
    label_summary.to_csv(tables / "response_by_celltype_and_subregion.csv", index=False)
    sensitivity.to_csv(tables / "responder_definition_sensitivity.csv", index=False)
    domain_genes.to_csv(tables / "scc_domain_genes_pair_anatomy_adjusted.csv", index=False)
    responder_genes.to_csv(tables / "responder_genes_pair_anatomy_adjusted.csv", index=False)
    program_table.to_csv(tables / "scc_domain_and_responder_programs.csv", index=False)
    program_def.to_csv(tables / "program_definitions.csv", index=False)
    pd.DataFrame(stability).to_csv(tables / "scc_domain_seed_stability.csv", index=False)

    audit_payload = {
        "data_shape": [int(data.n_obs), int(data.n_vars)],
        "young_shape": [int(young.n_obs), int(young.n_vars)],
        "model_obs_exact": True,
        "cc_obs_exact_condition_order": True,
        "var_names_exact_young": True,
        "counts_integer_valued": bool(np.allclose(data.layers["counts"].data if sparse.issparse(data.layers["counts"]) else data.layers["counts"], np.rint(data.layers["counts"].data if sparse.issparse(data.layers["counts"]) else data.layers["counts"]))),
        "spatial_finite": bool(np.isfinite(spatial_xy).all()),
        "graph_id_equals_slice_id": bool(np.all(data.obs["graph_id"].astype(str).to_numpy() == samples_all)),
        "condition_counts": data.obs["condition"].astype(str).value_counts().to_dict(),
        "slice_counts": data.obs["slice_id"].astype(str).value_counts().sort_index().to_dict(),
        "paired_batch_condition_table": pd.crosstab(data.obs["paired_batch"].astype(str), data.obs["condition"].astype(str)).to_dict(),
        "response_definition": "OSKM local (within-slice k=15) projection on matched-pair old-control to pooled-young cell_type x subregion direction; responder exceeds matched-control 95th percentile in pair x cell_type x subregion",
        "young_used_for_training": False,
        "cv_group": "paired_batch (equivalent to one OSKM mouse/slice per held-out fold)",
    }
    (audit / "alignment_and_design_audit.json").write_text(json.dumps(audit_payload, indent=2) + "\n")
    summary = {
        "n_oskm": int(len(target)),
        "n_scored": int(target["local_score_unscaled"].notna().sum()),
        "main_responder_rate": float(target["responder_main_q95"].mean()),
        "top_response_representation": rep_metrics[(rep_metrics.task == "continuous") & (rep_metrics.endpoint == "pair_adjusted_heterogeneity")].sort_values("spearman", ascending=False).iloc[0][["representation", "r2", "spearman"]].to_dict(),
        "richest_scc_domain": domain_summary.iloc[0].to_dict(),
        "poorest_scc_domain": domain_summary.iloc[-1].to_dict(),
        "scc_seed_stability_mean_ari": float(pd.DataFrame(stability)["adjusted_rand_index_50k"].mean()),
        "limitations": [
            "slice identity is perfectly confounded with condition; decoder conditioning does not guarantee latent batch removal",
            "only four matched pairs are available, so pair-level uncertainty is low-resolution",
            "response and gene associations use the same targeted 300-gene panel and are not independent validation",
            "cell type composition was a Haruka reconstruction target; domain composition is interpretive, not independent validation",
            "UMAP and static spatial patterns are descriptive, not lineage or causality",
        ],
    }
    (args.out / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
