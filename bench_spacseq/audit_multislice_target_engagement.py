"""Check whether each retained perturbation suppresses its named target gene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse
from scipy.stats import combine_pvalues, mannwhitneyu


def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 1.0
    for rank in range(len(p) - 1, -1, -1):
        index = order[rank]
        running = min(running, p[index] * len(p) / (rank + 1))
        adjusted[index] = running
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    adata = ad.read_h5ad(args.data)
    anchor = adata.obs.is_anchor.to_numpy(bool)
    perturbation = adata.obs.perturbation.astype(str).to_numpy()
    slices = adata.obs.slice_id.astype(str).to_numpy()
    counts = adata.layers["counts"]
    total = np.asarray(counts.sum(axis=1)).ravel().clip(min=1)
    targets = sorted(set(perturbation[anchor]) - {"NT"})
    records = []
    for target in targets:
        if target not in adata.var_names:
            records.append({"target": target, "available": False})
            continue
        column = counts[:, adata.var_names.get_loc(target)]
        value = np.asarray(column.toarray()).ravel() if sparse.issparse(column) else np.asarray(column).ravel()
        value = np.log1p(value * 1e4 / total)
        per_slice, pvalues, effects = {}, [], []
        for slice_id in sorted(set(slices)):
            nt = anchor & (slices == slice_id) & (perturbation == "NT")
            treated = anchor & (slices == slice_id) & (perturbation == target)
            effect = float(value[treated].mean() - value[nt].mean())
            pvalue = float(mannwhitneyu(value[treated], value[nt], alternative="less").pvalue)
            per_slice[slice_id] = {
                "n_target": int(treated.sum()), "n_NT": int(nt.sum()),
                "target_mean_log1p_cptt": float(value[treated].mean()),
                "NT_mean_log1p_cptt": float(value[nt].mean()),
                "effect_target_minus_NT": effect,
                "one_sided_mannwhitney_p": pvalue,
            }
            effects.append(effect); pvalues.append(pvalue)
        combined = float(combine_pvalues(pvalues, method="fisher").pvalue)
        records.append({
            "target": target, "available": True,
            "mean_effect_target_minus_NT": float(np.mean(effects)),
            "negative_slices": int(np.sum(np.asarray(effects) < 0)),
            "fisher_combined_one_sided_p": combined,
            "slices": per_slice,
        })
    available = [record for record in records if record.get("available")]
    adjusted = bh_adjust([record["fisher_combined_one_sided_p"] for record in available])
    for record, qvalue in zip(available, adjusted):
        record["BH_q"] = float(qvalue)
    result = {
        "data": args.data,
        "n_targets": len(targets),
        "n_negative_all_three_slices": int(sum(r.get("negative_slices") == 3 for r in available)),
        "n_BH_q_below_0.05": int(sum(r.get("BH_q", 1) < 0.05 for r in available)),
        "targets": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
