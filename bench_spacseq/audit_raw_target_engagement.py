"""Audit sgRNA target engagement from the full raw SPARC RNA matrices.

The 3,000-HVG training object omits most named target genes, so engagement must
be evaluated against the original full feature matrices rather than interpreted
as missing biology.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import combine_pvalues, mannwhitneyu


SOURCES = {
    "M001": ("Lung_Metastasis_M001", "lm992"),
    "M002": ("Lung_Metastasis_M002", "lm994"),
    "M003": ("Lung_Metastasis_M003", "lm993"),
}


def target_gene(guide: str) -> str:
    match = re.match(r"^sg(.+)_\d+$", str(guide))
    return match.group(1) if match else str(guide).removeprefix("sg")


def bh(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    result = np.empty_like(values)
    running = 1.0
    for rank in range(len(values) - 1, -1, -1):
        idx = order[rank]
        running = min(running, values[idx] * len(values) / (rank + 1))
        result[idx] = running
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--raw-root", default="bench_spacseq/raw")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    prepared = ad.read_h5ad(args.data, backed="r")
    obs = prepared.obs.copy()
    prepared.file.close()
    anchor = obs.is_anchor.to_numpy(bool)
    guide = obs.perturbation.astype(str).to_numpy()
    slices = obs.slice_id.astype(str).to_numpy()
    results = {label: {"guide": label, "target_gene": target_gene(label), "slices": {}}
               for label in sorted(set(guide[anchor]) - {"NT"})}

    for section, (directory, _) in SOURCES.items():
        rows = np.flatnonzero(anchor & (slices == section))
        barcodes = pd.Index([str(obs.index[i]).split(":", 1)[1] for i in rows])
        source = ad.read_h5ad(Path(args.raw_root) / directory / "filtered_gene_bc_matrix.h5", backed="r")
        pos = pd.Index(source.obs_names.astype(str)).get_indexer(barcodes)
        if (pos < 0).any():
            raise ValueError(f"{section}: {int((pos < 0).sum())} anchor barcodes absent")
        selected = source[pos].to_memory()
        source.file.close()
        total = np.asarray(selected.X.sum(1)).ravel().clip(min=1)
        local_guide = guide[rows]
        for label, result in results.items():
            gene = result["target_gene"]
            candidates = [g for g in selected.var_names if str(g).lower() == gene.lower()]
            if not candidates:
                result["slices"][section] = {"available": False}
                continue
            column = selected[:, candidates[0]].X
            value = column.toarray().ravel() if sp.issparse(column) else np.asarray(column).ravel()
            value = np.log1p(value * 1e4 / total)
            treated = local_guide == label
            control = local_guide == "NT"
            effect = float(value[treated].mean() - value[control].mean())
            pvalue = float(mannwhitneyu(value[treated], value[control], alternative="less").pvalue)
            result["slices"][section] = {
                "available": True,
                "n_target": int(treated.sum()),
                "n_nt": int(control.sum()),
                "effect_target_minus_nt": effect,
                "one_sided_p": pvalue,
            }

    records = []
    for result in results.values():
        usable = [x for x in result["slices"].values() if x.get("available")]
        result["mean_effect"] = float(np.mean([x["effect_target_minus_nt"] for x in usable]))
        result["negative_slices"] = int(np.sum([x["effect_target_minus_nt"] < 0 for x in usable]))
        result["fisher_p"] = float(combine_pvalues([x["one_sided_p"] for x in usable]).pvalue)
        records.append(result)
    adjusted = bh([x["fisher_p"] for x in records])
    for result, qvalue in zip(records, adjusted):
        result["BH_q"] = float(qvalue)
    frame = pd.DataFrame([{k: v for k, v in x.items() if k != "slices"} for x in records])
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"targets": records}, indent=2) + "\n")
    frame.sort_values("mean_effect").to_csv(output.with_suffix(".csv"), index=False)
    print(frame.sort_values("mean_effect").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
