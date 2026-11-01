"""Prepare balanced real-data studies for one-stage unsupervised Haruka.

The only biological label used by Haruka is ``condition`` (control/target).
Stage, cell identity, tissue domains, and plaque annotations are retained only
for post-fit evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors


SOURCES = {
    "dkd": "/home/yancui/trajectory/dkd_xenium/xenium_gfr.h5ad",
    "colitis": "/home/yancui/trajectory/colitis_merfish/adata.h5ad",
}


def _balanced_rows(obs: pd.DataFrame, key: str, cap: int, seed: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    selected, audit = [], {}
    values = obs[key].astype(str).to_numpy()
    for value in sorted(np.unique(values)):
        idx = np.flatnonzero(values == value)
        take = np.sort(rng.choice(idx, min(cap, len(idx)), replace=False))
        selected.append(take)
        audit[value] = {"available": int(len(idx)), "selected": int(len(take))}
    return np.sort(np.concatenate(selected)), audit


def _local_domains(data: ad.AnnData, n_domains: int = 8, k: int = 20, seed: int = 0) -> None:
    """Evaluation-only tissue-domain labels from local coarse-type composition."""
    cell_type = data.obs["cell_type"].astype(str).to_numpy()
    types = sorted(np.unique(cell_type))
    lookup = {value: i for i, value in enumerate(types)}
    onehot = np.zeros((data.n_obs, len(types)), dtype=np.float32)
    onehot[np.arange(data.n_obs), [lookup[x] for x in cell_type]] = 1.0
    comp = np.zeros_like(onehot)
    graph = data.obs["graph_id"].astype(str).to_numpy()
    xy = np.asarray(data.obsm["spatial"], dtype=np.float32)
    for value in np.unique(graph):
        idx = np.flatnonzero(graph == value)
        kk = min(k, len(idx))
        nbr = NearestNeighbors(n_neighbors=kk).fit(xy[idx]).kneighbors(return_distance=False)
        comp[idx] = onehot[idx[nbr]].mean(axis=1)
    n_domains = min(n_domains, max(2, data.n_obs // 100))
    labels = MiniBatchKMeans(
        n_clusters=n_domains, batch_size=4096, n_init=10, random_state=seed
    ).fit_predict(comp)
    data.obs["tissue_domain"] = pd.Categorical([f"domain_{x}" for x in labels])


def _finalize(data: ad.AnnData, output: Path, report: dict, seed: int) -> None:
    data.obs["slice_id"] = data.obs["graph_id"].astype(str).astype("category")
    for key in ("condition", "stage", "graph_id", "specimen_id", "cell_type", "cell_subtype"):
        data.obs[key] = data.obs[key].astype(str).astype("category")
    data.obs["perturbation"] = data.obs["stage"].astype(str).astype("category")
    data.layers["counts"] = data.X.copy()
    _local_domains(data, seed=seed)
    data.obs_names = pd.Index(
        [f"{g}:{i}" for g, i in zip(data.obs["graph_id"].astype(str), data.obs_names.astype(str))]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(output, compression="gzip")
    report.update({
        "output": str(output),
        "shape": [int(data.n_obs), int(data.n_vars)],
        "condition_counts": data.obs.condition.value_counts().astype(int).to_dict(),
        "stage_counts": data.obs.stage.value_counts().astype(int).to_dict(),
        "graphs_by_stage": pd.crosstab(data.obs.graph_id, data.obs.stage).astype(int).to_dict(),
        "training_labels": ["condition"],
        "evaluation_only_labels": [
            "stage", "cell_type", "cell_subtype", "tissue_domain",
            "plaque_tau", "plaque_abeta", "plaque_any",
        ],
        "batch_key": None,
        "batch_reason": "specimen/section is condition-confounded and is only a graph boundary",
    })
    output.with_suffix(".audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def prepare_dkd(output: Path, cap: int, seed: int) -> None:
    source = ad.read_h5ad(SOURCES["dkd"])
    rows, audit = _balanced_rows(source.obs, "sample", cap, seed)
    data = source[rows].copy()
    data.obs["condition"] = np.where(data.obs.GFR.astype(str) == ">60", "control", "disease")
    data.obs["stage"] = data.obs.GFR.astype(str)
    data.obs["graph_id"] = data.obs["sample"].astype(str)
    data.obs["specimen_id"] = data.obs["sample"].astype(str)
    data.obs["cell_subtype"] = data.obs["cell_type"].astype(str)
    _finalize(data, output, {
        "study": "dkd", "source": SOURCES["dkd"], "sampling": audit,
        "control": "GFR >60", "target": "GFR 30-60 or <30",
        "withheld_discovery": "30-60 versus <30 severity within the target arm",
    }, seed)


def prepare_colitis(output: Path, cap: int, seed: int) -> None:
    source = ad.read_h5ad(SOURCES["colitis"])
    rows, audit = _balanced_rows(source.obs, "Slice_ID", cap, seed)
    data = source[rows].copy()
    data.obsm["spatial"] = data.obs[["x", "y"]].to_numpy(dtype=np.float32)
    data.obs["condition"] = np.where(
        data.obs.Sample_type.astype(str) == "Healthy", "control", "disease"
    )
    data.obs["stage"] = data.obs.Sample_type.astype(str)
    data.obs["graph_id"] = data.obs.Slice_ID.astype(str)
    data.obs["specimen_id"] = data.obs.Mouse_ID.astype(str)
    data.obs["cell_type"] = data.obs.Tier1.astype(str)
    data.obs["cell_subtype"] = data.obs.Tier2.astype(str)
    _finalize(data, output, {
        "study": "colitis", "source": SOURCES["colitis"], "sampling": audit,
        "control": "Healthy/day 0", "target": "pooled DSS3, DSS9, DSS21",
        "withheld_discovery": "DSS day and injury-to-repair state within the target arm",
    }, seed)


def prepare_ad(output: Path, seed: int) -> None:
    paths = {
        "control_r1": "/home/yancui/Jill/spatial_13months-control-replicate_1.h5ad",
        "disease_r1": "/home/yancui/Jill/13months-disease-replicate_1_random.h5ad",
        "disease_r2": "/home/yancui/Jill/13months-disease-replicate_2_random.h5ad",
    }
    items = []
    common = None
    for graph, path in paths.items():
        a = ad.read_h5ad(path)
        common = set(a.var_names) if common is None else common & set(a.var_names)
        items.append((graph, a))
    genes = sorted(common)
    ready = []
    for graph, a in items:
        a = a[:, genes].copy()
        a.obs["graph_id"] = graph
        a.obs["specimen_id"] = graph
        a.obs["condition"] = "control" if graph.startswith("control") else "disease"
        a.obs["stage"] = "control" if graph.startswith("control") else "AD_13m"
        a.obs["cell_type"] = a.obs.ct_top.astype(str)
        a.obs["cell_subtype"] = a.obs.ct_sub.astype(str)
        a.obs["plaque_tau"] = (a.obs["p-tau"].to_numpy() > 0).astype(int) if "p-tau" in a.obs else 0
        a.obs["plaque_abeta"] = (a.obs["Aβ"].to_numpy() > 0).astype(int) if "Aβ" in a.obs else 0
        a.obs["plaque_any"] = ((a.obs.plaque_tau.to_numpy() + a.obs.plaque_abeta.to_numpy()) > 0).astype(int)
        a.obsm["spatial"] = a.obs[["x", "y"]].to_numpy(dtype=np.float32)
        ready.append(a)
    data = ad.concat(ready, join="inner", merge="same", index_unique=None)
    _finalize(data, output, {
        "study": "ad", "sources": paths,
        "control": "13-month control replicate 1", "target": "13-month AD replicates 1 and 2",
        "holdout_graph": None,
        "withheld_discovery": "p-tau and amyloid-beta adjacency",
    }, seed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("study", choices=["dkd", "colitis", "ad"])
    p.add_argument("--output", default=None)
    p.add_argument("--per-graph", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260809)
    args = p.parse_args()
    output = Path(args.output or f"bench_real_discovery/data/{args.study}.h5ad")
    if args.study == "dkd":
        prepare_dkd(output, args.per_graph or 3500, args.seed)
    elif args.study == "colitis":
        prepare_colitis(output, args.per_graph or 1500, args.seed)
    else:
        prepare_ad(output, args.seed)


if __name__ == "__main__":
    main()
