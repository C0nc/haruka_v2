#!/usr/bin/env python3
"""Per-section treated-tumor DPT using condition-only Haruka s-CC."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def orient_root(diffmap, response):
    dc1 = diffmap[:, 1] if diffmap.shape[1] > 1 else diffmap[:, 0]
    q = max(10, int(np.ceil(0.05 * len(dc1))))
    lo = np.argpartition(dc1, q - 1)[:q]
    hi = np.argpartition(dc1, len(dc1) - q)[-q:]
    endpoint = lo if np.nanmedian(response[lo]) <= np.nanmedian(response[hi]) else hi
    return int(endpoint[np.nanargmin(response[endpoint])])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cc-dir", type=Path, required=True)
    p.add_argument("--assignments", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--neighbors", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    figdir, tabdir = args.out_dir / "figures", args.out_dir / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)

    source_rows = np.load(args.cc_dir / "source_rows.npy")
    scc = np.load(args.cc_dir / "salient_spatial.npy", mmap_mode="r")
    assignment = pd.read_csv(args.assignments)
    rows = assignment["source_row"].to_numpy()
    pos = np.searchsorted(source_rows, rows)
    if np.any(pos >= len(source_rows)) or not np.array_equal(source_rows[pos], rows):
        raise ValueError("Tumor assignments do not align to condition-only CellCharter rows")

    results, trajectories = [], {}
    section_array = assignment["slice_id"].astype(str).to_numpy()
    sections = sorted(np.unique(section_array))
    for section in sections:
        take = np.flatnonzero(section_array == section)
        frame = assignment.iloc[take].copy().reset_index(drop=True)
        x = StandardScaler().fit_transform(np.asarray(scc[pos[take]], dtype=np.float32))
        npc = min(30, x.shape[1], len(x) - 1)
        x = PCA(n_components=npc, random_state=args.seed).fit_transform(x).astype(np.float32)
        data = ad.AnnData(x, obs=frame.set_index("cell_id"))
        sc.pp.neighbors(data, n_neighbors=min(args.neighbors, len(data) - 1), use_rep="X", random_state=args.seed)
        sc.tl.diffmap(data, n_comps=min(15, npc))
        root = orient_root(data.obsm["X_diffmap"], frame["response_score"].to_numpy(float))
        data.uns["iroot"] = root
        sc.tl.dpt(data)
        dpt = data.obs["dpt_pseudotime"].to_numpy(float)
        terminal = dpt >= np.nanquantile(dpt, 0.9)
        frame["dpt_pseudotime"] = dpt
        frame["dpt_root"] = np.arange(len(frame)) == root
        frame["dpt_terminal_10pct"] = terminal
        results.append(frame)
        trajectories[section] = (data.obsm["X_diffmap"][:, 1:3], frame, root, terminal)
        print(section, len(frame), flush=True)

    output = pd.concat(results, ignore_index=True)
    output.to_csv(tabdir / "treated_tumor_per_section_scc_dpt.csv.gz", index=False, compression="gzip")
    fig, axes = plt.subplots(2, len(sections), figsize=(5 * len(sections), 9), constrained_layout=True)
    if len(sections) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    states = sorted(output["celltype"].unique())
    state_colors = dict(zip(states, plt.cm.Set2(np.linspace(0, 1, len(states)))))
    color_plot = None
    for col, section in enumerate(sections):
        xy, frame, root, terminal = trajectories[section]
        color_plot = axes[0, col].scatter(xy[:, 0], xy[:, 1], c=frame["dpt_pseudotime"], cmap="viridis", s=5, linewidths=0, rasterized=True)
        axes[0, col].scatter(xy[root, 0], xy[root, 1], marker="*", s=130, c="#ef5350", edgecolor="black", linewidth=0.7)
        axes[0, col].scatter(xy[terminal, 0], xy[terminal, 1], s=14, facecolors="none", edgecolors="#ff7f0e", linewidths=0.5)
        axes[0, col].set_title(f"{section}: tumor s-CC DPT")
        for state, color in state_colors.items():
            mask = frame["celltype"].astype(str).to_numpy() == state
            axes[1, col].scatter(xy[mask, 0], xy[mask, 1], s=5, c=[color], linewidths=0, rasterized=True)
        axes[1, col].set_title(f"{section}: tumor state")
        for row in range(2):
            axes[row, col].set_xlabel("Diffusion component 1")
            axes[row, col].set_ylabel("Diffusion component 2")
    fig.colorbar(color_plot, ax=axes[0, :].tolist(), fraction=0.015, pad=0.01, label="Within-section DPT")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=s) for s, c in state_colors.items()]
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(1.04, 0.28), frameon=False)
    fig.suptitle("KRASi-treated tumor-only Haruka s-CC diffusion trajectories", fontsize=16)
    fig.savefig(figdir / "tumor_scc_per_sample_diffmap_pseudotime_state.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    summary = output.groupby("slice_id").agg(n_tumor=("cell_id", "size"), median_dpt=("dpt_pseudotime", "median"), terminal_cells=("dpt_terminal_10pct", "sum")).reset_index()
    summary.to_csv(tabdir / "per_section_dpt_summary.csv", index=False)
    metadata = {
        "population": "KRASi-treated tumor cells only",
        "representation": "condition-only Haruka s-CC, slice-bounded 3-hop CellCharter",
        "trajectory": "DPT fitted independently within each section",
        "root_orientation": "diffusion endpoint with lower decoded response score",
        "caveat": "descriptive state ordering, not longitudinal lineage or measured tumor shrinkage",
        "n_cells": int(len(output)), "sections": sections,
    }
    (args.out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
