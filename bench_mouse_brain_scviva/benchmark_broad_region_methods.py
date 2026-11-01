"""Rescore all completed mouse-brain methods on four broad anatomical regions."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


ROOT = Path("/home/yancui/haruka_v2/bench_mouse_brain_scviva")
OUT = ROOT / "results_cross_section_bg_diagnosis" / "broad_region_methods"
SECTIONS = ("A8", "A9", "B17", "B3", "B5")
SEED = 0
N_INIT = 10


def result_root(section: str) -> Path:
    return (
        ROOT / "results_direct_kmeans"
        if section == "A8"
        else ROOT / "results_cross_section"
    )


def method_dirs(section: str) -> dict[str, Path]:
    root = result_root(section)
    tag = f"mouse_brain_{section}_paired"
    return {
        "Haruka z/s + CellCharter": (
            root / "haruka_z_s_cellcharter" / tag / "haruka_z_s_cellcharter_seed0"
        ),
        "Mini-batch CINEMA-OT + CellCharter": (
            root
            / "cinema_minibatch"
            / tag
            / "cinemaot_minibatch_b2048_e3_seed0"
        ),
        "ContrastiveVI z/s + CellCharter": (
            root / "baselines" / tag / "contrastivevi_2x2_seed0"
        ),
        "scVI + CellCharter": (
            root / "baselines" / tag / "cellcharter_2x2_seed0"
        ),
        "Expression + neighborhood aggregation": (
            root / "baselines" / tag / "pca_2x2_seed0"
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for section in SECTIONS:
        data = ad.read_h5ad(
            ROOT
            / "data_composition_pairs_all"
            / f"mouse_brain_{section}_paired.h5ad",
            backed="r",
        )
        for method, directory in method_dirs(section).items():
            required = (
                directory / "spatial.npy",
                directory / "obs_names.npy",
                directory / "meta.json",
            )
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError(f"{method}, {section}: missing {missing}")
            names = np.load(directory / "obs_names.npy", allow_pickle=True).astype(str)
            positions = data.obs_names.astype(str).get_indexer(names)
            if (positions < 0).any():
                raise ValueError(f"{method}, {section}: embedding rows do not align")
            obs = data.obs.iloc[positions]
            truth = obs["region"].astype(str).to_numpy()
            control = obs["condition"].astype(str).to_numpy() == "control"
            representation = np.load(directory / "spatial.npy", mmap_mode="r")
            prediction = KMeans(
                n_clusters=len(np.unique(truth)),
                random_state=SEED,
                n_init=N_INIT,
            ).fit_predict(np.asarray(representation, dtype=np.float32))
            with open(directory / "meta.json") as handle:
                meta = json.load(handle)
            rows.append(
                {
                    "section": section,
                    "method": method,
                    "background_region_ari": float(
                        adjusted_rand_score(truth, prediction)
                    ),
                    "control_region_ari": float(
                        adjusted_rand_score(truth[control], prediction[control])
                    ),
                    "condition_region_ari": float(
                        adjusted_rand_score(truth[~control], prediction[~control])
                    ),
                    "n_cells": int(len(truth)),
                    "n_features": int(representation.shape[1]),
                    "result_dir": str(directory),
                    "method_meta": meta.get("method", ""),
                }
            )
        data.file.close()

    per_section = pd.DataFrame(rows)
    per_section.to_csv(OUT / "broad_region_per_section.csv", index=False)
    summary = (
        per_section.groupby("method")
        .agg(
            n_sections=("section", "size"),
            mean_ari=("background_region_ari", "mean"),
            sd_ari=("background_region_ari", "std"),
            control_mean_ari=("control_region_ari", "mean"),
            condition_mean_ari=("condition_region_ari", "mean"),
        )
        .reset_index()
        .sort_values("mean_ari", ascending=False)
    )
    summary["rank"] = np.arange(1, len(summary) + 1)
    summary.to_csv(OUT / "broad_region_summary.csv", index=False)

    order = summary["method"].tolist()
    values = summary.set_index("method").loc[order, "mean_ari"]
    errors = summary.set_index("method").loc[order, "sd_ari"]
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    colors = ["#4c78a8" if name.startswith("Haruka") else "#9ecae1" for name in order]
    ax.bar(np.arange(len(order)), values, yerr=errors, capsize=4, color=colors)
    ax.set_xticks(np.arange(len(order)), order, rotation=23, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Broad background region ARI")
    ax.set_title("Four-region mouse-brain benchmark across five paired sections")
    for index, value in enumerate(values):
        ax.text(index, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)
    fig.savefig(OUT / "broad_region_all_methods.png", dpi=220)
    fig.savefig(OUT / "broad_region_all_methods.pdf")
    plt.close(fig)

    print("\nPer section\n", per_section.to_string(index=False), flush=True)
    print("\nSummary\n", summary.to_string(index=False), flush=True)
    print(f"\nSaved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
