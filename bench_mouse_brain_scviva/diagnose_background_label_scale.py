"""Test anatomical label scale and trace CellCharter background leakage."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


ROOT = Path("/home/yancui/haruka_v2/bench_mouse_brain_scviva")
OUT = ROOT / "results_cross_section_bg_diagnosis" / "label_scale"
SECTIONS = ("A8", "A9", "B17", "B3", "B5")
SEED = 0
N_INIT = 10

sys.path.insert(0, str(ROOT.parent / "bench_merfish"))
from run_haruka_cvi_scviva import neighbor_index  # noqa: E402
from run_haruka_shared_decoder import annotated_composition  # noqa: E402


def latent_dir(section: str) -> Path:
    root = (
        ROOT / "results_direct_kmeans"
        if section == "A8"
        else ROOT / "results_cross_section"
    )
    return (
        root
        / "haruka_z_s_cellcharter"
        / f"mouse_brain_{section}_paired"
        / "haruka_z_s_cellcharter_seed0"
    )


def direct_dir(latent: Path) -> Path:
    with open(latent / "meta.json") as handle:
        return Path(json.load(handle)["source_result"])


def fit(x: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return KMeans(
        n_clusters=len(np.unique(truth)),
        random_state=SEED,
        n_init=N_INIT,
    ).fit_predict(np.asarray(x, dtype=np.float32))


def align(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred_levels = np.unique(prediction)
    truth_levels = np.unique(truth)
    counts = np.zeros((len(pred_levels), len(truth_levels)), dtype=np.int64)
    for i, pred in enumerate(pred_levels):
        for j, label in enumerate(truth_levels):
            counts[i, j] = np.sum((prediction == pred) & (truth == label))
    rows, cols = linear_sum_assignment(-counts)
    mapping = {pred_levels[i]: truth_levels[j] for i, j in zip(rows, cols)}
    return np.asarray([mapping.get(value, "unmapped") for value in prediction])


def same_label_neighbor_purity(labels: np.ndarray, neighbors: np.ndarray) -> float:
    return float(np.mean(labels[neighbors] == labels[:, None]))


def scatter(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    palette: dict[str, object],
) -> None:
    labels = np.asarray(labels).astype(str)
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(
            x[mask],
            y[mask],
            s=0.45,
            color=palette.get(label, "#bdbdbd"),
            linewidths=0,
            rasterized=True,
        )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    label_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []
    plot_items: dict[str, dict[str, object]] = {}
    region_levels: set[str] = set()

    for section in SECTIONS:
        data_path = (
            ROOT
            / "data_composition_pairs_all"
            / f"mouse_brain_{section}_paired.h5ad"
        )
        data = ad.read_h5ad(data_path, backed="r")
        emb_dir = latent_dir(section)
        names = np.load(emb_dir / "obs_names.npy", allow_pickle=True).astype(str)
        positions = data.obs_names.astype(str).get_indexer(names)
        if (positions < 0).any():
            raise ValueError(f"{section}: embeddings are not aligned")
        obs = data.obs.iloc[positions].copy()
        aligned = data[positions]
        control = obs["condition"].astype(str).to_numpy() == "control"
        target = ~control

        direct = direct_dir(emb_dir)
        raw_z = np.load(direct / "background.npy", mmap_mode="r")
        decoder = np.load(direct / "spatial.npy", mmap_mode="r")
        cellcharter = np.load(emb_dir / "spatial.npy", mmap_mode="r")
        representations = {
            "Raw z": raw_z,
            "z + CellCharter": cellcharter,
            "Background decoder": decoder,
        }

        labels = obs["observed_celltype"].astype(str).to_numpy()
        categories = np.unique(labels)
        neighbors = neighbor_index(aligned, 20, "slice_id")
        composition = annotated_composition(labels, neighbors, categories)
        paired_background = np.vstack([composition[control], composition[control]])
        representations["Paired composition oracle"] = paired_background

        broad_prediction = {}
        for label_name, key in (
            ("Fine subregion (7)", "gt_background_subregion"),
            ("Broad region (4)", "region"),
        ):
            truth = obs[key].astype(str).to_numpy()
            purity = same_label_neighbor_purity(truth, neighbors)
            for method, representation in representations.items():
                prediction = fit(representation, truth)
                label_rows.append(
                    {
                        "section": section,
                        "label_scale": label_name,
                        "method": method,
                        "n_labels": int(len(np.unique(truth))),
                        "ari_all": float(adjusted_rand_score(truth, prediction)),
                        "ari_control": float(
                            adjusted_rand_score(truth[control], prediction[control])
                        ),
                        "ari_condition": float(
                            adjusted_rand_score(truth[target], prediction[target])
                        ),
                        "neighbor_label_purity": purity,
                    }
                )
                if label_name.startswith("Broad") and method != "Paired composition oracle":
                    broad_prediction[method] = align(prediction, truth)

        sal_truth = obs.loc[target, "gt_sal_domain"].astype(str).to_numpy()
        blocks = {
            "Raw z": raw_z[target],
            "CellCharter hop 1": cellcharter[target, :32],
            "CellCharter hop 2": cellcharter[target, 32:64],
            "CellCharter hop 3": cellcharter[target, 64:96],
            "CellCharter hops 1-3": cellcharter[target],
            "Background decoder": decoder[target],
        }
        for block, representation in blocks.items():
            prediction = fit(representation, sal_truth)
            leakage_rows.append(
                {
                    "section": section,
                    "representation": block,
                    "salient_domain_ari": float(
                        adjusted_rand_score(sal_truth, prediction)
                    ),
                }
            )

        region_truth = obs["region"].astype(str).to_numpy()
        region_levels.update(np.unique(region_truth))
        plot_items[section] = {
            "obs": obs,
            "target": target,
            "truth": region_truth,
            "z_cellcharter": broad_prediction["z + CellCharter"],
            "decoder": broad_prediction["Background decoder"],
        }
        data.file.close()

    label_metrics = pd.DataFrame(label_rows)
    label_metrics.to_csv(OUT / "label_scale_per_section.csv", index=False)
    label_summary = (
        label_metrics.groupby(["label_scale", "method"])
        .agg(
            n_sections=("section", "size"),
            ari_all_mean=("ari_all", "mean"),
            ari_all_sd=("ari_all", "std"),
            ari_control_mean=("ari_control", "mean"),
            ari_condition_mean=("ari_condition", "mean"),
            neighbor_label_purity_mean=("neighbor_label_purity", "mean"),
        )
        .reset_index()
    )
    label_summary.to_csv(OUT / "label_scale_summary.csv", index=False)

    leakage = pd.DataFrame(leakage_rows)
    leakage.to_csv(OUT / "cellcharter_leakage_per_section.csv", index=False)
    leakage_summary = (
        leakage.groupby("representation")
        .agg(
            n_sections=("section", "size"),
            salient_domain_ari_mean=("salient_domain_ari", "mean"),
            salient_domain_ari_sd=("salient_domain_ari", "std"),
        )
        .reset_index()
        .sort_values("salient_domain_ari_mean", ascending=False)
    )
    leakage_summary.to_csv(OUT / "cellcharter_leakage_summary.csv", index=False)

    # Label-scale comparison.
    methods = (
        "Raw z",
        "z + CellCharter",
        "Background decoder",
        "Paired composition oracle",
    )
    scales = ("Fine subregion (7)", "Broad region (4)")
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    width = 0.18
    centers = np.arange(len(scales))
    for index, method in enumerate(methods):
        values = [
            label_summary.loc[
                (label_summary["label_scale"] == scale)
                & (label_summary["method"] == method),
                "ari_all_mean",
            ].iloc[0]
            for scale in scales
        ]
        ax.bar(
            centers + (index - 1.5) * width,
            values,
            width,
            label=method,
        )
    ax.set_xticks(centers, scales)
    ax.set_ylabel("Background ARI across five sections")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=2)
    ax.set_title("Does a broader anatomical label match the composition target better?")
    fig.savefig(OUT / "label_scale_ari.png", dpi=220)
    fig.savefig(OUT / "label_scale_ari.pdf")
    plt.close(fig)

    # Leakage by spatial hop.
    ordered = (
        "Raw z",
        "CellCharter hop 1",
        "CellCharter hop 2",
        "CellCharter hop 3",
        "CellCharter hops 1-3",
        "Background decoder",
    )
    means = leakage_summary.set_index("representation").loc[
        list(ordered), "salient_domain_ari_mean"
    ]
    sds = leakage_summary.set_index("representation").loc[
        list(ordered), "salient_domain_ari_sd"
    ]
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.bar(np.arange(len(ordered)), means, yerr=sds, capsize=4, color="#4c78a8")
    ax.set_xticks(np.arange(len(ordered)), ordered, rotation=25, ha="right")
    ax.set_ylabel("Salient-domain ARI from background representation")
    ax.set_title("Source of salient leakage in z + CellCharter")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.savefig(OUT / "cellcharter_leakage_by_hop.png", dpi=220)
    fig.savefig(OUT / "cellcharter_leakage_by_hop.pdf")
    plt.close(fig)

    # Broad region spatial maps.
    region_labels = sorted(region_levels)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(region_labels)))
    palette = dict(zip(region_labels, colors))
    fig, axes = plt.subplots(len(SECTIONS), 3, figsize=(13, 17), constrained_layout=True)
    for column, title in enumerate(
        ("True broad region", "z + CellCharter", "Background decoder")
    ):
        axes[0, column].set_title(title, fontsize=13)
    for row, section in enumerate(SECTIONS):
        item = plot_items[section]
        obs = item["obs"]
        target = item["target"]
        x = obs.loc[target, "center_x"].to_numpy()
        y = obs.loc[target, "center_y"].to_numpy()
        scatter(axes[row, 0], x, y, item["truth"][target], palette)
        scatter(axes[row, 1], x, y, item["z_cellcharter"][target], palette)
        scatter(axes[row, 2], x, y, item["decoder"][target], palette)
        axes[row, 0].set_ylabel(section, fontsize=12)
    fig.savefig(OUT / "broad_region_spatial.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "broad_region_spatial.pdf", bbox_inches="tight")
    plt.close(fig)

    print("\nLabel-scale summary\n", label_summary.to_string(index=False), flush=True)
    print("\nLeakage summary\n", leakage_summary.to_string(index=False), flush=True)
    print(f"\nSaved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
