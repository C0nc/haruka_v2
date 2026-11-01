"""Diagnose and visualize background recovery in paired mouse-brain simulations.

The script compares the original Haruka z/s + CellCharter readout with the
background-Wasserstein variant. Clusters are fitted without labels, exactly as
in the benchmark. Ground-truth labels are used only after clustering to align
cluster colors for visualization and to calculate evaluation metrics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


ROOT = Path("/home/yancui/haruka_v2/bench_mouse_brain_scviva")
SECTIONS = ("A8", "A9", "B17", "B3", "B5")
OUT = ROOT / "results_cross_section_bg_diagnosis"
SEED = 0
N_INIT = 10


def latent_dir(section: str, wasserstein: bool) -> Path:
    if wasserstein:
        root = ROOT / "results_cross_section_bg_wass20"
    elif section == "A8":
        root = ROOT / "results_direct_kmeans"
    else:
        root = ROOT / "results_cross_section"
    return (
        root
        / "haruka_z_s_cellcharter"
        / f"mouse_brain_{section}_paired"
        / "haruka_z_s_cellcharter_seed0"
    )


def direct_dir(latent: Path) -> Path:
    with open(latent / "meta.json") as handle:
        return Path(json.load(handle)["source_result"])


def aligned_obs(data: ad.AnnData, embedding_dir: Path) -> pd.DataFrame:
    names = np.load(embedding_dir / "obs_names.npy", allow_pickle=True).astype(str)
    positions = data.obs_names.astype(str).get_indexer(names)
    if (positions < 0).any():
        raise ValueError(f"Rows in {embedding_dir} do not align to paired data")
    return data.obs.iloc[positions].copy()


def fit_clusters(x: np.ndarray, n_clusters: int) -> np.ndarray:
    return KMeans(
        n_clusters=n_clusters,
        random_state=SEED,
        n_init=N_INIT,
    ).fit_predict(np.asarray(x, dtype=np.float32))


def align_to_truth(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred_levels = np.unique(prediction)
    truth_levels = np.unique(truth)
    counts = np.zeros((len(pred_levels), len(truth_levels)), dtype=np.int64)
    for i, pred in enumerate(pred_levels):
        for j, label in enumerate(truth_levels):
            counts[i, j] = np.sum((prediction == pred) & (truth == label))
    rows, cols = linear_sum_assignment(-counts)
    mapping = {pred_levels[i]: truth_levels[j] for i, j in zip(rows, cols)}
    return np.asarray([mapping.get(value, "unmapped") for value in prediction])


def paired_cosine(x: np.ndarray, control: np.ndarray, target: np.ndarray) -> float:
    left = np.asarray(x[control], dtype=np.float32)
    right = np.asarray(x[target], dtype=np.float32)
    if len(left) != len(right):
        return float("nan")
    numerator = np.einsum("ij,ij->i", left, right)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return float(np.mean(numerator / np.maximum(denominator, 1e-8)))


def background_metrics(
    section: str,
    method: str,
    representation: np.ndarray,
    truth: np.ndarray,
    salient_truth: np.ndarray,
    control: np.ndarray,
    target: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    prediction = fit_clusters(representation, len(np.unique(truth)))
    control_pred = prediction[control]
    target_pred = prediction[target]
    leakage_pred = fit_clusters(representation[target], len(np.unique(salient_truth)))
    metrics = {
        "section": section,
        "method": method,
        "n_features": int(representation.shape[1]),
        "background_ari_all": float(adjusted_rand_score(truth, prediction)),
        "background_ari_control": float(
            adjusted_rand_score(truth[control], control_pred)
        ),
        "background_ari_condition": float(
            adjusted_rand_score(truth[target], target_pred)
        ),
        "paired_partition_ari": float(
            adjusted_rand_score(control_pred, target_pred)
        ),
        "paired_exact_cluster_fraction": float(np.mean(control_pred == target_pred)),
        "paired_cosine_mean": paired_cosine(representation, control, target),
        "salient_leakage_ari": float(
            adjusted_rand_score(salient_truth, leakage_pred)
        ),
    }
    return metrics, prediction


def scatter_labels(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    palette: dict[str, object],
    size: float,
) -> None:
    labels = np.asarray(labels).astype(str)
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(
            x[mask],
            y[mask],
            s=size,
            c=[palette.get(label, "#bdbdbd")],
            linewidths=0,
            rasterized=True,
        )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def confusion_table(truth: np.ndarray, aligned: np.ndarray) -> pd.DataFrame:
    table = pd.crosstab(
        pd.Series(truth, name="truth"),
        pd.Series(aligned, name="prediction"),
        normalize="index",
    )
    return table


def plot_confusions(confusions: dict[str, pd.DataFrame], path: Path) -> None:
    labels = sorted(set().union(*(x.index for x in confusions.values())))
    fig, axes = plt.subplots(1, len(confusions), figsize=(13, 5), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    for ax, (name, table) in zip(axes, confusions.items()):
        matrix = table.reindex(index=labels, columns=labels, fill_value=0.0)
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(name)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_xlabel("Aligned predicted subregion")
        ax.set_ylabel("True subregion")
    fig.colorbar(image, ax=axes.tolist(), shrink=0.75, label="Fraction within true label")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    section_plots: dict[str, dict[str, np.ndarray | pd.DataFrame]] = {}
    pooled_truth: list[np.ndarray] = []
    pooled_w0: list[np.ndarray] = []
    pooled_w20: list[np.ndarray] = []

    all_bg_labels: set[str] = set()
    all_sal_labels: set[str] = set()
    for section in SECTIONS:
        data_path = ROOT / "data_composition_pairs_all" / f"mouse_brain_{section}_paired.h5ad"
        data = ad.read_h5ad(data_path, backed="r")
        w0_dir = latent_dir(section, wasserstein=False)
        w20_dir = latent_dir(section, wasserstein=True)
        obs = aligned_obs(data, w0_dir)
        control = obs["condition"].astype(str).to_numpy() == "control"
        target = ~control
        if not np.allclose(
            obs.loc[control, ["center_x", "center_y"]].to_numpy(),
            obs.loc[target, ["center_x", "center_y"]].to_numpy(),
        ):
            raise ValueError(f"{section}: control and condition rows are not paired")
        truth = obs["gt_background_subregion"].astype(str).to_numpy()
        salient_truth = obs.loc[target, "gt_sal_domain"].astype(str).to_numpy()
        all_bg_labels.update(np.unique(truth))
        all_sal_labels.update(np.unique(salient_truth))

        arrays = {
            "Haruka z (lambda=0)": np.load(direct_dir(w0_dir) / "background.npy", mmap_mode="r"),
            "Haruka decoder (lambda=0)": np.load(direct_dir(w0_dir) / "spatial.npy", mmap_mode="r"),
            "Haruka z+CellCharter (lambda=0)": np.load(w0_dir / "spatial.npy", mmap_mode="r"),
            "Haruka z (lambda=20)": np.load(direct_dir(w20_dir) / "background.npy", mmap_mode="r"),
            "Haruka decoder (lambda=20)": np.load(direct_dir(w20_dir) / "spatial.npy", mmap_mode="r"),
            "Haruka z+CellCharter (lambda=20)": np.load(w20_dir / "spatial.npy", mmap_mode="r"),
        }
        predictions: dict[str, np.ndarray] = {}
        for method, representation in arrays.items():
            metrics, prediction = background_metrics(
                section,
                method,
                representation,
                truth,
                salient_truth,
                control,
                target,
            )
            metric_rows.append(metrics)
            predictions[method] = prediction

        w0_salient_rep = np.load(w0_dir / "salient_spatial.npy", mmap_mode="r")[target]
        w20_salient_rep = np.load(w20_dir / "salient_spatial.npy", mmap_mode="r")[target]
        w0_salient_pred = fit_clusters(w0_salient_rep, len(np.unique(salient_truth)))
        w20_salient_pred = fit_clusters(w20_salient_rep, len(np.unique(salient_truth)))

        w0_bg_pred = predictions["Haruka z+CellCharter (lambda=0)"]
        w20_bg_pred = predictions["Haruka z+CellCharter (lambda=20)"]
        w0_bg_aligned = align_to_truth(w0_bg_pred, truth)
        w20_bg_aligned = align_to_truth(w20_bg_pred, truth)
        w0_sal_aligned = align_to_truth(w0_salient_pred, salient_truth)
        w20_sal_aligned = align_to_truth(w20_salient_pred, salient_truth)

        pooled_truth.append(truth)
        pooled_w0.append(w0_bg_aligned)
        pooled_w20.append(w20_bg_aligned)
        section_plots[section] = {
            "obs": obs,
            "control": control,
            "target": target,
            "truth": truth,
            "salient_truth": salient_truth,
            "w0_bg": w0_bg_aligned,
            "w20_bg": w20_bg_aligned,
            "w0_sal": w0_sal_aligned,
            "w20_sal": w20_sal_aligned,
        }
        data.file.close()

    bg_labels = sorted(all_bg_labels)
    sal_labels = sorted(all_sal_labels)
    bg_colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(bg_labels)))
    sal_colors = plt.get_cmap("Set1")(np.linspace(0, 1, len(sal_labels)))
    bg_palette = dict(zip(bg_labels, bg_colors))
    sal_palette = dict(zip(sal_labels, sal_colors))

    # One compact cross-section figure: condition tissue only.
    fig, axes = plt.subplots(
        len(SECTIONS), 6, figsize=(25, 18), constrained_layout=True
    )
    column_titles = (
        "True background",
        "Haruka background lambda=0",
        "Haruka background lambda=20",
        "True salient domain",
        "Haruka salient lambda=0",
        "Haruka salient lambda=20",
    )
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=13)
    for row, section in enumerate(SECTIONS):
        item = section_plots[section]
        obs = item["obs"]
        target = item["target"]
        x = obs.loc[target, "center_x"].to_numpy()
        y = obs.loc[target, "center_y"].to_numpy()
        scatter_labels(axes[row, 0], x, y, item["truth"][target], bg_palette, 0.35)
        scatter_labels(axes[row, 1], x, y, item["w0_bg"][target], bg_palette, 0.35)
        scatter_labels(axes[row, 2], x, y, item["w20_bg"][target], bg_palette, 0.35)
        scatter_labels(axes[row, 3], x, y, item["salient_truth"], sal_palette, 0.35)
        scatter_labels(axes[row, 4], x, y, item["w0_sal"], sal_palette, 0.35)
        scatter_labels(axes[row, 5], x, y, item["w20_sal"], sal_palette, 0.35)
        axes[row, 0].set_ylabel(section, fontsize=12)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=label, markersize=6)
        for label, color in bg_palette.items()
    ]
    handles += [
        Line2D(
            [0], [0], marker="s", linestyle="", color=color, label=label, markersize=6
        )
        for label, color in sal_palette.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9)
    fig.savefig(OUT / "all_sections_background_salient_spatial.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "all_sections_background_salient_spatial.pdf", bbox_inches="tight")
    plt.close(fig)

    # Detailed per-section figures show whether the error is introduced by condition.
    for section in SECTIONS:
        item = section_plots[section]
        obs = item["obs"]
        control = item["control"]
        target = item["target"]
        control_x = obs.loc[control, "center_x"].to_numpy()
        control_y = obs.loc[control, "center_y"].to_numpy()
        target_x = obs.loc[target, "center_x"].to_numpy()
        target_y = obs.loc[target, "center_y"].to_numpy()
        fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
        scatter_labels(axes[0, 0], target_x, target_y, item["truth"][target], bg_palette, 0.7)
        axes[0, 0].set_title("True background")
        scatter_labels(axes[0, 1], control_x, control_y, item["w0_bg"][control], bg_palette, 0.7)
        axes[0, 1].set_title("lambda=0, control")
        scatter_labels(axes[0, 2], target_x, target_y, item["w0_bg"][target], bg_palette, 0.7)
        axes[0, 2].set_title("lambda=0, condition")
        scatter_labels(axes[0, 3], target_x, target_y, item["w20_bg"][target], bg_palette, 0.7)
        axes[0, 3].set_title("lambda=20, condition")
        scatter_labels(
            axes[1, 0], target_x, target_y, item["salient_truth"], sal_palette, 0.7
        )
        axes[1, 0].set_title("True salient")
        scatter_labels(axes[1, 1], target_x, target_y, item["w0_sal"], sal_palette, 0.7)
        axes[1, 1].set_title("Salient lambda=0")
        scatter_labels(axes[1, 2], target_x, target_y, item["w20_sal"], sal_palette, 0.7)
        axes[1, 2].set_title("Salient lambda=20")
        error = np.where(item["w0_bg"][target] == item["truth"][target], "correct", "wrong")
        scatter_labels(
            axes[1, 3],
            target_x,
            target_y,
            error,
            {"correct": "#d9d9d9", "wrong": "#d73027"},
            0.7,
        )
        axes[1, 3].set_title("lambda=0 background errors")
        fig.suptitle(f"Mouse brain {section}: Haruka spatial decomposition", fontsize=15)
        fig.savefig(OUT / f"{section}_background_salient_spatial.png", dpi=220, bbox_inches="tight")
        fig.savefig(OUT / f"{section}_background_salient_spatial.pdf", bbox_inches="tight")
        plt.close(fig)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "background_diagnostics_per_section.csv", index=False)
    summary = (
        metrics.groupby("method")
        .agg(
            n_sections=("section", "size"),
            background_ari_all_mean=("background_ari_all", "mean"),
            background_ari_all_sd=("background_ari_all", "std"),
            background_ari_control_mean=("background_ari_control", "mean"),
            background_ari_condition_mean=("background_ari_condition", "mean"),
            paired_partition_ari_mean=("paired_partition_ari", "mean"),
            paired_exact_cluster_fraction_mean=("paired_exact_cluster_fraction", "mean"),
            paired_cosine_mean=("paired_cosine_mean", "mean"),
            salient_leakage_ari_mean=("salient_leakage_ari", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / "background_diagnostics_summary.csv", index=False)

    truth_all = np.concatenate(pooled_truth)
    w0_all = np.concatenate(pooled_w0)
    w20_all = np.concatenate(pooled_w20)
    confusion_w0 = confusion_table(truth_all, w0_all)
    confusion_w20 = confusion_table(truth_all, w20_all)
    confusion_w0.to_csv(OUT / "background_confusion_lambda0.csv")
    confusion_w20.to_csv(OUT / "background_confusion_lambda20.csv")
    plot_confusions(
        {"Haruka lambda=0": confusion_w0, "Haruka lambda=20": confusion_w20},
        OUT / "background_confusion",
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Saved diagnostics to {OUT}", flush=True)


if __name__ == "__main__":
    main()
