"""Plot cross-section metrics and representative spatial disentanglement maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


METHOD_ORDER = ("baseline", "zs1", "zs5")
METHOD_LABELS = {
    "baseline": "Original",
    "zs1": "Independence λ=1",
    "zs5": "Independence λ=5",
}
METHOD_COLORS = {"baseline": "#777777", "zs1": "#2878B5", "zs5": "#D9534F"}


def aligned_clusters(x: np.ndarray, truth: np.ndarray) -> np.ndarray:
    categories, encoded = np.unique(truth.astype(str), return_inverse=True)
    prediction = KMeans(
        n_clusters=len(categories), random_state=0, n_init=20
    ).fit_predict(np.asarray(x, dtype=np.float32))
    contingency = np.zeros((len(categories), len(categories)), dtype=np.int64)
    np.add.at(contingency, (prediction, encoded), 1)
    rows, columns = linear_sum_assignment(-contingency)
    mapping = {row: categories[column] for row, column in zip(rows, columns)}
    return np.asarray([mapping[value] for value in prediction])


def embedding_dir(root: Path, variant: str, section: str) -> Path:
    tag = f"mouse_brain_{section}_paired"
    if variant == "baseline":
        result_root = root / ("results_direct_kmeans" if section == "A8" else "results_cross_section")
    else:
        result_root = root / f"results_disentangle_full_{variant}"
    return result_root / "haruka_z_s_cellcharter" / tag / "haruka_z_s_cellcharter_seed0"


def plot_benchmark(frame: pd.DataFrame, output: Path) -> None:
    panels = (
        ("background_fine_ari", "Fine background ARI", True),
        ("background_broad_ari", "Broad background ARI", True),
        ("salient_domain_ari", "Salient-domain ARI", True),
        ("z_salient_leakage_ari", "Salient leakage in z", False),
        ("cv_r2_z_from_s", "Held-out R²: z ← s", False),
        ("s_background_broad_leakage_ari", "Broad background leakage in s", False),
    )
    sections = ("A8", "A9", "B17", "B3", "B5")
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), constrained_layout=True)
    for ax, (metric, title, higher) in zip(axes.flat, panels):
        for variant in METHOD_ORDER:
            current = frame.loc[frame["variant"] == variant].set_index("section").loc[list(sections)]
            ax.plot(
                sections,
                current[metric],
                marker="o",
                linewidth=2,
                markersize=5,
                color=METHOD_COLORS[variant],
                label=METHOD_LABELS[variant],
            )
        ax.set_title(title)
        ax.set_ylabel("Higher is better" if higher else "Lower is better")
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Haruka directional z ← s disentanglement benchmark", fontsize=15)
    fig.savefig(output / "disentanglement_benchmark.png", dpi=240)
    fig.savefig(output / "disentanglement_benchmark.pdf")
    plt.close(fig)



def plot_method_boxplot(root: Path, disentangle: pd.DataFrame, output: Path) -> None:
    prior = pd.read_csv(
        root / "results_cross_section" / "summary_all_methods" / "per_section_metrics.csv"
    )
    learned = (
        "Haruka z/s + CellCharter",
        "Mini-batch CINEMA-OT + CellCharter",
        "ContrastiveVI z/s + CellCharter",
        "scVI + CellCharter",
        "Expression + neighborhood aggregation",
    )
    prior = prior.loc[prior["method"].isin(learned)].copy()
    zs5 = disentangle.loc[disentangle["variant"] == "zs5"].copy()
    zs5 = zs5.rename(columns={"background_fine_ari": "background_subregion_ari"})
    zs5["method"] = "Haruka λ=5 z/s + CellCharter"
    frame = pd.concat(
        [
            zs5[["section", "method", "background_subregion_ari", "salient_domain_ari"]],
            prior[["section", "method", "background_subregion_ari", "salient_domain_ari"]],
        ],
        ignore_index=True,
    )
    frame.to_csv(output / "lambda5_vs_baselines_per_section.csv", index=False)
    (
        frame.groupby("method")
        .agg(
            background_mean=("background_subregion_ari", "mean"),
            background_sd=("background_subregion_ari", "std"),
            salient_mean=("salient_domain_ari", "mean"),
            salient_sd=("salient_domain_ari", "std"),
        )
        .to_csv(output / "lambda5_vs_baselines_mean_sd.csv")
    )
    order = (
        "Haruka λ=5 z/s + CellCharter",
        "Haruka z/s + CellCharter",
        "Mini-batch CINEMA-OT + CellCharter",
        "ContrastiveVI z/s + CellCharter",
        "scVI + CellCharter",
        "Expression + neighborhood aggregation",
    )
    labels = (
        "Haruka λ=5",
        "Haruka original",
        "Mini CINEMA-OT",
        "ContrastiveVI",
        "scVI",
        "Expression",
    )
    colors = ("#D9534F", "#888888", "#4C78A8", "#72B7B2", "#F2CF5B", "#9D755D")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.9), constrained_layout=True)
    rng = np.random.default_rng(19)
    for ax, metric, title in zip(
        axes,
        ("background_subregion_ari", "salient_domain_ari"),
        ("Fine-grained background subregion", "Salient spatial domain"),
    ):
        values = [
            frame.loc[frame["method"] == method, metric].to_numpy(float)
            for method in order
        ]
        boxes = ax.boxplot(
            values,
            positions=np.arange(len(order)),
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.4},
            whiskerprops={"linewidth": 1.1},
            capprops={"linewidth": 1.1},
        )
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        for index, (method, color) in enumerate(zip(order, colors)):
            current = frame.loc[frame["method"] == method].set_index("section")
            section_order = ("A8", "A9", "B17", "B3", "B5")
            points = current.loc[list(section_order), metric].to_numpy(float)
            jitter = rng.uniform(-0.09, 0.09, len(points))
            ax.scatter(
                index + jitter,
                points,
                s=25,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
        ax.set_title(title)
        ax.set_ylabel("Adjusted Rand index")
        ax.set_xticks(np.arange(len(order)), labels, rotation=24, ha="right")
        ax.set_ylim(-0.03, 1.02)
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Haruka λ=5 versus learned baselines (five independent brain sections)",
        fontsize=15,
    )
    fig.savefig(output / "lambda5_vs_baselines_boxplot.png", dpi=240)
    fig.savefig(output / "lambda5_vs_baselines_boxplot.pdf")
    plt.close(fig)

def scatter_map(ax, coordinates, labels, palette, title, indices) -> None:
    colors = np.asarray([palette[str(value)] for value in labels[indices]])
    ax.scatter(
        coordinates[indices, 0],
        coordinates[indices, 1],
        c=colors,
        s=1.0,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)


def add_palette_legend(ax, palette, ncol: int) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=4,
            label=label,
        )
        for label, color in palette.items()
    ]
    ax.legend(
        handles=handles, loc="lower left", frameon=True, fontsize=5.5,
        ncol=ncol, columnspacing=0.5, handletextpad=0.2, borderpad=0.25,
    )


def plot_spatial(root: Path, output: Path) -> None:
    sections = ("A9", "B3")
    fig, axes = plt.subplots(2, 6, figsize=(18, 7.3), constrained_layout=True)
    rng = np.random.default_rng(0)
    for row, section in enumerate(sections):
        data = ad.read_h5ad(
            root / "data_composition_pairs_all" / f"mouse_brain_{section}_paired.h5ad",
            backed="r",
        )
        obs = data.obs
        coordinates = np.asarray(data.obsm["spatial"])
        condition = obs["condition"].astype(str).to_numpy()
        control = np.flatnonzero(condition == "control")
        target = np.flatnonzero(condition == "disease")
        fine = obs["gt_background_subregion"].astype(str).to_numpy()
        salient = obs["gt_sal_domain"].astype(str).to_numpy()
        region_categories = np.unique(fine)
        salient_categories = np.unique(salient[target])
        region_colors = plt.get_cmap("tab10")(np.arange(len(region_categories)))
        salient_colors = plt.get_cmap("Set2")(np.arange(len(salient_categories)))
        region_palette = dict(zip(region_categories, region_colors))
        salient_palette = dict(zip(salient_categories, salient_colors))
        control_show = np.sort(
            rng.choice(control, size=min(30000, len(control)), replace=False)
        )
        target_show = np.sort(
            rng.choice(target, size=min(30000, len(target)), replace=False)
        )

        base = embedding_dir(root, "baseline", section)
        zs5 = embedding_dir(root, "zs5", section)
        base_bg = aligned_clusters(np.load(base / "spatial.npy")[control], fine[control])
        zs5_bg = aligned_clusters(np.load(zs5 / "spatial.npy")[control], fine[control])
        base_sal = aligned_clusters(
            np.load(base / "salient_spatial.npy")[target], salient[target]
        )
        zs5_sal = aligned_clusters(
            np.load(zs5 / "salient_spatial.npy")[target], salient[target]
        )
        # Predictions above are indexed within their arm; expand only for a shared plot helper.
        base_bg_full = np.empty(len(obs), dtype=object)
        zs5_bg_full = np.empty(len(obs), dtype=object)
        base_sal_full = np.empty(len(obs), dtype=object)
        zs5_sal_full = np.empty(len(obs), dtype=object)
        base_bg_full[control], zs5_bg_full[control] = base_bg, zs5_bg
        base_sal_full[target], zs5_sal_full[target] = base_sal, zs5_sal

        scatter_map(axes[row, 0], coordinates, fine, region_palette, f"{section}: true fine background", control_show)
        scatter_map(axes[row, 1], coordinates, base_bg_full, region_palette, "Original z", control_show)
        scatter_map(axes[row, 2], coordinates, zs5_bg_full, region_palette, "λ=5 z", control_show)
        scatter_map(axes[row, 3], coordinates, salient, salient_palette, "True salient domain", target_show)
        scatter_map(axes[row, 4], coordinates, base_sal_full, salient_palette, "Original s", target_show)
        scatter_map(axes[row, 5], coordinates, zs5_sal_full, salient_palette, "λ=5 s", target_show)
        add_palette_legend(axes[row, 0], region_palette, ncol=2)
        add_palette_legend(axes[row, 3], salient_palette, ncol=1)
        data.file.close()
    fig.suptitle("Fine-background and salient spatial recovery with Haruka λ=5", fontsize=15)
    fig.savefig(output / "disentanglement_spatial_A9_B3.png", dpi=240)
    fig.savefig(output / "disentanglement_spatial_A9_B3.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary)
    plot_benchmark(summary, output)
    plot_method_boxplot(root, summary, output)
    plot_spatial(root, output)
    print(f"Saved figures to {output}")


if __name__ == "__main__":
    main()
