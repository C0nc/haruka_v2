"""Aggregate all direct-KMeans mouse-brain benchmarks over independent sections."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path("/home/yancui/haruka_v2/bench_mouse_brain_scviva")
RESULTS = ROOT / "results_cross_section"
OUT = RESULTS / "summary_all_methods"
SECTIONS = ["A8", "A9", "B17", "B3", "B5"]

METHOD_ORDER = [
    "Haruka z/s + CellCharter",
    "Mini-batch CINEMA-OT + CellCharter",
    "ContrastiveVI z/s + CellCharter",
    "scVI + CellCharter",
    "Expression + neighborhood aggregation",
    "Observed 20-NN composition [diagnostic]",
    "Paired composition oracle [diagnostic]",
]


def read_direct(path: Path, method: str) -> dict:
    with path.open() as handle:
        result = json.load(handle)
    return {
        "section": result["section"],
        "method": method,
        "background_subregion_ari": result["scores"]["background_subregion"]["ari"],
        "salient_domain_ari": result["scores"]["salient_domain"]["ari"],
        "result_json": str(path.resolve()),
    }


def baseline_path(section: str, leaf: str) -> Path:
    if section == "A8":
        return (
            ROOT
            / "results_direct_kmeans/baselines"
            / f"mouse_brain_{section}_paired"
            / leaf
            / "direct_kmeans_ari.json"
        )
    return (
        RESULTS
        / "baselines"
        / f"mouse_brain_{section}_paired"
        / leaf
        / "direct_kmeans_ari.json"
    )


def composition_path(section: str) -> Path:
    if section == "A8":
        return ROOT / "results_direct_kmeans/composition_controls_A8.json"
    return RESULTS / f"composition_controls_{section}.json"


def main() -> None:
    prior = pd.read_csv(RESULTS / "summary/per_section_metrics.csv")
    rows = prior.to_dict("records")

    baselines = {
        "contrastivevi_2x2_seed0": "ContrastiveVI z/s + CellCharter",
        "cellcharter_2x2_seed0": "scVI + CellCharter",
        "pca_2x2_seed0": "Expression + neighborhood aggregation",
    }
    for section in SECTIONS:
        for leaf, method in baselines.items():
            rows.append(read_direct(baseline_path(section, leaf), method))

        path = composition_path(section)
        with path.open() as handle:
            control = json.load(handle)
        for key, method in [
            ("observed_local_composition", "Observed 20-NN composition [diagnostic]"),
            ("paired_composition_oracle", "Paired composition oracle [diagnostic]"),
        ]:
            score = control["scores"][key]
            rows.append(
                {
                    "section": section,
                    "method": method,
                    "background_subregion_ari": score["background_subregion_ari"],
                    "salient_domain_ari": score["salient_domain_ari"],
                    "result_json": str(path.resolve()),
                }
            )

    frame = pd.DataFrame(rows)
    frame["method"] = pd.Categorical(frame["method"], METHOD_ORDER, ordered=True)
    frame = frame.sort_values(["method", "section"]).reset_index(drop=True)

    summaries = []
    for method, group in frame.groupby("method", observed=True, sort=False):
        for metric in ["background_subregion_ari", "salient_domain_ari"]:
            values = group[metric].to_numpy(float)
            sem = stats.sem(values)
            half_width = stats.t.ppf(0.975, len(values) - 1) * sem
            summaries.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_sections": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "sem": sem,
                    "ci95_low": values.mean() - half_width,
                    "ci95_high": values.mean() + half_width,
                }
            )
    summary = pd.DataFrame(summaries)
    summary["method"] = pd.Categorical(summary["method"], METHOD_ORDER, ordered=True)

    rank_rows = []
    for metric in ["background_subregion_ari", "salient_domain_ari"]:
        metric_summary = summary[summary["metric"] == metric].copy()
        metric_summary["rank_all"] = metric_summary["mean"].rank(ascending=False, method="min")
        learned = ~metric_summary["method"].astype(str).str.contains("diagnostic")
        metric_summary["rank_learned"] = np.nan
        metric_summary.loc[learned, "rank_learned"] = metric_summary.loc[learned, "mean"].rank(
            ascending=False, method="min"
        )
        rank_rows.append(metric_summary)
    ranking = pd.concat(rank_rows, ignore_index=True).sort_values(["metric", "rank_all"])

    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "per_section_metrics.csv", index=False)
    summary.to_csv(OUT / "mean_sd_ci95.csv", index=False)
    ranking.to_csv(OUT / "rankings.csv", index=False)

    labels = [
        "Haruka z/s + CC",
        "Mini CINEMA-OT + CC",
        "ContrastiveVI z/s + CC",
        "scVI + CC",
        "Expression + neighbor agg.",
        "Observed composition*",
        "Paired oracle*",
    ]
    colors = ["#CC4C4C", "#4C78A8", "#72B7B2", "#F2CF5B", "#9D755D", "#B5B5B5", "#6F6F6F"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=True)
    rng = np.random.default_rng(17)
    for ax, metric, title in zip(
        axes,
        ["background_subregion_ari", "salient_domain_ari"],
        ["Background subregion", "Salient domain"],
    ):
        for y, (method, label, color) in enumerate(zip(METHOD_ORDER, labels, colors)):
            vals = frame.loc[frame["method"] == method, metric].to_numpy(float)
            mean = vals.mean()
            sd = vals.std(ddof=1)
            jitter = rng.uniform(-0.10, 0.10, len(vals))
            ax.scatter(vals, y + jitter, s=25, color=color, alpha=0.58, zorder=2)
            ax.errorbar(mean, y, xerr=sd, fmt="o", ms=7, color=color, capsize=4, lw=2.0, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Adjusted Rand index (mean ± SD; dots = sections)")
        ax.set_xlim(-0.03, 1.02)
        ax.grid(axis="x", alpha=0.22)
        ax.invert_yaxis()
    axes[0].set_yticks(range(len(labels)), labels)
    fig.suptitle("Mouse-brain paired simulation: direct KMeans on each representation (n=5 sections)")
    fig.text(0.01, 0.015, "* Annotation/pairing-informed diagnostic control; not a learned-method competitor.", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(OUT / "all_methods_ari_errorbars.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "all_methods_ari_errorbars.pdf", bbox_inches="tight")
    plt.close(fig)

    print(summary[["method", "metric", "n_sections", "mean", "sd"]].to_string(index=False))


if __name__ == "__main__":
    main()
