"""Summarize direct-KMeans ARIs with sections as independent replicates."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats


def read_scores(method, paths):
    rows = []
    for path in paths:
        with open(path) as handle:
            result = json.load(handle)
        rows.append(
            {
                "section": result["section"],
                "method": method,
                "background_subregion_ari": result["scores"]["background_subregion"]["ari"],
                "salient_domain_ari": result["scores"]["salient_domain"]["ari"],
                "result_json": os.path.abspath(path),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--haruka", nargs="+", required=True)
    parser.add_argument("--cinema", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    frame = pd.DataFrame(
        read_scores("Haruka z/s + CellCharter", args.haruka)
        + read_scores("Mini-batch CINEMA-OT + CellCharter", args.cinema)
    ).sort_values(["section", "method"])
    metrics = ["background_subregion_ari", "salient_domain_ari"]
    summaries = []
    for method, group in frame.groupby("method", sort=False):
        for metric in metrics:
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

    wide = frame.pivot(index="section", columns="method", values=metrics)
    paired = []
    for metric in metrics:
        haruka = wide[(metric, "Haruka z/s + CellCharter")]
        cinema = wide[(metric, "Mini-batch CINEMA-OT + CellCharter")]
        difference = haruka - cinema
        test = stats.ttest_rel(haruka, cinema)
        sem = stats.sem(difference)
        half_width = stats.t.ppf(0.975, len(difference) - 1) * sem
        paired.append(
            {
                "metric": metric,
                "n_sections": len(difference),
                "haruka_minus_cinema_mean": difference.mean(),
                "difference_sd": difference.std(ddof=1),
                "ci95_low": difference.mean() - half_width,
                "ci95_high": difference.mean() + half_width,
                "paired_t_pvalue": test.pvalue,
                "haruka_wins": int((difference > 0).sum()),
            }
        )
    paired = pd.DataFrame(paired)

    os.makedirs(args.out_dir, exist_ok=True)
    frame.to_csv(os.path.join(args.out_dir, "per_section_metrics.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "mean_sd_ci95.csv"), index=False)
    paired.to_csv(os.path.join(args.out_dir, "paired_differences.csv"), index=False)
    print("Per section\n", frame.to_string(index=False), sep="")
    print("\nMean / error bars\n", summary.to_string(index=False), sep="")
    print("\nPaired differences\n", paired.to_string(index=False), sep="")


if __name__ == "__main__":
    main()
