"""Combine section-level disentanglement diagnostics and paired deltas."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SECTIONS = ("A8", "A9", "B17", "B3", "B5")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant-root",
        action="append",
        nargs=2,
        metavar=("VARIANT", "ROOT"),
        required=True,
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    frames = []
    for expected_variant, root in args.variant_root:
        root = Path(root)
        for section in SECTIONS:
            frame = pd.read_csv(root / f"summary_{section}.csv")
            frame.insert(0, "section", section)
            frames.append(frame.loc[frame["variant"].isin(["baseline", expected_variant])])
    per_section = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["section", "variant"]
    )
    metrics = [column for column in per_section.columns if column not in ("section", "variant")]
    summary = per_section.groupby("variant")[metrics].agg(["mean", "std"])

    baseline = per_section.loc[per_section["variant"] == "baseline"].set_index("section")
    delta_rows = []
    for variant in per_section["variant"].unique():
        if variant == "baseline":
            continue
        current = per_section.loc[per_section["variant"] == variant].set_index("section")
        for metric in metrics:
            delta = current[metric] - baseline[metric]
            delta_rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "mean_delta": delta.mean(),
                    "sd_delta": delta.std(),
                    "sections_improved": int((delta > 0).sum()),
                    "n_sections": int(delta.notna().sum()),
                }
            )
    deltas = pd.DataFrame(delta_rows)

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    per_section.to_csv(output / "per_section.csv", index=False)
    summary.to_csv(output / "mean_std.csv")
    deltas.to_csv(output / "paired_deltas.csv", index=False)
    print("Per-section results\n", per_section.to_string(index=False))
    print("\nMean +/- SD\n", summary.to_string())
    print("\nPaired deltas from baseline\n", deltas.to_string(index=False))


if __name__ == "__main__":
    main()
