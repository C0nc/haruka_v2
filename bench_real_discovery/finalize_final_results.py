#!/usr/bin/env python3
"""Package shared AD outputs and create deterministic final figure indexes."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd


def copy_matching(source: Path, destination: Path, prefix: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob(f"{prefix}*")):
        if path.is_file():
            relative = path.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/yancui/haruka_v2/final_results"))
    args = parser.parse_args()
    shared = args.root / "ad_shared"
    for dataset, prefix in (("starmap_ad", "starmap"), ("codex_ad", "codex")):
        copy_matching(shared / "plaque_distance", args.root / dataset / "biology/plaque_distance", prefix)
        copy_matching(shared / "trajectory", args.root / dataset / "biology/trajectory", prefix)

    rows = []
    for path in sorted(args.root.rglob("*.png")):
        relative = path.relative_to(args.root)
        dataset = relative.parts[0]
        rows.append({
            "dataset": dataset,
            "relative_path": str(relative),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.root / "figure_index.csv", index=False)
    lines = ["# Final figure index", ""]
    for dataset, part in frame.groupby("dataset", sort=True):
        lines.extend([f"## {dataset}", ""])
        for item in part.itertuples(index=False):
            lines.append(f"- [{Path(item.relative_path).name}]({item.relative_path})")
        lines.append("")
    (args.root / "FIGURE_INDEX.md").write_text("\n".join(lines) + "\n")
    print(frame.groupby("dataset").size().rename("n_figures").to_string())
    print(f"total figures: {len(frame)}")


if __name__ == "__main__":
    main()
