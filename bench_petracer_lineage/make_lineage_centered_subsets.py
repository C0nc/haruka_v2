"""Downsample exact PEtracer pairs while preserving lineage microenvironments."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-original-cells", type=int, default=18000)
    parser.add_argument("--neighbor-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for section_i, path in enumerate(sorted(args.input_dir.glob("*.h5ad"))):
        data = ad.read_h5ad(path)
        n = data.n_obs // 2
        if 2 * n != data.n_obs:
            raise ValueError(f"{path}: expected equal paired arms")
        control = data.obs.iloc[:n]
        target = data.obs.iloc[n:]
        if not np.array_equal(
            control["pair_id"].astype(str).to_numpy(),
            target["pair_id"].astype(str).to_numpy(),
        ):
            raise ValueError(f"{path}: pair IDs are not aligned")
        eligible = target["lineage_effect_eligible"].astype(bool).to_numpy()
        selected = set(np.flatnonzero(eligible).tolist())

        coords = np.asarray(data.obsm["spatial"][:n], dtype=np.float64)
        if selected and args.neighbor_k > 0:
            tree = cKDTree(coords)
            _, neighbors = tree.query(
                coords[np.fromiter(selected, dtype=np.int64)],
                k=min(args.neighbor_k + 1, n),
            )
            selected.update(np.asarray(neighbors).ravel().astype(int).tolist())

        budget = args.max_original_cells - len(selected)
        if budget < 0:
            raise ValueError(
                f"{path}: lineage neighborhoods already contain {len(selected)} cells; "
                f"increase --max-original-cells"
            )
        rng = np.random.default_rng(args.seed + section_i)
        remaining = np.setdiff1d(np.arange(n), np.fromiter(selected, dtype=np.int64))
        if budget and len(remaining):
            # Proportional niche-stratified sampling preserves the section-wide
            # background distribution after retaining local lineage neighbors.
            niche = control["gt_background_niche"].astype(str).to_numpy()
            chunks = []
            for value in np.unique(niche[remaining]):
                pool = remaining[niche[remaining] == value]
                take = min(
                    len(pool),
                    int(np.floor(budget * len(pool) / max(len(remaining), 1))),
                )
                if take:
                    chunks.append(rng.choice(pool, take, replace=False))
            sampled = np.concatenate(chunks) if chunks else np.empty(0, dtype=int)
            short = min(budget - len(sampled), len(np.setdiff1d(remaining, sampled)))
            if short > 0:
                pool = np.setdiff1d(remaining, sampled)
                sampled = np.concatenate([sampled, rng.choice(pool, short, replace=False)])
            selected.update(sampled.astype(int).tolist())

        chosen = np.asarray(sorted(selected), dtype=np.int64)
        paired_rows = np.concatenate([chosen, chosen + n])
        subset = data[paired_rows].copy()
        subset.uns["lineage_centered_subset"] = {
            "source": str(path.resolve()),
            "all_lineage_cells_retained": True,
            "neighbor_k": args.neighbor_k,
            "max_original_cells": args.max_original_cells,
            "background_sampling": "proportional gt_background_niche stratification",
            "seed": args.seed + section_i,
        }
        suffix = f"_paired_{args.max_original_cells // 1000}k.h5ad"
        output = args.output_dir / path.name.replace("_paired.h5ad", suffix)
        subset.write_h5ad(output, compression="gzip")
        row = {
            "section": str(subset.obs["section_id"].iloc[0]),
            "n_original_full": n,
            "n_original_subset": len(chosen),
            "n_paired_subset": subset.n_obs,
            "n_lineage_cells": int(eligible.sum()),
            "fraction_lineage_in_subset": float(eligible.sum() / len(chosen)),
            "path": str(output.resolve()),
        }
        manifest.append(row)
        print(row, flush=True)

    pd.DataFrame(manifest).to_csv(args.output_dir / "manifest.csv", index=False)


if __name__ == "__main__":
    main()
