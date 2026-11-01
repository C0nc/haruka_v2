"""Create identical paired-cell subsets feasible for dense CINEMA-OT."""

from __future__ import annotations

import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--pairs_per_section", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    manifest = pd.read_csv(args.manifest)
    for _, item in manifest.iterrows():
        data = ad.read_h5ad(item["path"])
        n = data.n_obs // 2
        if data.n_obs != 2 * n:
            raise ValueError(f"Expected an even paired dataset: {item['path']}")
        control = data.obs["condition"].astype(str).to_numpy() == "control"
        if not np.array_equal(np.flatnonzero(control), np.arange(n)):
            raise ValueError("Expected control rows followed by condition rows")
        bg0 = data.obs["gt_background_subregion"].astype(str).to_numpy()[:n]
        bg1 = data.obs["gt_background_subregion"].astype(str).to_numpy()[n:]
        if not np.array_equal(bg0, bg1):
            raise ValueError("Control/condition background labels are not paired")
        sal = data.obs["gt_sal_domain"].astype(str).to_numpy()[n:]
        strata = np.char.add(np.char.add(bg0.astype(str), "::"), sal.astype(str))
        all_idx = np.arange(n)
        chosen, _ = train_test_split(
            all_idx, train_size=min(args.pairs_per_section, n),
            random_state=args.seed, stratify=strata,
        )
        chosen = np.sort(chosen)
        paired_rows = np.concatenate([chosen, chosen + n])
        subset = data[paired_rows].copy()
        subset.uns["fair_baseline_subset"] = {
            "source": os.path.abspath(item["path"]),
            "pairs": int(len(chosen)),
            "selection": "stratified by background subregion x condition salient domain",
            "seed": int(args.seed),
            "reason": "official CINEMA-OT uses a dense arm-by-arm transport matrix",
        }
        section = str(item["section"])
        out = os.path.join(args.out_dir, f"mouse_brain_{section}_paired_5k.h5ad")
        subset.write_h5ad(out, compression="gzip")
        rows.append({"section": section, "n_pairs": len(chosen), "n_cells": subset.n_obs,
                     "path": os.path.abspath(out)})
        print(f"[save] {section}: {subset.n_obs} cells -> {out}", flush=True)
    pd.DataFrame(rows).sort_values("section").to_csv(
        os.path.join(args.out_dir, "manifest.csv"), index=False
    )


if __name__ == "__main__":
    main()
