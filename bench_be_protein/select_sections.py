"""Rank BE CODEX sections for a balanced five-patient benchmark panel."""

from __future__ import annotations

import argparse

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import normalized_mutual_info_score

from inspect_be import categorical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="/home/yancui/trajectory/EsophagusAtlas_CODEX.h5mu"
    )
    parser.add_argument("--top_niches", type=int, default=6)
    parser.add_argument("--out", default="bench_be_protein/section_selection.csv")
    args = parser.parse_args()

    rows = []
    with h5py.File(args.source, "r") as handle:
        for sid in sorted(handle["mod"]):
            group = handle[f"mod/{sid}"]
            niche = categorical(group["obs/neigh_name"])
            composition = categorical(group["obs/neigh_sub"])
            labels, counts = np.unique(niche, return_counts=True)
            top = labels[np.argsort(-counts)[: args.top_niches]]
            keep = np.isin(niche, top)
            niche, composition = niche[keep], composition[keep]
            coords = np.asarray(group["obsm/spatial"][:], dtype=np.float32)[keep]
            distances, neighbors = cKDTree(coords).query(coords, k=2)
            rows.append(
                {
                    "section": sid,
                    "patient": sid.split("_")[0],
                    "n_retained": int(keep.sum()),
                    "retained_fraction": float(keep.mean()),
                    "min_top_niche": int(np.unique(niche, return_counts=True)[1].min()),
                    "min_composition": int(
                        np.unique(composition, return_counts=True)[1].min()
                    ),
                    "nmi_niche_composition": normalized_mutual_info_score(
                        niche, composition
                    ),
                    "median_1nn_distance": float(np.median(distances[:, 1])),
                    "same_niche_1nn": float(
                        np.mean(niche == niche[neighbors[:, 1]])
                    ),
                    "same_composition_1nn": float(
                        np.mean(composition == composition[neighbors[:, 1]])
                    ),
                    "top_niches": " | ".join(top),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
