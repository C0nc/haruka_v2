"""Audit BE CODEX regions for the Haruka protein benchmark."""

from __future__ import annotations

import argparse

import h5py
import numpy as np
import pandas as pd


def categorical(group: h5py.Group) -> np.ndarray:
    categories = np.asarray(group["categories"][:]).astype(str)
    codes = np.asarray(group["codes"][:], dtype=int)
    out = np.full(len(codes), "missing", dtype=object)
    good = codes >= 0
    out[good] = categories[codes[good]]
    return out.astype(str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="/home/yancui/trajectory/EsophagusAtlas_CODEX.h5mu"
    )
    parser.add_argument("--out")
    args = parser.parse_args()

    rows = []
    with h5py.File(args.source, "r") as handle:
        for sid in sorted(handle["mod"]):
            group = handle[f"mod/{sid}"]
            niche = categorical(group["obs/neigh_name"])
            composition = categorical(group["obs/neigh_sub"])
            protein = np.asarray(group["X"][:], dtype=np.float32)
            coords = np.asarray(group["obsm/spatial"][:], dtype=np.float32)
            rows.append(
                {
                    "section": sid,
                    "n_cells": len(niche),
                    "n_niches": len(np.unique(niche)),
                    "min_niche_cells": min(np.unique(niche, return_counts=True)[1]),
                    "n_compositions": len(np.unique(composition)),
                    "min_composition_cells": min(
                        np.unique(composition, return_counts=True)[1]
                    ),
                    "protein_min": float(protein.min()),
                    "protein_median": float(np.median(protein)),
                    "protein_p99": float(np.quantile(protein, 0.99)),
                    "protein_max": float(protein.max()),
                    "coord_x_span": float(np.ptp(coords[:, 0])),
                    "coord_y_span": float(np.ptp(coords[:, 1])),
                    "niche_counts": " | ".join(
                        f"{label}:{count}"
                        for label, count in zip(*np.unique(niche, return_counts=True))
                    ),
                    "composition_counts": " | ".join(
                        f"{label}:{count}"
                        for label, count in zip(
                            *np.unique(composition, return_counts=True)
                        )
                    ),
                }
            )
    table = pd.DataFrame(rows)
    if args.out:
        table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
