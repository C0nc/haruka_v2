"""Prepare a balanced MERFISH aging subset for four-slot Haruka."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


EARLY = [3.4, 3.8, 4.3, 5.4, 6.6, 9.8]
MIDDLE = [12.9, 15.5, 15.8, 18.8, 19.8, 21.4, 23.5]
LATE = [24.6, 26.7, 28.5, 30.9, 32.6, 33.2, 34.5]


def group_age(value: float) -> str:
    if value in EARLY:
        return "early"
    if value in MIDDLE:
        return "middle"
    if value in LATE:
        return "late"
    raise ValueError(f"unrecognized age: {value}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/home/yancui/milabo/aging_coronal.h5ad?download=1")
    p.add_argument("--output", default="bench_aging/data/merfish_aging_100k.h5ad")
    p.add_argument("--per-mouse", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260807)
    args = p.parse_args()

    # Sequentially materialize once; scattered HDF5 fancy indexing is very slow.
    source = ad.read_h5ad(args.input)
    rng = np.random.default_rng(args.seed)
    chosen = []
    audit = {}
    # Random sampling within every mouse preserves each complete tissue section
    # without allowing large sections to dominate the contrastive objective.
    for mouse in sorted(source.obs.mouse_id.unique(), key=lambda x: int(x)):
        idx = np.flatnonzero(source.obs.mouse_id.to_numpy() == mouse)
        take = np.sort(rng.choice(idx, size=min(args.per_mouse, len(idx)), replace=False))
        chosen.append(take)
        audit[str(mouse)] = {"available": int(len(idx)), "selected": int(len(take))}
    selected = np.sort(np.concatenate(chosen))
    data = source[selected].copy()

    age = data.obs.age.astype(float).to_numpy()
    groups = np.array([group_age(x) for x in age])
    data.obs["age_group"] = pd.Categorical(groups, categories=["early", "middle", "late"])
    data.obs["condition"] = pd.Categorical(
        np.where(groups == "early", "control", "disease"),
        categories=["control", "disease"],
    )
    # The graph/CellCharter library is the mouse. Never use this as a removable
    # batch because exact age and mouse identity are one-to-one in this atlas.
    data.obs["slice_id"] = data.obs.mouse_id.astype(str).astype("category")
    data.obs["graph_id"] = data.obs.mouse_id.astype(str).astype("category")
    data.obs["technical_batch"] = data.obs.batch.astype(str).astype("category")
    data.obs["cell_type"] = data.obs.celltype.astype(str).astype("category")
    data.obs["region_label"] = data.obs.subregion.astype(str).astype("category")
    data.obs["perturbation"] = data.obs.age_group.astype(str).astype("category")
    data.layers["counts"] = data.X.copy()
    data.obs_names = pd.Index([f"mouse{x}:{name}" for x, name in zip(data.obs.mouse_id, data.obs_names)])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(output, compression="gzip")
    report = {
        "source": args.input,
        "shape": [int(data.n_obs), int(data.n_vars)],
        "seed": args.seed,
        "per_mouse": args.per_mouse,
        "ages": {"early": EARLY, "middle": MIDDLE, "late": LATE},
        "condition_counts": data.obs.condition.value_counts().astype(int).to_dict(),
        "age_group_counts": data.obs.age_group.value_counts().astype(int).to_dict(),
        "technical_batch_by_group": pd.crosstab(data.obs.age_group, data.obs.technical_batch).astype(int).to_dict(),
        "mouse_sampling": audit,
        "contract": {
            "training_condition": "early control versus pooled middle+late disease",
            "middle_late_labels_used_in_training": False,
            "graph_library": "mouse_id",
            "removable_batch": "technical A/B only",
            "exact_age_mouse_confounding": "one mouse per exact age",
        },
    }
    output.with_suffix(".audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
