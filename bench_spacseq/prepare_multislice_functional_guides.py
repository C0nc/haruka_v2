"""Build a multi-slice dataset for guide-invariant target-function learning.

Targets must have two independent sgRNAs with at least 50 tumor anchors in
aggregate and at least 30 combined anchors in every slice.  Each guide is
sampled separately (up to 40 anchors per slice), all eligible NT anchors are
retained, and the final graph is ordinary within-slice physical KNN rather than
being partitioned by target label.
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prepare_multislice_all_guides as base  # noqa: E402


MIN_GUIDE_TOTAL = 50
MIN_TARGET_PER_SLICE = 30
MAX_ANCHORS_PER_GUIDE_SLICE = 40
SELECTED_GUIDES: dict[str, list[str]] = {}


def select_functional_targets(slice_data, min_anchors, require_guides):
    rows = []
    for slice_id, data in slice_data.items():
        index = np.flatnonzero(data["single"] & data["tumor"])
        for row in index:
            feature = data["single_feature"][row]
            target = str(data["feature_target"][feature])
            if target != "NT":
                rows.append((slice_id, target, str(data["names"][feature])))
    frame = pd.DataFrame(rows, columns=["slice_id", "target", "guide"])
    table = frame.groupby(["target", "guide", "slice_id"]).size().unstack(fill_value=0)
    guide_total = table.sum(axis=1)
    powered = guide_total[guide_total >= MIN_GUIDE_TOTAL]
    target_support = table.groupby(level=0).sum()
    targets = sorted(
        target for target in target_support.index
        if target_support.loc[target].min() >= MIN_TARGET_PER_SLICE
        and int((powered.index.get_level_values("target") == target).sum()) >= require_guides
    )
    SELECTED_GUIDES.clear()
    for target in targets:
        choices = powered.loc[target].sort_values(ascending=False)
        SELECTED_GUIDES[target] = choices.index.astype(str).tolist()
    guide_counts = pd.Series({target: len(SELECTED_GUIDES[target]) for target in targets})
    return targets, target_support, guide_counts


def guide_balanced_anchors(data, targets, ignored_n_per_group, seed):
    rng = np.random.default_rng(seed)
    feature = data["single_feature"]
    names = np.full(len(feature), "", dtype=object)
    valid = feature >= 0
    names[valid] = data["names"][feature[valid]]
    target = np.full(len(feature), "", dtype=object)
    target[valid] = data["feature_target"][feature[valid]]
    eligible = data["single"] & data["tumor"]
    result = {"NT": np.flatnonzero(eligible & (target == "NT"))}
    for function in targets:
        selected = []
        for guide in SELECTED_GUIDES[function]:
            index = np.flatnonzero(eligible & (names == guide))
            if len(index) > MAX_ANCHORS_PER_GUIDE_SLICE:
                index = np.sort(
                    rng.choice(index, MAX_ANCHORS_PER_GUIDE_SLICE, replace=False)
                )
            selected.append(index)
        result[function] = np.sort(np.concatenate(selected))
    return result


def main():
    # Reuse the battle-tested loader/context builder while replacing only target
    # selection and anchor balancing.
    base.select_supported_targets = select_functional_targets
    base.balanced_anchor_indices = guide_balanced_anchors
    parser = base.argparse.ArgumentParser()
    parser.add_argument("--raw_root", default="bench_spacseq/raw")
    parser.add_argument(
        "--out",
        default="bench_spacseq/data/spac_lung_functional_guides_k8_hvg3000.h5ad",
    )
    parser.add_argument("--min_anchors_per_slice", type=int, default=MIN_TARGET_PER_SLICE)
    parser.add_argument("--min_guides_per_target", type=int, default=2)
    parser.add_argument("--max_anchors_per_group", type=int, default=10_000)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--n_hvg", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    base.build(args)

    # Do not let target labels define graph membership.  The GAT sees only the
    # ordinary physical neighborhood within each cancer slice.
    output = Path(args.out)
    adata = ad.read_h5ad(output)
    adata.obs["graph_id"] = adata.obs["slice_id"].astype(str).astype("category")
    adata.obs["function"] = adata.obs["perturbation"].astype(str).astype("category")
    adata.obs["guide_identity"] = adata.obs["guide_name"].astype(str).astype("category")
    adata.uns["functional_guide_design"] = {
        "targets": list(SELECTED_GUIDES),
        "guides_by_target": SELECTED_GUIDES,
        "min_guide_total": MIN_GUIDE_TOTAL,
        "min_target_per_slice": MIN_TARGET_PER_SLICE,
        "max_anchors_per_guide_slice": MAX_ANCHORS_PER_GUIDE_SLICE,
        "anchor_only_function_supervision": True,
        "graph_partition": "slice_only_physical_knn",
    }
    adata.write_h5ad(output, compression="gzip")
    print(adata.uns["functional_guide_design"])
    print(f"[rewrite] {output}: graph_id=slice_id, shape={adata.shape}")


if __name__ == "__main__":
    main()
