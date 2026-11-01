"""Give every baseline the same fine-Cell-Type background composition block."""

from __future__ import annotations

import argparse
import json
import os
import sys

import anndata as ad
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "bench_merfish"))
from common import load_embeddings, save_embeddings  # noqa: E402
from apply_celltype_background import celltype_composition, compress  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    a = ad.read_h5ad(args.data)
    loaded = load_embeddings(args.input_dir)
    coords = np.asarray(a.obsm["spatial"])
    samples = a.obs["slice_id"].astype(str).to_numpy()
    ct = celltype_composition(
        a.obs["cell_type_label"].astype(str).to_numpy(), coords, samples
    )
    spatial = np.concatenate(
        [compress(np.asarray(loaded["spatial"]), 24), compress(ct, 8)], axis=1
    )
    meta = loaded.get("meta", {})
    save_embeddings(
        args.output_dir,
        background=loaded["background"],
        salient=loaded["salient"],
        spatial=spatial,
        salient_spatial=loaded["salient_spatial"],
        obs_names=loaded["obs_names"],
        method=f"{meta.get('method', 'baseline')}_celltype",
        cell_composition_annotation="Cell Type",
        background_method_dim=24,
        background_composition_dim=8,
        source_method_dir=os.path.abspath(args.input_dir),
        supervised_niche_or_salient_labels=False,
    )


if __name__ == "__main__":
    main()
