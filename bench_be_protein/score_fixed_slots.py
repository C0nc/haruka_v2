"""Audit cell-level versus adaptive-spatial recovery for fixed Haruka."""

from __future__ import annotations

import argparse
import json
import os
import sys

import anndata as ad
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench_merfish"))
from common import ari, embed, gmm_on_embedding, load_embeddings  # noqa: E402


def score(x, truth):
    z = embed(np.asarray(x), seed=0)
    return ari(gmm_on_embedding(z, len(np.unique(truth)), 0), truth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb_dir", required=True)
    args = parser.parse_args()
    a = ad.read_h5ad(args.data)
    loaded = load_embeddings(args.emb_dir)
    treated = a.obs["condition"].astype(str).to_numpy() == "disease"
    background = a.obs["gt_background_domain"].astype(str).to_numpy()
    salient = a.obs.loc[treated, "gt_salient_domain"].astype(str).to_numpy()
    result = {
        "section": str(a.obs["section"].iloc[0]),
        "cell_background_ari": score(loaded["background"], background),
        "fused_background_ari": score(loaded["spatial"], background),
        "cell_salient_ari": score(np.asarray(loaded["salient"])[treated], salient),
        "fused_salient_ari": score(
            np.asarray(loaded["salient_spatial"])[treated], salient
        ),
    }
    with open(os.path.join(args.emb_dir, "fixed_slot_audit.json"), "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
