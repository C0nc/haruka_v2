"""SIMVI itself, mapped onto the 2x2 (conda env `simvi`).

SIMVI has TWO latents, not four:

    intrinsic   z      MLP on own expression, graph-unaware  -> cell type
    interaction z_gat  GATv2 over neighbours' q_m            -> spatial variation

and no notion of a control arm at all. So it fills the 2x2 the same way the other
spatial-only methods do: the intrinsic latent goes in BOTH intrinsic slots and the
interaction latent in BOTH spatial slots. That is not a handicap imposed by the
harness -- it is SIMVI's actual claim. It asserts one decomposition of the tissue,
and the benchmark asks what that costs when the data also contains a
condition-specific axis it was never designed to separate.

Reference points this gives us that nothing else does:
  * `spatial -> gt_niche` for SIMVI vs Haruka's `z_sp`, on identical data. Haruka's
    spatial branch is a native reimplementation of SIMVI's, so this is the direct
    test of whether the reimplementation matches the original.
  * `background -> gt_arch` for SIMVI's intrinsic latent vs Haruka's `z`.

Requires scvi-tools <= 0.16.2 (`LossRecorder`, `parse_use_gpu_arg`), which is why
this runs in its own environment.

    python run_simvi.py --data data/simv3_d1_s0.h5ad --out_root results_v3
"""

from __future__ import annotations

import argparse
import json
import os
import time

import anndata as ad
import numpy as np
import scanpy as sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_root", default="results_v3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--n_latent", type=int, default=32)
    ap.add_argument("--n_spatial", type=int, default=16)
    ap.add_argument("--k", type=int, default=8, help="spatial graph k, matching Haruka's k_spatial")
    ap.add_argument("--use_gpu", action="store_true",
                    help="train on CUDA (SIMVI's None default selects CPU)")
    args = ap.parse_args()

    import torch
    from simvi.model import SimVI

    adata = ad.read_h5ad(args.data)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    SimVI.setup_anndata(adata, layer="counts", batch_key="slice_id")
    # `batch_key` here keeps the k-NN graph WITHIN each slice. Without it the
    # graph would link cells across the eight slices, which share a coordinate
    # frame -- every cell would acquire neighbours from other timepoints and
    # conditions at the same (x, y), and the interaction latent would be pooling
    # over the wrong tissue. Haruka's graph is built per `slice_id` for the same
    # reason, so this keeps the two spatial branches comparable.
    edge_index = SimVI.extract_edge_index(adata, batch_key="slice_id",
                                          spatial_key="spatial", n_neighbors=args.k)

    model = SimVI(adata, n_intrinsic=args.n_latent, n_spatial=args.n_spatial,
                  kl_gatweight=0.01)     # SIMVI's own 100x down-weight
    model.train(edge_index, max_epochs=args.max_epochs, batch_size=args.batch_size,
                use_gpu=args.use_gpu)
    z = np.asarray(model.get_latent_representation(edge_index, representation_kind="intrinsic"),
                   dtype=np.float32)
    z_gat = np.asarray(model.get_latent_representation(edge_index, representation_kind="interaction"),
                       dtype=np.float32)
    runtime = time.time() - t0

    tag = os.path.basename(args.data).replace(".h5ad", "")
    out_dir = os.path.join(args.out_root, tag, f"simvi_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    for name, arr in (("background", z), ("salient", z),
                      ("spatial", z_gat), ("salient_spatial", z_gat)):
        np.save(os.path.join(out_dir, f"{name}.npy"), arr)
    np.save(os.path.join(out_dir, "obs_names.npy"),
            adata.obs_names.to_numpy().astype(object), allow_pickle=True)
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(dict(method="simvi", single_rep=True, runtime_s=runtime,
                       n_latent=args.n_latent, n_spatial=args.n_spatial, k=args.k,
                       use_gpu=args.use_gpu,
                       n_layer=0,   # raw latents, no cellcharter aggregation
                       note="SIMVI has two latents and no control arm: intrinsic in "
                            "both intrinsic slots, interaction in both spatial slots"),
                  fh, indent=2)
    print(f"[simvi] wrote {out_dir}  z{z.shape} z_gat{z_gat.shape}  {runtime:.0f}s")


if __name__ == "__main__":
    main()
