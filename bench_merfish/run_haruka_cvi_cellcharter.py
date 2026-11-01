"""ContrastiveVI intrinsic factors with CellCharter spatial encoding.

This is the clean non-SIMVI Haruka benchmark requested for the hybrid 2x2 data:

    background intrinsic  = ContrastiveVI z
    salient intrinsic     = ContrastiveVI s
    background spatial    = CellCharter aggregate(z)
    salient spatial       = CellCharter aggregate(s), with control s set to zero

The spatial construction follows the separation used by scVIVA: first learn a
spatially unaware single-cell state, then encode its local environment.  We use
CellCharter's multi-hop aggregation as the environment encoder, so this runner
does not instantiate a GAT and has no SIMVI KL or mutual-information terms.
No biological labels or ground-truth factors enter training.
"""

from __future__ import annotations

import argparse
import os

import anndata as ad
import numpy as np
import scanpy as sc

from common import Timer, save_embeddings
from run_baselines_2x2 import _cc_aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_root", default="results_cvi_cellcharter_spatial")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--n_latent", type=int, default=32)
    parser.add_argument("--batch_key", default="timepoint")
    parser.add_argument("--condition_key", default="condition")
    parser.add_argument("--target_label", default="disease")
    args = parser.parse_args()

    import scvi
    from scvi.external import ContrastiveVI

    adata = ad.read_h5ad(args.data)
    required = {args.batch_key, args.condition_key, "slice_id"}
    missing = sorted(required.difference(adata.obs.columns))
    if missing:
        raise KeyError(f"Missing required obs columns: {missing}")
    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] is required")

    a = adata.copy()
    if "counts" not in a.layers:
        a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

    condition = a.obs[args.condition_key].astype(str).to_numpy()
    target = np.flatnonzero(condition == args.target_label)
    background = np.flatnonzero(condition != args.target_label)
    if len(target) == 0 or len(background) == 0:
        raise ValueError("ContrastiveVI requires non-empty target and background arms")

    with Timer() as timer:
        scvi.settings.seed = args.seed
        ContrastiveVI.setup_anndata(a, layer="counts", batch_key=args.batch_key)
        model = ContrastiveVI(
            a,
            n_me=0,
            n_background_latent=args.n_latent,
            n_salient_latent=args.n_latent,
            use_observed_lib_size=True,
        )
        model.train(
            background_indices=background,
            target_indices=target,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            train_size=1.0,
            early_stopping=False,
            gradient_clip_val=1.0,
        )
        z = model.get_latent_representation(
            a, representation_kind="background"
        ).astype(np.float32)
        s = model.get_latent_representation(
            a, representation_kind="salient"
        ).astype(np.float32)

        # A control cell has no condition-specific state by the ContrastiveVI
        # generative contract. Zero it before neighborhood encoding so control
        # posterior noise cannot manufacture a salient spatial environment.
        s_for_spatial = s.copy()
        s_for_spatial[background] = 0.0
        z_sp = _cc_aggregate(a, z, "haruka_cvi_background")
        s_sp = _cc_aggregate(a, s_for_spatial, "haruka_cvi_salient")

    tag = os.path.basename(args.data).removesuffix(".h5ad")
    out_dir = os.path.join(
        args.out_root, tag, f"haruka_cvi_cellcharter_seed{args.seed}"
    )
    save_embeddings(
        out_dir,
        background=z,
        salient=s,
        spatial=z_sp,
        salient_spatial=s_sp,
        obs_names=a.obs_names.to_numpy(),
        method="haruka_cvi_cellcharter",
        seed=args.seed,
        runtime_s=timer.seconds,
        n_latent=args.n_latent,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        batch_key=args.batch_key,
        condition_key=args.condition_key,
        target_label=args.target_label,
        intrinsic_model="ContrastiveVI",
        spatial_encoder="CellCharter aggregate_neighbors",
        n_hops=3,
        spatial_own_block=False,
        salient_controls_zeroed_before_spatial_aggregation=True,
        simvi_components=False,
        supervised_labels=False,
        postfit=False,
    )


if __name__ == "__main__":
    main()
