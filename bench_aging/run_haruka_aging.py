"""Train latest one-stage unsupervised four-slot Haruka on MERFISH aging."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "bench_merfish", ROOT / "bench_spacseq", ROOT / "src"):
    sys.path.insert(0, str(path))

from batch_invariance import install_four_slot_batch_mmd  # noqa: E402
from common import Timer, save_embeddings  # noqa: E402
from run_haruka import extract_all_cells  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="bench_aging/data/merfish_aging_100k.h5ad")
    p.add_argument("--out-root", default="bench_aging/results")
    p.add_argument("--tag", default="one_stage_unsup_large")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--k-spatial", type=int, default=8)
    p.add_argument("--batch-mmd-weight", type=float, default=10.0)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--salient-dim", type=int, default=16)
    p.add_argument("--spatial-dim", type=int, default=16)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=0)
    p.add_argument("--save-model", action="store_true")
    args = p.parse_args()

    from haruka_spatial import HarukaSpatial

    base = ad.read_h5ad(args.data)
    required = {"condition", "age_group", "slice_id", "graph_id", "technical_batch"}
    missing = sorted(required - set(base.obs))
    if missing:
        raise KeyError(f"missing aging columns: {missing}")
    table = pd.crosstab(base.obs.technical_batch, base.obs.condition)
    if (table > 0).sum(axis=1).min() < 2:
        raise ValueError(f"technical batch is condition-confounded:\n{table}")
    if "counts" not in base.layers:
        base.layers["counts"] = base.X.copy()
    base.layers["counts"] = np.rint(base.layers["counts"]).astype(np.int32)
    base.X = base.layers["counts"].copy()
    sc.pp.normalize_total(base, target_sum=1e4)
    sc.pp.log1p(base)

    out = Path(args.out_root) / Path(args.data).stem / f"haruka_{args.tag}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    h = HarukaSpatial(
        base, cov_gene_num=0, seed=args.seed,
        k_spatial=args.k_spatial, library_key="graph_id",
    )
    h.prepare_spatial_graph()
    h.setup_model(
        seed=args.seed,
        latent_dim=args.latent_dim,
        salient_latent_dim=args.salient_dim,
        n_spatial_latent=args.spatial_dim,
        n_hidden=args.hidden,
        n_layers=args.layers,
        n_attention_heads=args.heads,
        spatial_attach="both",
        gene_likelihood="poisson",
        me_weight=0.0,
        me_reduction="sum",
        reg_to_use="mi",
        lam_mi=5.0,
        kl_spatial_weight=0.01,
        spatial_encoder="gat",
        condition_key="condition",
        condition_label="disease",
        # Only A/B is removable. Mouse is the graph library and is never injected.
        batch_key="technical_batch",
        mae_epochs=0,
        wasserstein_penalty=0.0,
        spatial_invariance_penalty=0.0,
        background_mmd_penalty=0.0,
        mask_salient_mean_on_background=False,
        paired_slot_reg_weight=0.0,
        background_representation="private",
        salient_within_penalty=0.0,
        spatial_pair_penalty=1.0,
        salient_prior_components=1,
        salient_assignment_penalty=0.0,
        salient_assignment_temperature=0.5,
    )
    install_four_slot_batch_mmd(h.model.module, args.batch_mmd_weight)

    with Timer() as timer:
        h.model.module.background_only_training = False
        h.model.module.mae_epochs = 0
        print(
            f"[Haruka aging] one-stage unsupervised; early=control; "
            f"middle+late=target; middle/late hidden; epochs={args.epochs}",
            flush=True,
        )
        h.train(
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            plan_kwargs={"lr": args.lr},
        )

    reps = extract_all_cells(h, n_layer=args.n_layer)
    save_embeddings(
        str(out),
        background=reps["background"],
        spatial=reps["spatial"],
        salient=reps["salient"],
        salient_spatial=reps["salient_spatial"],
        obs_names=base.obs_names.to_numpy(),
        method="haruka_aging_one_stage_unsupervised",
        runtime_s=timer.seconds,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        training_condition="early control versus pooled middle+late target",
        middle_late_supervision=False,
        graph_library_key="mouse_id",
        batch_key="technical_batch",
        exact_age_mouse_confounding=True,
        encoder_batch_injection=True,
        neighbor_batch_injection=True,
        decoder_batch_injection=True,
        four_slot_batch_mmd_weight=args.batch_mmd_weight,
        training_mode="one_stage",
        perturbation_supervision=False,
        control_only_pretraining=False,
        paired_slot_orthogonalization=True,
        latent_dim=args.latent_dim,
        salient_latent_dim=args.salient_dim,
        n_spatial_latent=args.spatial_dim,
        n_hidden=args.hidden,
        model_n_layers=args.layers,
        n_attention_heads=args.heads,
        k_spatial=args.k_spatial,
        n_layer=args.n_layer,
    )
    if args.save_model:
        h.model.save(str(out / "model"), overwrite=True, save_anndata=False)


if __name__ == "__main__":
    main()
