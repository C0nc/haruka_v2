"""Train latest one-stage unsupervised four-slot Haruka on a prepared study."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "bench_merfish", ROOT / "bench_spacseq", ROOT / "src"):
    sys.path.insert(0, str(path))

from common import Timer, save_embeddings  # noqa: E402
from haruka_spatial import HarukaSpatial, build_neighbor_index  # noqa: E402


SLOTS = ("background", "spatial", "salient", "salient_spatial")


def _ridge_map(source: np.ndarray, target: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    gram = source.T @ source
    gram.flat[:: gram.shape[0] + 1] += alpha
    return np.linalg.solve(gram, source.T @ target)


def _extract(model, adata: ad.AnnData) -> dict[str, np.ndarray]:
    return {
        slot: model.get_latent_representation(adata, representation_kind=slot, batch_size=2048).astype(np.float32)
        for slot in SLOTS
    }


def _normalize(data: ad.AnnData) -> None:
    if "counts" not in data.layers:
        data.layers["counts"] = data.X.copy()
    # The registered layer is the model input; cov_gene_num=0 leaves X unused.
    data.X = data.layers["counts"].copy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--study", required=True)
    p.add_argument("--out-root", default="bench_real_discovery/results")
    p.add_argument("--tag", default="one_stage_unsup_large")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--k-spatial", type=int, default=8)
    p.add_argument("--holdout-graph", default=None)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--salient-dim", type=int, default=16)
    p.add_argument("--spatial-dim", type=int, default=16)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--save-model", action="store_true")
    args = p.parse_args()

    full = ad.read_h5ad(args.data)
    if args.holdout_graph:
        holdout = full.obs.graph_id.astype(str).to_numpy() == args.holdout_graph
        if not holdout.any():
            raise ValueError(f"holdout graph {args.holdout_graph!r} is absent")
        train = full[~holdout].copy()
        query = full[holdout].copy()
    else:
        train, query = full.copy(), None
    _normalize(train)
    if query is not None:
        _normalize(query)

    if set(train.obs.condition.astype(str)) != {"control", "disease"}:
        raise ValueError("training data must contain both control and disease")
    out = Path(args.out_root) / args.study / f"haruka_{args.tag}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    h = HarukaSpatial(
        train, cov_gene_num=0, seed=args.seed,
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
        gene_likelihood="normal" if args.study == "ad" else "poisson",
        me_weight=0.0,
        me_reduction="sum",
        reg_to_use="mi",
        lam_mi=5.0,
        kl_spatial_weight=0.01,
        spatial_encoder="gat",
        condition_key="condition",
        condition_label="disease",
        batch_key=None,
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

    with Timer() as timer:
        h.model.module.background_only_training = False
        h.model.module.mae_epochs = 0
        print(
            f"[Haruka {args.study}] one-stage unsupervised; no label supervision; "
            f"no removable batch; epochs={args.epochs}; holdout={args.holdout_graph}",
            flush=True,
        )
        h.train(max_epochs=args.epochs, batch_size=args.batch_size, plan_kwargs={"lr": args.lr})

    # Extract own-cell slots. The same label-free ridge cleanup used by the latest
    # aging run is fit on training cells and transferred unchanged to a held-out graph.
    # HarukaSpatial owns the registered graph-augmented AnnData.
    train_registered = h.model.adata
    train_rep = _extract(h.model, train_registered)
    bg_map = _ridge_map(train_rep["spatial"], train_rep["background"])
    ss_map = _ridge_map(train_rep["spatial"], train_rep["salient_spatial"])
    train_rep["background"] -= train_rep["spatial"] @ bg_map
    train_rep["salient_spatial"] -= train_rep["spatial"] @ ss_map

    if query is not None:
        build_neighbor_index(query, k=args.k_spatial, library_key="graph_id")
        registered_query = h.model._validate_anndata(query)
        h.model.attach_neighbor_bank(registered_query)
        query_rep = _extract(h.model, registered_query)
        query_rep["background"] -= query_rep["spatial"] @ bg_map
        query_rep["salient_spatial"] -= query_rep["spatial"] @ ss_map
        names = np.concatenate([train.obs_names.to_numpy(), query.obs_names.to_numpy()])
        reps = {slot: np.concatenate([train_rep[slot], query_rep[slot]]) for slot in SLOTS}
    else:
        names, reps = train.obs_names.to_numpy(), train_rep

    save_embeddings(
        str(out),
        background=reps["background"], spatial=reps["spatial"],
        salient=reps["salient"], salient_spatial=reps["salient_spatial"],
        obs_names=names,
        method=f"haruka_{args.study}_one_stage_unsupervised",
        runtime_s=timer.seconds, seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr,
        training_condition="control versus pooled target",
        expression_contract=("preprocessed continuous expression with normal likelihood"
                             if args.study == "ad" else "raw counts with Poisson likelihood"),
        stage_supervision=False, celltype_supervision=False, plaque_supervision=False,
        graph_library_key="graph_id", batch_key=None,
        encoder_batch_injection=False, decoder_batch_injection=False,
        training_mode="one_stage", control_only_pretraining=False,
        perturbation_supervision=False, paired_slot_orthogonalization=True,
        orthogonalization_fit="training cells only",
        holdout_graph=args.holdout_graph,
        holdout_transductive=False if query is not None else None,
        latent_dim=args.latent_dim, salient_latent_dim=args.salient_dim,
        n_spatial_latent=args.spatial_dim, n_hidden=args.hidden,
        model_n_layers=args.layers, n_attention_heads=args.heads,
        k_spatial=args.k_spatial, n_layer=0,
    )
    if args.save_model:
        h.model.attach_neighbor_bank(train_registered)
        h.model.save(str(out / "model"), overwrite=True, save_anndata=False)


if __name__ == "__main__":
    main()
