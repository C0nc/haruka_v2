"""One-stage Haruka with the full shared scVIVA-style niche decoder.

Two niche heads are shared between control and condition cells. Both see
``[z, 0]`` for the background counterfactual and ``[z, s]`` for the observed
condition state. They reconstruct (1) annotated local cell-type composition
and (2) cell-type-specific mean neighbor expression. Together with the
ordinary ContrastiveVI expression decoder, these are the three scVIVA-style
reconstruction objectives. There are no separate background/salient niche
decoders, no reference model, and no post-fit stage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score
from tqdm.auto import tqdm

MERFISH = os.path.join(os.path.dirname(__file__), "..", "bench_merfish")
BE_PROTEIN = os.path.join(os.path.dirname(__file__), "..", "bench_be_protein")
sys.path.insert(0, MERFISH)
sys.path.insert(0, BE_PROTEIN)

from common import Timer, save_embeddings  # noqa: E402
from mender_composition import aggregate_mender_compositions  # noqa: E402
from protein_models import (  # noqa: E402
    attach_batch_aware_protein_mse_decoder,
    attach_protein_mse_decoder,
    prepare_protein_adata,
)
from run_haruka_cvi_scviva import (  # noqa: E402
    augment_with_targets,
    neighbor_index,
)


class SharedNicheDecoder(torch.nn.Module):
    """Composition and niche-expression heads shared by both contrastive arms.

    In composition_gated mode the expression head predicts a profile for
    every neighbor cell type and the decoded composition multiplicatively
    gates those profiles. Niche expression therefore cannot be decoded
    independently of predicted neighborhood composition.
    """

    def __init__(
        self,
        latent_dim: int,
        n_cell_types: int,
        neighbor_embedding_dim: int,
        hidden: int = 256,
        n_layers: int = 2,
        n_batches: int = 0,
        batch_embedding_dim: int = 0,
        neighbor_expression_mode: str = "composition_gated",
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("SharedNicheDecoder requires n_layers >= 1")
        self.n_cell_types = n_cell_types
        self.neighbor_embedding_dim = neighbor_embedding_dim
        if neighbor_expression_mode not in {"independent", "composition_gated"}:
            raise ValueError(
                "neighbor_expression_mode must be independent or composition_gated"
            )
        self.neighbor_expression_mode = neighbor_expression_mode
        self.batch_embedding = None
        input_width = latent_dim
        if batch_embedding_dim > 0:
            if n_batches < 1:
                raise ValueError("A batch-aware niche decoder requires n_batches > 0")
            self.batch_embedding = torch.nn.Embedding(
                int(n_batches), int(batch_embedding_dim)
            )
            input_width += int(batch_embedding_dim)

        def head(output_dim, head_input_width=None):
            layers = []
            width = input_width if head_input_width is None else head_input_width
            for layer_index in range(n_layers):
                layers.append(torch.nn.Linear(width, hidden))
                if layer_index == 0:
                    layers.append(torch.nn.LayerNorm(hidden))
                layers.append(torch.nn.GELU())
                width = hidden
            layers.append(torch.nn.Linear(width, output_dim))
            return torch.nn.Sequential(*layers)

        self.composition = head(n_cell_types)
        expression_input_width = input_width
        if self.neighbor_expression_mode == "composition_gated":
            expression_input_width += n_cell_types
        self.neighbor_mean = head(
            n_cell_types * neighbor_embedding_dim,
            head_input_width=expression_input_width,
        )

    def forward(self, latent, batch_index=None):
        if self.batch_embedding is not None:
            if batch_index is None:
                raise ValueError("Batch-aware niche decoding requires batch_index")
            batch_code = batch_index.reshape(-1).long()
            latent = torch.cat(
                [latent, self.batch_embedding(batch_code)], dim=-1
            )
        composition_logits = self.composition(latent)
        if self.neighbor_expression_mode == "composition_gated":
            composition = F.softmax(composition_logits, dim=-1)
            profiles = self.neighbor_mean(
                torch.cat([latent, composition], dim=-1)
            ).reshape(-1, self.n_cell_types, self.neighbor_embedding_dim)
            neighbor_expression = (
                profiles * composition.unsqueeze(-1)
            ).reshape(latent.shape[0], -1)
        else:
            neighbor_expression = self.neighbor_mean(latent)
        return torch.cat([composition_logits, neighbor_expression], dim=-1)


def annotated_composition(labels, neighbors, categories):
    """scVIVA-style proportions using vectorized, collision-free updates.

    For a fixed neighbor column every focal-cell row occurs exactly once, so
    the ``(row, cell_type)`` destinations are unique. NumPy's much slower
    atomic ``add.at`` is unnecessary; ordinary indexed addition is exact.
    """
    lookup = {label: index for index, label in enumerate(categories)}
    codes = np.asarray([lookup[label] for label in labels], dtype=np.int32)
    n, k = neighbors.shape
    rows = np.arange(n)
    composition = np.zeros((n, len(categories)), dtype=np.float32)
    for neighbor_column in tqdm(
        range(k), desc="Composition target", unit="neighbor", leave=False
    ):
        neighbor_codes = codes[neighbors[:, neighbor_column]]
        composition[rows, neighbor_codes] += 1.0
    composition /= float(k)
    return composition

def expression_embedding(
    adata, n_components, seed, matrix_layer="counts", pretransformed=False
):
    """Spatially unaware assay embedding for the neighbor-mean niche target."""
    counts = adata.layers[matrix_layer]
    if not sp.issparse(counts):
        counts = sp.csr_matrix(np.asarray(counts))
    else:
        counts = counts.tocsr().astype(np.float32)
    counts = counts.astype(np.float32, copy=True)
    if pretransformed:
        normalized = counts
    else:
        library = np.asarray(counts.sum(1)).reshape(-1)
        scale = np.divide(
            1e4,
            library,
            out=np.zeros_like(library, dtype=np.float32),
            where=library > 0,
        )
        normalized = sp.diags(scale).dot(counts).tocsr()
        normalized.data = np.log1p(normalized.data)
    n_components = min(n_components, normalized.shape[1] - 1)
    if n_components < 1:
        raise ValueError("neighbor embedding requires at least two assay features")
    embedding = TruncatedSVD(
        n_components=n_components, random_state=seed
    ).fit_transform(normalized).astype(np.float32)
    embedding -= embedding.mean(0, keepdims=True)
    embedding /= np.maximum(embedding.std(0, keepdims=True), 1e-4)
    return np.clip(embedding, -5.0, 5.0).astype(np.float32)

def cell_type_specific_neighbor_mean(
    embedding, labels, neighbors, categories
):
    """Mean neighbor embedding via vectorized collision-free accumulation."""
    lookup = {label: index for index, label in enumerate(categories)}
    codes = np.asarray([lookup[label] for label in labels], dtype=np.int32)
    n_cells, k = neighbors.shape
    n_types = len(categories)
    n_features = embedding.shape[1]
    sums = np.zeros((n_cells, n_types, n_features), dtype=np.float32)
    counts = np.zeros((n_cells, n_types), dtype=np.float32)
    rows = np.arange(n_cells)
    for neighbor_column in tqdm(
        range(k), desc="Neighbor-mean target", unit="neighbor", leave=False
    ):
        neighbor_rows = neighbors[:, neighbor_column]
        neighbor_codes = codes[neighbor_rows]
        # Rows are unique within each update, so indexed destinations are
        # unique even when many neighbors share a cell type.
        counts[rows, neighbor_codes] += 1.0
        sums[rows, neighbor_codes] += embedding[neighbor_rows]
    np.divide(
        sums,
        counts[..., None],
        out=sums,
        where=counts[..., None] > 0,
    )
    return sums.reshape(n_cells, n_types * n_features)



def counterfactual_shift_membership(shift, target_mask):
    """Encode signed probability shifts without discarding their magnitude.

    For a difference between two probability vectors, total variation is in
    [0, 1]. The output has one no-change channel followed by positive and
    negative channels for every cell type, and sums to one on condition cells.
    """
    shift = np.asarray(shift, dtype=np.float32)
    positive = np.maximum(shift, 0.0)
    negative = np.maximum(-shift, 0.0)
    total_variation = 0.5 * np.abs(shift).sum(1, keepdims=True)
    total_variation = np.clip(total_variation, 0.0, 1.0)
    membership = np.concatenate(
        [1.0 - total_variation, 0.5 * positive, 0.5 * negative], axis=1
    ).astype(np.float32)
    membership[~target_mask] = 0.0
    return membership


def decode_niche(
    decoder, z, s, device, batch_codes=None, batch_size=8192
):
    composition, neighbor_mean = [], []
    decoder.eval()
    with torch.no_grad():
        for left in range(0, len(z), batch_size):
            latent = torch.as_tensor(
                np.concatenate([z[left : left + batch_size], s[left : left + batch_size]], axis=1),
                device=device,
            )
            batch_index = None
            if batch_codes is not None:
                batch_index = torch.as_tensor(
                    batch_codes[left : left + batch_size],
                    dtype=torch.long,
                    device=device,
                ).reshape(-1, 1)
            prediction = decoder(latent, batch_index)
            cut = decoder.n_cell_types
            composition.append(
                F.softmax(prediction[:, :cut], dim=-1).cpu().numpy()
            )
            neighbor_mean.append(prediction[:, cut:].cpu().numpy())
    return (
        np.concatenate(composition).astype(np.float32),
        np.concatenate(neighbor_mean).astype(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train one-stage Haruka with shared counterfactual composition and "
            "neighbor-mean niche decoders."
        )
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_root", default="results_shared_decoder")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument(
        "--n_top_genes", type=int, default=0,
        help="Select this many HVGs from raw counts; 0 keeps every assay feature.",
    )
    parser.add_argument(
        "--hvg_flavor", choices=["seurat_v3"], default="seurat_v3",
        help="Count-aware HVG method used before all Haruka target construction.",
    )
    parser.add_argument("--n_latent", type=int, default=32)
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument(
        "--composition_loss_weight",
        type=float,
        default=1.0,
        help="Relative weight of niche-composition cross entropy.",
    )
    parser.add_argument(
        "--neighbor_mean_loss_weight",
        type=float,
        default=1.0,
        help="Relative weight of niche-expression reconstruction.",
    )
    parser.add_argument(
        "--neighbor_expression_mode",
        choices=["independent", "composition_gated"],
        default="composition_gated",
        help=(
            "composition_gated implements p(niche expression | predicted "
            "composition, z, s) using multiplicatively gated cell-type profiles."
        ),
    )
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--k_niche", type=int, default=20)
    parser.add_argument("--neighbor_embedding_dim", type=int, default=16)
    parser.add_argument(
        "--assay_type",
        choices=["sequencing", "protein"],
        default="sequencing",
        help="Protein mode uses the robust affine transform and exact summed MSE.",
    )
    parser.add_argument(
        "--neighbor_input_mode",
        choices=["normalize", "pretransformed"],
        default="normalize",
        help="Use pretransformed for normalized STARmap or protein matrices.",
    )
    parser.add_argument(
        "--decoder_batch_dim", type=int, default=0,
        help="Learned batch context supplied only to assay and shared niche decoders.",
    )
    parser.add_argument(
        "--spatial_weight",
        type=float,
        default=100.0,
        help=(
            "Weight on annotated-neighborhood composition reconstruction "
            "(five-section benchmark default: 100)."
        ),
    )
    parser.add_argument("--mender_scale", type=int, default=6)
    parser.add_argument("--background_wasserstein_penalty", type=float, default=0.0)
    parser.add_argument(
        "--zs_independence_penalty",
        type=float,
        default=None,
        help=(
            "BioLORD-inspired directional z <- s independence weight. "
            "Defaults to 10 for composition_gated and 25 for independent."
        ),
    )
    parser.add_argument(
        "--symmetric_zs_independence",
        action="store_true",
        help="Apply ridge independence cleanup in both z<-s and s<-z directions.",
    )
    parser.add_argument(
        "--detach_target_z_for_composition",
        action="store_true",
        help=(
            "Stop target composition-loss gradients at z while retaining the shared "
            "D([z,s]) forward decoder; forces the target niche residual into s."
        ),
    )
    parser.add_argument("--mask_salient_mean_on_background", action="store_true")
    parser.add_argument(
        "--direct_only",
        action="store_true",
        help=(
            "Benchmark raw counterfactual decoder compositions without MENDER. "
            "This mode uses only each cell's observed neighborhood composition; "
            "it never constructs paired control/condition targets."
        ),
    )
    parser.add_argument("--mender_radius", type=float, default=15.0)
    parser.add_argument("--mender_neighbors", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--augmentation_mode", choices=["sparse", "dense"], default="sparse",
        help=(
            "Storage used for the augmented assay+niche matrix. Dense gives much "
            "faster random minibatches on large memory nodes; sparse minimizes RAM."
        ),
    )
    parser.add_argument("--batch_key", default="timepoint")
    parser.add_argument(
        "--gene_likelihood",
        choices=["poisson", "nb", "normal"],
        default=None,
        help=(
            "Expression likelihood. When omitted, RNA/sequencing defaults to NB "
            "and protein defaults to normal/MSE. Poisson remains available only "
            "as an explicit legacy override for equidispersed count simulations."
        ),
    )
    parser.add_argument(
        "--batch_injection_mode",
        choices=["all", "salient_encoder", "decoder_only"],
        default="all",
        help=(
            "Where the scVI batch covariate is injected. Keep all for the "
            "paired benchmark default; decoder_only avoids encoding slice ID "
            "in biological z/s representations on multi-slice real data."
        ),
    )
    parser.add_argument("--sample_key", default="slice_id")
    parser.add_argument("--condition_key", default="condition")
    parser.add_argument("--target_label", default="disease")
    parser.add_argument(
        "--shuffled_s_null", type=int, default=0,
        help="If > 0, save this many null composition shifts D([z, s_perm]) - D([z,0]), with s permuted "
        "among condition cells within sample x cell type (z kept), plus the trained model.",
    )
    parser.add_argument(
        "--save_counterfactual_expression", type=float, default=0.0,
        help="If > 0, save decoded expression of all cells from [z,0] (background) "
        "and [z,s] (salient) at this library size, plus the trained model.",
    )
    parser.add_argument(
        "--exclude_label", default=None,
        help="Cells whose condition equals this label serve only as spatial neighbours "
        "for the niche targets; they are excluded from both training arms.",
    )
    parser.add_argument("--cell_type_key", default="observed_celltype")
    args = parser.parse_args()
    if args.gene_likelihood is None:
        args.gene_likelihood = "normal" if args.assay_type == "protein" else "nb"
    if args.zs_independence_penalty is None:
        args.zs_independence_penalty = (
            10.0 if args.neighbor_expression_mode == "composition_gated" else 25.0
        )

    import scvi
    from scvi import REGISTRY_KEYS
    from scvi.external import ContrastiveVI

    raw = ad.read_h5ad(args.data)
    if args.assay_type == "protein":
        adata = prepare_protein_adata(raw)
        adata.layers["counts"] = adata.X.copy()
        assay_feature_count = raw.n_vars
    else:
        adata = raw
        assay_feature_count = adata.n_vars
    batch_key = None if args.batch_key.lower() in {"none", "null", ""} else args.batch_key
    required = {args.sample_key, args.condition_key, args.cell_type_key}
    if batch_key is not None:
        required.add(batch_key)
    missing = required.difference(adata.obs.columns)
    if missing:
        raise KeyError(f"Missing required obs columns: {sorted(missing)}")
    if "counts" not in adata.layers:
        raise KeyError("Haruka requires raw counts in adata.layers['counts']")
    if "spatial" not in adata.obsm:
        raise KeyError("Haruka requires coordinates in adata.obsm['spatial']")
    if args.decoder_batch_dim > 0 and batch_key is None:
        raise ValueError("--decoder_batch_dim requires --batch_key")
    if args.assay_type == "protein" and args.gene_likelihood != "normal":
        raise ValueError("Protein mode requires --gene_likelihood normal")

    input_assay_feature_count = int(adata.n_vars)
    hvg_seconds = 0.0
    if args.n_top_genes < 0:
        raise ValueError("--n_top_genes must be non-negative")
    if args.n_top_genes > 0:
        if args.assay_type != "sequencing":
            raise ValueError("HVG selection is only supported for sequencing assays")
        if args.n_top_genes >= adata.n_vars:
            print(
                f"[Haruka HVG] requested {args.n_top_genes}, but input has only "
                f"{adata.n_vars}; retaining all genes",
                flush=True,
            )
        else:
            import scanpy as sc

            hvg_start = time.perf_counter()
            print(
                f"[Haruka HVG] selecting {args.n_top_genes} pooled "
                f"{args.hvg_flavor} HVGs from {adata.n_vars} genes",
                flush=True,
            )
            sc.pp.highly_variable_genes(
                adata,
                layer="counts",
                n_top_genes=args.n_top_genes,
                flavor=args.hvg_flavor,
                subset=False,
                check_values=False,
            )
            hvg_mask = adata.var["highly_variable"].to_numpy(dtype=bool)
            if int(hvg_mask.sum()) != args.n_top_genes:
                raise RuntimeError(
                    f"Expected {args.n_top_genes} HVGs, got {int(hvg_mask.sum())}"
                )
            adata = adata[:, hvg_mask].copy()
            assay_feature_count = int(adata.n_vars)
            hvg_seconds = time.perf_counter() - hvg_start
            print(f"[Haruka HVG] complete in {hvg_seconds:.1f}s", flush=True)

    target_mask = (
        adata.obs[args.condition_key].astype(str).to_numpy() == args.target_label
    )
    context_only = (
        adata.obs[args.condition_key].astype(str).to_numpy() == args.exclude_label
        if args.exclude_label is not None
        else np.zeros(adata.n_obs, dtype=bool)
    )
    background_idx = np.flatnonzero(~target_mask & ~context_only)
    target_idx = np.flatnonzero(target_mask)
    if not len(background_idx) or not len(target_idx):
        raise ValueError("ContrastiveVI requires non-empty target and background arms")

    preprocessing_start = time.perf_counter()
    print("[Haruka preprocessing] vectorized target construction started", flush=True)
    labels = adata.obs[args.cell_type_key].astype(str).to_numpy()
    categories = np.unique(labels)
    neighbors = neighbor_index(adata, args.k_niche, args.sample_key)
    composition_target = annotated_composition(labels, neighbors, categories)
    neighbor_embedding = expression_embedding(
        adata,
        args.neighbor_embedding_dim,
        args.seed,
        matrix_layer="counts",
        pretransformed=args.neighbor_input_mode == "pretransformed",
    )
    neighbor_mean_target = cell_type_specific_neighbor_mean(
        neighbor_embedding, labels, neighbors, categories
    )
    niche_target = np.concatenate(
        [composition_target, neighbor_mean_target], axis=1
    ).astype(np.float32)
    augmented = augment_with_targets(
        adata,
        niche_target,
        force_dense=args.augmentation_mode == "dense",
        preserve_processed_x=True,
    )
    preprocessing_seconds = time.perf_counter() - preprocessing_start
    print(
        f"[Haruka preprocessing] complete in {preprocessing_seconds:.1f}s",
        flush=True,
    )

    with Timer() as timer:
        scvi.settings.seed = args.seed
        ContrastiveVI.setup_anndata(
            augmented, layer="counts", batch_key=batch_key
        )
        model = ContrastiveVI(
            augmented,
            n_me=niche_target.shape[1],
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            background_wasserstein_penalty=args.background_wasserstein_penalty,
            zs_independence_penalty=args.zs_independence_penalty,
            symmetric_zs_independence=args.symmetric_zs_independence,
            detach_target_z_for_composition=args.detach_target_z_for_composition,
            batch_injection_mode=args.batch_injection_mode,
            mask_salient_mean_on_background=args.mask_salient_mean_on_background,
            n_background_latent=args.n_latent,
            n_salient_latent=args.n_latent,
            use_observed_lib_size=True,
            wasserstein_penalty=0.2,
            gene_likelihood=args.gene_likelihood,
            me_weight=args.spatial_weight,
            me_reduction="mean",
            n_spatial_latent=0,
            niche_context=False,
        )
        n_batches = (
            int(augmented.obs[batch_key].astype("category").cat.categories.size)
            if batch_key is not None
            else 0
        )
        if args.assay_type == "protein":
            if args.decoder_batch_dim > 0:
                attach_batch_aware_protein_mse_decoder(
                    model,
                    2 * args.n_latent,
                    assay_feature_count,
                    n_batches=n_batches,
                    batch_embedding_dim=args.decoder_batch_dim,
                )
            else:
                attach_protein_mse_decoder(
                    model, 2 * args.n_latent, assay_feature_count
                )
        model.module.decoder_me = SharedNicheDecoder(
            2 * args.n_latent,
            len(categories),
            neighbor_embedding.shape[1],
            args.n_hidden,
            args.decoder_layers,
            n_batches=n_batches,
            batch_embedding_dim=args.decoder_batch_dim,
            neighbor_expression_mode=args.neighbor_expression_mode,
        )

        def niche_loss(x, index, prediction, reduction="mean"):
            truth = x[..., index:]
            n_types = len(categories)
            embedding_dim = neighbor_embedding.shape[1]
            truth_composition = truth[..., :n_types]
            predicted_composition = prediction[..., :n_types]
            composition_loss = -(
                truth_composition
                * F.log_softmax(predicted_composition, dim=-1)
            ).sum(-1)

            truth_mean = truth[..., n_types:].reshape(
                *truth.shape[:-1], n_types, embedding_dim
            )
            predicted_mean = prediction[..., n_types:].reshape(
                *prediction.shape[:-1], n_types, embedding_dim
            )
            if args.neighbor_expression_mode == "composition_gated":
                truth_contribution = (
                    truth_mean * truth_composition.unsqueeze(-1)
                )
                per_type_mse = (
                    predicted_mean - truth_contribution
                ).square().mean(-1)
                contribution_loss = per_type_mse.sum(-1)
                aggregate_loss = (
                    predicted_mean.sum(-2) - truth_contribution.sum(-2)
                ).square().mean(-1)
                neighbor_mean_loss = contribution_loss + aggregate_loss
            else:
                present = (truth_composition > 0).to(prediction.dtype)
                per_type_mse = (predicted_mean - truth_mean).square().mean(-1)
                neighbor_mean_loss = (
                    (per_type_mse * present).sum(-1)
                    / present.sum(-1).clamp_min(1.0)
                )
            return (
                args.composition_loss_weight * composition_loss
                + args.neighbor_mean_loss_weight * neighbor_mean_loss
            )

        model.module.me_reconstruction_loss = niche_loss
        model.train(
            background_indices=background_idx,
            target_indices=target_idx,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            train_size=1.0,
            early_stopping=False,
            gradient_clip_val=1.0,
            datasplitter_kwargs={
                "num_workers": args.num_workers,
                "persistent_workers": args.num_workers > 0,
            },
        )
        z = model.get_latent_representation(
            augmented,
            representation_kind="background",
            batch_size=args.batch_size,
        ).astype(np.float32)
        s = model.get_latent_representation(
            augmented,
            representation_kind="salient",
            batch_size=args.batch_size,
        ).astype(np.float32)

        device = next(model.module.parameters()).device
        batch_codes = None
        if args.decoder_batch_dim > 0:
            batch_codes = np.asarray(
                model.adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
            ).reshape(-1).astype(np.int64)
        zero_s = np.zeros_like(s)
        background_composition, background_neighbor_mean = decode_niche(
            model.module.decoder_me,
            z,
            zero_s,
            device,
            batch_codes=batch_codes,
        )
        condition_composition, condition_neighbor_mean = decode_niche(
            model.module.decoder_me,
            z,
            s,
            device,
            batch_codes=batch_codes,
        )
        composition_shift = condition_composition - background_composition
        composition_shift[~target_mask] = 0.0
        neighbor_mean_shift = condition_neighbor_mean - background_neighbor_mean
        neighbor_mean_shift[~target_mask] = 0.0
        salient_membership = counterfactual_shift_membership(
            composition_shift, target_mask
        )

        if args.direct_only:
            spatial = np.concatenate(
                [background_composition, background_neighbor_mean], axis=1
            )
            salient_spatial = np.concatenate(
                [composition_shift, neighbor_mean_shift], axis=1
            )
            radius_diagnostics = {
                "used": False,
                "reason": "direct decoder benchmark requested",
            }
        else:
            spatial, salient_spatial, radius_diagnostics = aggregate_mender_compositions(
                augmented.obsm["spatial"],
                augmented.obs[args.sample_key].astype(str).to_numpy(),
                background_composition,
                salient_membership,
                scale=args.mender_scale,
                radius=args.mender_radius,
                radius_neighbors=args.mender_neighbors,
                return_diagnostics=True,
            )

    tag = os.path.basename(args.data).removesuffix(".h5ad")
    run_name = (
        f"haruka_shared_full_niche_seed{args.seed}"
        if args.direct_only
        else f"haruka_shared_full_niche_mender_seed{args.seed}"
    )
    if args.neighbor_expression_mode == "composition_gated":
        run_name = run_name.replace(
            f"_seed{args.seed}",
            f"_compgated_seed{args.seed}",
        )
    if args.decoder_batch_dim > 0:
        run_name = run_name.replace(
            f"_seed{args.seed}",
            f"_decoderbatch{args.decoder_batch_dim}_seed{args.seed}",
        )
    if args.batch_injection_mode != "all":
        run_name = run_name.replace(
            f"_seed{args.seed}",
            f"_batchmode-{args.batch_injection_mode}_seed{args.seed}",
        )
    out_dir = os.path.join(args.out_root, tag, run_name)
    save_embeddings(
        out_dir,
        background=z,
        salient=s,
        spatial=spatial,
        salient_spatial=salient_spatial,
        obs_names=augmented.obs_names.to_numpy(),
        method=(
            "haruka_symmetric_full_niche"
            if args.direct_only
            else "haruka_shared_counterfactual_spatial"
        ),
        seed=args.seed,
        runtime_s=timer.seconds,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        n_latent=args.n_latent,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        decoder_layers=args.decoder_layers,
        k_niche=args.k_niche,
        neighbor_embedding_dim=neighbor_embedding.shape[1],
        composition_loss_weight=args.composition_loss_weight,
        neighbor_mean_loss_weight=args.neighbor_mean_loss_weight,
        neighbor_expression_mode=args.neighbor_expression_mode,
        spatial_weight=args.spatial_weight,
        background_wasserstein_penalty=args.background_wasserstein_penalty,
        zs_independence_penalty=args.zs_independence_penalty,
        symmetric_zs_independence=args.symmetric_zs_independence,
        detach_target_z_for_composition=args.detach_target_z_for_composition,
        mask_salient_mean_on_background=args.mask_salient_mean_on_background,
        mender_scale=args.mender_scale,
        mender_radius=args.mender_radius,
        mender_neighbors=args.mender_neighbors,
        cell_type_key=args.cell_type_key,
        batch_key=batch_key,
        batch_injection_mode=args.batch_injection_mode,
        assay_type=args.assay_type,
        assay_reconstruction=(
            "exact summed MSE"
            if args.assay_type == "protein"
            else f"sequencing {args.gene_likelihood} likelihood"
        ),
        decoder_batch_injection=(args.decoder_batch_dim > 0),
        decoder_batch_embedding_dim=args.decoder_batch_dim,
        neighbor_input_mode=args.neighbor_input_mode,
        augmentation_mode=args.augmentation_mode,
        neighbor_target_implementation="vectorized indexed accumulation v1",
        preprocessing_seconds=preprocessing_seconds,
        input_assay_feature_count=input_assay_feature_count,
        selected_assay_feature_count=int(adata.n_vars),
        n_top_genes=args.n_top_genes,
        hvg_flavor=args.hvg_flavor if args.n_top_genes > 0 else None,
        hvg_batch_key=None,
        hvg_seconds=hvg_seconds,
        preserved_input_x=True,
        gene_likelihood=args.gene_likelihood,
        cell_type_categories=categories.tolist(),
        spatial_decoder=(
            "two task heads shared across arms: control D_niche([z,0]), "
            "condition D_niche([z,s]); no background/salient dual decoders"
        ),
        spatial_target=(
            "annotated kNN cell-type composition plus cell-type-specific "
            "mean neighbor expression embedding"
        ),
        niche_expression_dependency=(
            "predicted composition multiplicatively gates cell-type-specific "
            "expression profiles; target is the composition-weighted profile "
            "and loss also reconstructs aggregate niche expression"
            if args.neighbor_expression_mode == "composition_gated"
            else "independent composition and neighbor-mean decoder heads"
        ),
        paired_target_construction=False,
        target_definition="each cell's own observed neighborhood composition",
        salient_spatial_definition=(
            "[composition and neighbor-mean] D_niche([z,s]) - "
            "D_niche([z,0])"
        ),
        salient_readout=(
            "concatenated raw composition and neighbor-mean counterfactual shifts"
            if args.direct_only
            else (
                "MENDER on [no-change, positive shifts, negative shifts] membership"
            )
        ),
        one_stage=True,
        reference_pretraining=False,
        supervised_domain_labels=False,
        supervised_cell_type_niche_target=True,
        full_scviva_three_reconstruction_objectives=True,
        postfit=False,
        direct_decoder_readout=args.direct_only,
    )
    if args.n_top_genes > 0:
        np.savetxt(os.path.join(out_dir, "hvg_genes.txt"), adata.var_names, fmt="%s")
    np.save(
        os.path.join(out_dir, "background_composition.npy"),
        background_composition,
    )
    np.save(
        os.path.join(out_dir, "background_neighbor_mean.npy"),
        background_neighbor_mean,
    )
    np.save(
        os.path.join(out_dir, "condition_neighbor_mean.npy"),
        condition_neighbor_mean,
    )
    np.save(
        os.path.join(out_dir, "salient_neighbor_mean_shift.npy"),
        neighbor_mean_shift,
    )
    np.save(
        os.path.join(out_dir, "condition_composition.npy"),
        condition_composition,
    )
    np.save(
        os.path.join(out_dir, "salient_composition_shift.npy"),
        composition_shift,
    )
    np.save(
        os.path.join(out_dir, "salient_shift_membership.npy"),
        salient_membership,
    )
    with open(os.path.join(out_dir, "mender_radius_diagnostics.json"), "w") as handle:
        json.dump(radius_diagnostics, handle, indent=2)
    if args.save_counterfactual_expression > 0:
        # Decode directly with the trained decoder: px_scale is the per-cell normalised expression
        # (sums to 1 over assay genes); x library size -> counterfactual expression.
        # background = D([z,0]) for every cell; salient = D([z,s]) (meaningful for condition cells).
        module = model.module.eval(); device = next(module.parameters()).device
        codes = batch_codes if batch_codes is not None else np.zeros(len(z), dtype=np.int64)
        outs = {"background": [], "salient": []}
        with torch.no_grad():
            for left in range(0, len(z), args.batch_size):
                zz = torch.as_tensor(z[left:left + args.batch_size], device=device)
                ss = torch.as_tensor(s[left:left + args.batch_size], device=device)
                bi = torch.as_tensor(codes[left:left + args.batch_size], device=device).reshape(-1, 1)
                lib = torch.zeros(len(zz), 1, device=device)
                for kind, sv in (("background", torch.zeros_like(ss)), ("salient", ss)):
                    scale = module._generic_generative(z=zz, s=sv, library=lib, batch_index=bi)["px_scale"]
                    outs[kind].append((scale[:, :assay_feature_count] * args.save_counterfactual_expression).cpu().numpy())
        for kind, parts in outs.items():
            np.save(os.path.join(out_dir, f"cf_expression_{kind}.npy"), np.concatenate(parts).astype(np.float32))
        np.save(os.path.join(out_dir, "cf_expression_condition_mask.npy"), target_mask)
        print(f"[Haruka] counterfactual expression saved: {np.concatenate(outs['salient']).shape} at library size "
              f"{args.save_counterfactual_expression}", flush=True)
        model.save(os.path.join(out_dir, "model"), overwrite=True)
    if args.shuffled_s_null > 0:
        # Null for "does the baseline z predict the treatment effect": keep each cell's z, give it the s of
        # another condition cell of the same sample x cell type, decode. Any z -> shift dependence that
        # survives here comes from the decoder's z x s interaction, not from which s the cell actually has.
        import pandas as pd
        rng = np.random.default_rng(args.seed)
        smp = augmented.obs[args.sample_key].astype(str).to_numpy()
        cty = augmented.obs[args.cell_type_key].astype(str).to_numpy()
        groups = pd.Series(np.arange(len(z))[target_mask]).groupby(
            [smp[target_mask], cty[target_mask]]).apply(lambda v: v.to_numpy())
        for k in range(args.shuffled_s_null):
            s_perm = s.copy()
            for idx in groups:
                s_perm[idx] = s[rng.permutation(idx)]
            perm_comp, _ = decode_niche(model.module.decoder_me, z, s_perm, device, batch_codes=batch_codes)
            null_shift = perm_comp - background_composition
            null_shift[~target_mask] = 0.0
            np.save(os.path.join(out_dir, f"null_shuffled_s_composition_shift_{k}.npy"), null_shift.astype(np.float32))
        print(f"[Haruka] saved {args.shuffled_s_null} shuffled-s null composition shifts", flush=True)
        if not os.path.isdir(os.path.join(out_dir, "model")):
            model.save(os.path.join(out_dir, "model"), overwrite=True)


if __name__ == "__main__":
    main()
