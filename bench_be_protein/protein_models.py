"""Protein-specific preprocessing and exact-MSE decoders for BE CODEX."""

from __future__ import annotations

import types

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch


PROTEIN_LAYER = "protein_mse"


def prepare_protein_adata(adata: ad.AnnData, clip: float = 5.0) -> ad.AnnData:
    """Put robust nonnegative standardized intensities in the model layer.

    The source CODEX matrix is already feature-wise z-scored. ContrastiveVI's
    generic generative path clamps decoder means positive, so we apply a common
    affine shift after symmetric clipping. The transform preserves ordering and
    differences inside the robust range and is shared by every benchmark method.
    """
    out = adata.copy()
    x = out.layers["protein"] if "protein" in out.layers else out.X
    x = x.toarray() if sp.issparse(x) else np.asarray(x)
    transformed = (np.clip(x.astype(np.float32), -clip, clip) + clip) / clip
    out.X = transformed.copy()
    out.layers[PROTEIN_LAYER] = transformed.copy()
    return out


class ProteinMSEDecoder(torch.nn.Module):
    """Unconstrained-amplitude protein decoder with a positive shifted mean."""

    def __init__(self, latent_dim: int, n_proteins: int, hidden: int = 256):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, n_proteins),
            torch.nn.Softplus(),
        )

    def forward(self, dispersion, latent, library, batch_index):
        mean = self.network(latent)
        ones = torch.ones_like(mean)
        zeros = torch.zeros_like(mean)
        return mean, ones, mean, zeros


class BatchAwareProteinMSEDecoder(torch.nn.Module):
    """Continuous-protein decoder with an explicit learned batch embedding."""

    def __init__(
        self,
        latent_dim: int,
        n_proteins: int,
        n_batches: int,
        batch_embedding_dim: int = 8,
        hidden: int = 256,
    ):
        super().__init__()
        if n_batches < 1:
            raise ValueError("n_batches must be positive")
        self.n_batches = int(n_batches)
        self.batch_embedding = torch.nn.Embedding(
            self.n_batches, int(batch_embedding_dim)
        )
        self.network = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + int(batch_embedding_dim), hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, n_proteins),
            torch.nn.Softplus(),
        )

    def forward(self, dispersion, latent, library, batch_index):
        if batch_index is None:
            batch_code = torch.zeros(
                latent.shape[0], dtype=torch.long, device=latent.device
            )
        else:
            batch_code = batch_index.reshape(-1).long()
        if torch.any(batch_code < 0) or torch.any(batch_code >= self.n_batches):
            raise ValueError("Decoder received a batch index outside its vocabulary")
        context = self.batch_embedding(batch_code)
        mean = self.network(torch.cat([latent, context], dim=-1))
        ones = torch.ones_like(mean)
        zeros = torch.zeros_like(mean)
        return mean, ones, mean, zeros


def protein_mse_reconstruction(
    x, index, px_rate, px_r, px_dropout, px_scale, gene_likelihood
):
    """Exact summed squared error per cell; no count likelihood terms."""
    return torch.square(px_rate - x[..., :index]).sum(dim=-1)


def attach_protein_mse_decoder(model, latent_dim: int, n_proteins: int) -> None:
    """Replace ContrastiveVI's sequencing decoder and reconstruction objective."""
    model.module.decoder = ProteinMSEDecoder(latent_dim, n_proteins)
    # Instance assignment intentionally avoids descriptor binding: loss calls this
    # attribute with the seven tensors/arguments shown above.
    model.module.reconstruction_loss = protein_mse_reconstruction
    model.module.gene_likelihood = "protein_mse"


def attach_batch_aware_protein_mse_decoder(
    model,
    latent_dim: int,
    n_proteins: int,
    n_batches: int,
    batch_embedding_dim: int = 8,
) -> None:
    """Attach the protein-MSE decoder with decoder-only batch context."""
    model.module.decoder = BatchAwareProteinMSEDecoder(
        latent_dim=latent_dim,
        n_proteins=n_proteins,
        n_batches=n_batches,
        batch_embedding_dim=batch_embedding_dim,
    )
    model.module.reconstruction_loss = protein_mse_reconstruction
    model.module.gene_likelihood = "protein_mse"


def protein_transform_description(clip: float = 5.0) -> str:
    return f"source feature-z-score clipped to [-{clip:g},{clip:g}], shifted/scaled to [0,2]"
