"""Protein-aware PCA, CellCharter/AE, and ContrastiveVI baselines."""

from __future__ import annotations

import argparse
import os
import sys

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(__file__)
MERFISH = os.path.join(HERE, "..", "bench_merfish")
sys.path.insert(0, MERFISH)
sys.path.insert(0, HERE)

from common import Timer, save_embeddings  # noqa: E402
from protein_models import (  # noqa: E402
    PROTEIN_LAYER,
    attach_protein_mse_decoder,
    prepare_protein_adata,
    protein_transform_description,
)
from run_baselines_2x2 import _cc_aggregate  # noqa: E402


N_LATENT = 32


class ProteinAE(torch.nn.Module):
    def __init__(self, n_input, n_latent=N_LATENT, hidden=256):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_input, hidden), torch.nn.LayerNorm(hidden),
            torch.nn.GELU(), torch.nn.Linear(hidden, n_latent),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_latent, hidden), torch.nn.LayerNorm(hidden),
            torch.nn.GELU(), torch.nn.Linear(hidden, n_input), torch.nn.Softplus(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def fit_protein_ae(x, seed, max_epochs, batch_size):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProteinAE(x.shape[1]).to(device)
    dataset = TensorDataset(torch.as_tensor(x, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(max_epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            _, prediction = model(batch)
            loss = torch.square(prediction - batch).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    result = []
    with torch.no_grad():
        for (batch,) in DataLoader(dataset, batch_size=8192):
            result.append(model.encoder(batch.to(device)).cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def run_pca(a, out_dir, seed, **kwargs):
    x = np.asarray(a.layers[PROTEIN_LAYER], dtype=np.float32)
    with Timer() as timer:
        spatial = _cc_aggregate(a, x, "protein_pca")
    save_embeddings(
        out_dir, background=x, salient=x, spatial=spatial,
        salient_spatial=spatial, obs_names=a.obs_names.to_numpy(),
        method="pca_cellcharter_protein", runtime_s=timer.seconds,
        likelihood="not applicable", protein_transform=protein_transform_description(),
        note="one protein representation reused in both arms; CellCharter spatial aggregation",
    )


def run_cellcharter(a, out_dir, seed, max_epochs, batch_size, **kwargs):
    x = np.asarray(a.layers[PROTEIN_LAYER], dtype=np.float32)
    with Timer() as timer:
        z = fit_protein_ae(x, seed, max_epochs, batch_size)
        spatial = _cc_aggregate(a, z, "protein_ae")
    save_embeddings(
        out_dir, background=z, salient=z, spatial=spatial,
        salient_spatial=spatial, obs_names=a.obs_names.to_numpy(),
        method="cellcharter_protein_mse", runtime_s=timer.seconds,
        likelihood="exact summed MSE", protein_transform=protein_transform_description(),
        note="protein MSE autoencoder latent plus CellCharter spatial aggregation",
    )


def run_contrastivevi(a, out_dir, seed, max_epochs, batch_size, **kwargs):
    import scvi
    from scvi.external import ContrastiveVI

    target = a.obs["condition"].astype(str).to_numpy() == "disease"
    with Timer() as timer:
        scvi.settings.seed = seed
        ContrastiveVI.setup_anndata(a, layer=PROTEIN_LAYER, batch_key="section")
        model = ContrastiveVI(
            a, n_me=0, n_hidden=256, n_background_latent=N_LATENT,
            n_salient_latent=N_LATENT, use_observed_lib_size=True,
            wasserstein_penalty=0.2, gene_likelihood="normal",
        )
        attach_protein_mse_decoder(model, 2 * N_LATENT, a.n_vars)
        model.train(
            background_indices=np.flatnonzero(~target),
            target_indices=np.flatnonzero(target),
            max_epochs=max_epochs, batch_size=batch_size, train_size=1.0,
            early_stopping=False, gradient_clip_val=1.0,
        )
        z = model.get_latent_representation(
            a, representation_kind="background"
        ).astype(np.float32)
        s = model.get_latent_representation(
            a, representation_kind="salient"
        ).astype(np.float32)
        s_control_zero = s.copy()
        s_control_zero[~target] = 0.0
        z_spatial = _cc_aggregate(a, z, "protein_cvi_bg")
        s_spatial = _cc_aggregate(a, s_control_zero, "protein_cvi_sal")
    save_embeddings(
        out_dir, background=z, salient=s, spatial=z_spatial,
        salient_spatial=s_spatial, obs_names=a.obs_names.to_numpy(),
        method="contrastivevi_cellcharter_protein_mse", runtime_s=timer.seconds,
        likelihood="exact summed MSE", protein_transform=protein_transform_description(),
        note="protein-MSE ContrastiveVI plus CellCharter aggregation of each latent",
    )


RUNNERS = {
    "pca": run_pca,
    "cellcharter": run_cellcharter,
    "contrastivevi": run_contrastivevi,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_root", default="bench_be_protein/results")
    parser.add_argument("--methods", nargs="+", default=list(RUNNERS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2048)
    args = parser.parse_args()

    a = prepare_protein_adata(ad.read_h5ad(args.data))
    tag = os.path.basename(args.data).removesuffix(".h5ad")
    for method in args.methods:
        print(f"[baseline] {method} on {tag}", flush=True)
        out_dir = os.path.join(
            args.out_root, tag, f"{method}_protein_seed{args.seed}"
        )
        RUNNERS[method](
            a, out_dir, args.seed,
            max_epochs=args.max_epochs, batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
