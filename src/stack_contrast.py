"""Chunked axial-attention contrastive model.

A cell chunk is the unit of computation. Every cell in a chunk is equal -- there is no
centre cell, no neighbour gather and no pooling. Cells see each other because they sit in
the same chunk, and the chunk is built from spatially contiguous cells, so chunk membership
*is* the niche definition.

Two independent towers produce one embedding per cell::

    z = background_tower(chunk)      shared structure
    s = salient_tower(chunk)         condition-specific, forced to 0 on control cells
    latent = z + s                   control: z,  perturbed: z + s

Embeddings are deterministic. There is no reparameterisation and no KL; the marginal is
shaped by a sliced-Wasserstein term against N(0, I), following Arc's stack.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "AxialBlock",
    "AxialTower",
    "StackContrast",
    "sliced_wasserstein",
    "spatial_chunks",
    "ChunkLoader",
]


def sliced_wasserstein(x: torch.Tensor, n_proj: int = 64) -> torch.Tensor:
    """Sliced 2-Wasserstein between ``x`` and standard normal samples of the same shape.

    ``x`` is ``(B, N, d)``. Projections are sorted along the cell axis, so this matches the
    empirical distribution *within each chunk* -- the same convention stack uses.
    """
    b, n, d = x.shape
    proj = torch.randn(n_proj, d, device=x.device, dtype=x.dtype)
    proj = proj / proj.norm(dim=1, keepdim=True).clamp(min=1e-8)
    xs = (x @ proj.t()).sort(dim=1).values
    ys = (torch.randn_like(x) @ proj.t()).sort(dim=1).values
    return ((xs - ys) ** 2).mean()


class AxialBlock(nn.Module):
    """One axial layer: attention over gene tokens, then over cells, then a token-wise MLP."""

    def __init__(
        self,
        n_tokens: int,
        token_dim: int,
        n_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        wide = n_tokens * token_dim
        self.n_tokens, self.token_dim = n_tokens, token_dim
        self.norm_tok = nn.LayerNorm(token_dim)
        self.attn_tok = nn.MultiheadAttention(
            token_dim, _fit_heads(token_dim, n_heads), dropout=dropout, batch_first=True
        )
        self.norm_cell = nn.LayerNorm(wide)
        self.attn_cell = nn.MultiheadAttention(
            wide, _fit_heads(wide, n_heads), dropout=dropout, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * mlp_ratio, token_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        dist_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x (B, N, T, D) | pad_mask (B, N) True where the slot is padding
        b, n, t, d = x.shape

        h = self.norm_tok(x).reshape(b * n, t, d)
        x = x + self.attn_tok(h, h, h, need_weights=False)[0].reshape(b, n, t, d)

        h = self.norm_cell(x.reshape(b, n, t * d))
        # dist_bias (B*heads, N, N) is added to the attention logits; nearer cells score higher.
        a = self.attn_cell(
            h, h, h, key_padding_mask=pad_mask, attn_mask=dist_bias, need_weights=False
        )[0]
        x = x + a.reshape(b, n, t, d)

        x = x + self.mlp(self.norm_mlp(x))
        if pad_mask is not None:
            x = x * (~pad_mask)[:, :, None, None].to(x.dtype)
        return x


class AxialTower(nn.Module):
    """Chunk of raw expression in, one embedding per cell out."""

    def __init__(
        self,
        n_genes: int,
        n_out: int,
        n_tokens: int = 16,
        token_dim: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_tokens, self.token_dim = n_tokens, token_dim
        wide = n_tokens * token_dim
        self.tokenize = nn.Sequential(nn.Linear(n_genes, wide), nn.GELU(), nn.Dropout(dropout))
        # Positions live on the gene-module axis only. Nothing marks a cell's position in the
        # chunk, so the tower is permutation-equivariant over cells by construction.
        self.token_pos = nn.Parameter(torch.randn(n_tokens, token_dim) * 0.02)
        self.blocks = nn.ModuleList(
            AxialBlock(n_tokens, token_dim, n_heads, mlp_ratio, dropout) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(wide)
        self.head = nn.Linear(wide, n_out)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        dist_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.tokenize(x).reshape(b, n, self.n_tokens, self.token_dim) + self.token_pos
        for blk in self.blocks:
            h = blk(h, pad_mask, dist_bias)
        return self.head(self.norm(h.reshape(b, n, -1)))


class StackContrast(nn.Module):
    """Two towers, additive latent, NB reconstruction."""

    def __init__(
        self,
        n_genes: int,
        n_latent: int = 32,
        n_tokens: int = 16,
        token_dim: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        n_hidden_dec: int = 128,
        sw_weight: float = 1.0,
        n_proj: int = 64,
        lam_w: float = 0.0,
    ) -> None:
        super().__init__()
        kw = dict(
            n_tokens=n_tokens,
            token_dim=token_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        # z and s are summed, so they must share a space and therefore a width.
        self.background_tower = AxialTower(n_genes, n_latent, **kw)
        self.salient_tower = AxialTower(n_genes, n_latent, **kw)
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_hidden_dec),
            nn.LayerNorm(n_hidden_dec),
            nn.GELU(),
            nn.Linear(n_hidden_dec, n_genes),
        )
        self.px_r = nn.Parameter(torch.randn(n_genes) * 0.1)
        self.sw_weight, self.n_proj, self.lam_w = sw_weight, n_proj, lam_w

    def embed(self, counts, is_control, pad_mask=None, dist_bias=None):
        x = torch.log1p(counts)
        z = self.background_tower(x, pad_mask, dist_bias)
        s = self.salient_tower(x, pad_mask, dist_bias)
        # Control cells carry no salient signal by definition, so it is masked rather than
        # learned-then-discarded. With lam_w = 0 nothing else would constrain it.
        s = s * (~is_control).unsqueeze(-1).to(s.dtype)
        return z, s

    def forward(self, counts, is_control, pad_mask=None, dist_bias=None):
        z, s = self.embed(counts, is_control, pad_mask, dist_bias)
        latent = z + s
        px_scale = F.softmax(self.decoder(latent), dim=-1)
        library = counts.sum(-1, keepdim=True).clamp(min=1.0)
        px_rate = px_scale * library
        theta = torch.exp(self.px_r).clamp(min=1e-4, max=1e4)

        nll = _nb_nll(counts, px_rate, theta)
        if pad_mask is not None:
            keep = (~pad_mask).to(nll.dtype)
            recon = (nll.sum(-1) * keep).sum() / keep.sum().clamp(min=1.0)
        else:
            recon = nll.sum(-1).mean()

        sw = sliced_wasserstein(z, self.n_proj)
        perturbed = (~is_control).any()
        if perturbed:
            sw = sw + sliced_wasserstein(s, self.n_proj)
        loss = recon + self.sw_weight * sw
        if self.lam_w > 0:
            # Only meaningful if the salient mask above is lifted; kept for the ablation.
            loss = loss + self.lam_w * (s * is_control.unsqueeze(-1)).pow(2).sum(-1).mean()
        return {"loss": loss, "recon": recon.detach(), "sw": sw.detach(), "z": z, "s": s}


def _nb_nll(x, mu, theta):
    theta = theta.expand_as(mu)
    log_theta_mu = torch.log(theta + mu + 1e-8)
    return -(
        theta * (torch.log(theta + 1e-8) - log_theta_mu)
        + x * (torch.log(mu + 1e-8) - log_theta_mu)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )


def _fit_heads(dim: int, want: int) -> int:
    for h in range(min(want, dim), 0, -1):
        if dim % h == 0:
            return h
    return 1


def spatial_chunks(coords, group, chunk_size=64, seed=0):
    """Partition cells into spatially contiguous chunks, never crossing ``group``.

    Ordering is what defines a chunk, exactly as in stack, so cells are sorted along a
    serpentine raster within each group and then cut into fixed-size blocks.
    """
    coords = np.asarray(coords, dtype=float)
    group = np.asarray(group)
    rng = np.random.default_rng(seed)
    chunks = []
    for g in np.unique(group):
        idx = np.where(group == g)[0]
        xy = coords[idx]
        # Serpentine raster: bucket rows, then alternate the x-direction per row so the
        # order stays contiguous where consecutive rows meet.
        nb = max(1, int(np.sqrt(len(idx) / max(chunk_size, 1))))
        row = np.clip(
            ((xy[:, 1] - xy[:, 1].min()) / (np.ptp(xy[:, 1]) + 1e-9) * nb).astype(int), 0, nb - 1
        )
        key = np.where(row % 2 == 0, xy[:, 0], -xy[:, 0])
        order = idx[np.lexsort((key, row))]
        for i in range(0, len(order), chunk_size):
            chunks.append(order[i : i + chunk_size])
    rng.shuffle(chunks)
    return chunks


class ChunkLoader:
    """Yields padded chunk batches: counts, control mask, pad mask, cell indices."""

    def __init__(self, counts, is_control, chunks, batch_chunks=4, device="cpu", shuffle=True):
        self.counts, self.is_control = counts, is_control
        self.chunks, self.bc = chunks, batch_chunks
        self.device, self.shuffle = device, shuffle
        self.n_max = max(len(c) for c in chunks)

    def __len__(self):
        return int(np.ceil(len(self.chunks) / self.bc))

    def __iter__(self):
        order = np.random.permutation(len(self.chunks)) if self.shuffle else np.arange(len(self.chunks))
        for i in range(0, len(order), self.bc):
            sel = [self.chunks[j] for j in order[i : i + self.bc]]
            b, n, g = len(sel), self.n_max, self.counts.shape[1]
            X = np.zeros((b, n, g), dtype=np.float32)
            C = np.zeros((b, n), dtype=bool)
            P = np.ones((b, n), dtype=bool)
            I = np.full((b, n), -1, dtype=np.int64)
            for k, ch in enumerate(sel):
                m = len(ch)
                X[k, :m] = self.counts[ch]
                C[k, :m] = self.is_control[ch]
                P[k, :m] = False
                I[k, :m] = ch
            t = lambda a, d: torch.as_tensor(a, dtype=d, device=self.device)
            yield t(X, torch.float32), t(C, torch.bool), t(P, torch.bool), t(I, torch.int64)
