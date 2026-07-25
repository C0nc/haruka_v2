"""PyTorch module for Contrastive VI for single cell expression data."""

import math
import warnings
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scvi import REGISTRY_KEYS
from scvi.distributions import (NegativeBinomial,
            Normal,
            Poisson,
            ZeroInflatedNegativeBinomial)
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, FCLayers

torch.backends.cudnn.benchmark = True


class _SpatialRegistryKeys(NamedTuple):
    NEIGHBOR_IDX_KEY: str = "spatial_neighbor_idx"
    NEIGHBOR_MASK_KEY: str = "spatial_neighbor_mask"


SPATIAL_REGISTRY_KEYS = _SpatialRegistryKeys()

# Joint latent width past which the MMD independence penalty stops discriminating
# dependence from independence. See ContrastiveVAE.__init__ for the measurements.
_MMD_DIM_WARN = 128

def AOT_Distance(sample, mean, reduction: str = "mean"):
    """
    :meta private:

    ``reduction`` controls how the per-entry squared errors are pooled.

    ``"mean"`` is the original behaviour and is kept as the default so existing
    models reproduce bit-for-bit. It is, however, not commensurate with the gene
    likelihood, which SUMS over features: the ratio of the two terms then falls
    like ``1 / n_features`` and the COVET block stops influencing the fit at any
    realistic feature count (measured ``me_weight * me_recon / recon`` = 1.2e-2 at
    60 genes, 1.6e-3 at 2000, 5.1e-4 at 6000, and the arcseq RNA has 16253).

    ``"sum"`` pools the same way the gene term does, which makes ``me_weight`` a
    genuine relative weight between the two reconstruction blocks rather than a
    number whose meaning silently depends on ``n_input``.
    """
    # Reshape tensors to 2D: [batch_size, -1]
    sample = torch.reshape(sample, (sample.shape[0], -1))
    mean = torch.reshape(mean, (mean.shape[0], -1))

    # Calculate negative squared difference
    log_prob = -torch.square(sample - mean)

    if reduction == "sum":
        return torch.sum(log_prob, dim=-1)
    if reduction != "mean":
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
    return torch.mean(log_prob, dim=-1)


def grammian_cov(dec_cov):
    """
    :meta private:
    """
    # First create a function to replicate jax.experimental.sparse.fill_triangular
    def fill_triangular(x):
        """
        Fills the lower triangular part of a matrix with the elements from x.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n*(n+1)//2).
        
        Returns:
            torch.Tensor: Filled lower triangular matrices of shape (batch_size, n, n).
        """
        # Determine the size of the triangular matrix
        n = int((np.sqrt(8 * x.shape[-1] + 1) - 1) / 2)
        
        # Initialize a tensor of zeros
        filled = torch.zeros(x.shape[0], n, n, dtype=x.dtype, device=x.device)
        
        # Get the lower triangular indices
        tril_indices = torch.tril_indices(row=n, col=n, offset=0)
        
        # Fill the lower triangular part
        filled[:, tril_indices[0], tril_indices[1]] = x
        
        return filled

    # Fill the triangular matrix
    dec_cov = fill_triangular(dec_cov)

    #print(dec_cov.shape)
    
    # Perform batched matrix multiplication
    # transpose in PyTorch uses different dimension ordering
    return torch.bmm(dec_cov, dec_cov.transpose(1, 2))


class NeighborGATv2(nn.Module):
    """GATv2 attention over a densely gathered neighbor tensor.

    Equivalent to ``torch_geometric.nn.GATv2Conv(n_in, n_out, heads=heads,
    concat=False, add_self_loops=False)`` evaluated on a fixed-size neighbor set,
    which is what lets it run under a minibatch loader that yields spatially
    non-contiguous cells.

    Following torch_geometric's own binding, ``lin_l`` projects the source
    (neighbor) node and is additionally the value path, while ``lin_r`` projects
    the target (self) node and only ever enters the attention logits. The centre
    cell therefore never contributes to its own output except through the
    attention weights, which is the meaning of ``add_self_loops=False``.

    Parameters
    ----------
    n_in
        Width of the per-node input features.
    n_out
        Width of the output features (per head; heads are averaged).
    heads
        Number of attention heads.
    negative_slope
        LeakyReLU slope used inside the attention logit.
    """

    def __init__(self, n_in: int, n_out: int, heads: int = 1, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.heads = heads
        self.negative_slope = negative_slope
        self.lin_l = nn.Linear(n_in, heads * n_out, bias=True)
        self.lin_r = nn.Linear(n_in, heads * n_out, bias=True)
        self.att = nn.Parameter(torch.empty(1, 1, heads, n_out))
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.zeros_(self.lin_l.bias)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.zeros_(self.lin_r.bias)
        nn.init.xavier_uniform_(self.att)

    def forward(
        self,
        h_self: torch.Tensor,
        h_nbr: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attention-pool the neighbors of each cell.

        Parameters
        ----------
        h_self
            ``(batch, n_in)`` features of the centre cells.
        h_nbr
            ``(batch, k, n_in)`` features of each centre cell's k neighbors.
        mask
            ``(batch, k)`` boolean tensor, ``True`` for real neighbors. Padded
            slots are excluded from the softmax and contribute exactly zero.

        Returns
        -------
        ``(batch, n_out)`` pooled neighbor representation.
        """
        batch, k, _ = h_nbr.shape
        src = self.lin_l(h_nbr).view(batch, k, self.heads, self.n_out)
        tgt = self.lin_r(h_self).view(batch, 1, self.heads, self.n_out)
        logits = (F.leaky_relu(src + tgt, self.negative_slope) * self.att).sum(dim=-1)

        if mask is None:
            alpha = torch.softmax(logits, dim=1)
            out = (alpha.unsqueeze(-1) * src).sum(dim=1).mean(dim=1)
            return out + self.bias

        mask = mask.bool()
        keep = mask.unsqueeze(-1)
        # A large finite penalty rather than -inf: a row whose neighbors are all
        # padding (possible for a slice with fewer than k+1 cells) would otherwise
        # softmax to NaN and poison every gradient in the batch.
        logits = logits.masked_fill(~keep, -1e9)
        alpha = torch.softmax(logits, dim=1) * keep.to(logits.dtype)
        out = (alpha.unsqueeze(-1) * src).sum(dim=1).mean(dim=1)
        out = out * mask.any(dim=1, keepdim=True).to(out.dtype)
        return out + self.bias


def _largest_valid_head_count(dim: int, heads: int) -> int:
    """Largest head count ``<= heads`` that divides ``dim``.

    ``nn.MultiheadAttention`` requires ``dim % heads == 0``. Falling back quietly,
    as stack does, beats raising: the caller picks ``token_dim`` for capacity
    reasons and should not have to solve a divisibility puzzle to change it.
    """
    heads = max(1, int(heads))
    for h in range(min(heads, max(dim, 1)), 0, -1):
        if dim % h == 0:
            return h
    return 1


#: ``nn.MultiheadAttention`` launches one CUDA block per sequence, and a CUDA grid
#: dimension caps at 65535. Both axial reshapes below fold real dimensions into the
#: leading (sequence) axis -- the neighbor pass builds ``batch * n_tokens`` sequences
#: -- so at the ContrastiveVI data loader's batch of 4096 with the default 16 tokens
#: that is exactly 65536 and the kernel dies with "CUDA error: invalid configuration
#: argument". That threshold is crossed by ANY dataset with >= 4096 cells per arm,
#: i.e. every real spatial run. Attention is independent across the leading axis, so
#: chunking it is bit-for-bit equivalent, not an approximation.
_MHA_MAX_SEQS = 65535


def _chunked_mha(attn, h, key_padding_mask=None):
    """``attn(h, h, h)`` split along the leading axis to stay under the CUDA grid cap."""
    if h.shape[0] <= _MHA_MAX_SEQS:
        return attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)[0]
    outs = []
    for a in range(0, h.shape[0], _MHA_MAX_SEQS):
        b = min(a + _MHA_MAX_SEQS, h.shape[0])
        kpm = None if key_padding_mask is None else key_padding_mask[a:b]
        chunk = h[a:b]
        outs.append(attn(chunk, chunk, chunk, key_padding_mask=kpm,
                         need_weights=False)[0])
    return torch.cat(outs, dim=0)


class _AxialSetBlock(nn.Module):
    """One axial attention layer over a ``(batch, k, n_tokens, token_dim)`` tensor.

    Two attention passes per layer, following the tabular/axial transformer of the
    Arc "stack" model at roughly 1/300th of its width:

    * TOKEN axis -- each neighbor attends over its own gene-module tokens, on
      ``(batch * k, n_tokens, token_dim)``. Neighbors never mix here.
    * NEIGHBOR axis -- each token slot attends across the k neighbors, on
      ``(batch * n_tokens, k, token_dim)``.

    Then a token-wise MLP with ``mlp_ratio`` expansion. Pre-norm residuals
    throughout; stack post-norms, but that choice is orthogonal to everything this
    block exists for and pre-norm is the more stable ordering.

    There is deliberately NO positional encoding on the neighbor axis. Attention
    without positions is permutation-equivariant, and the pooling that follows is
    permutation-invariant, so the composite is a set function of the k neighbors --
    which is what a niche is. The token axis DOES carry a learned positional
    embedding, because token slots are distinguishable feature groups rather than
    an unordered set.
    """

    def __init__(
        self, token_dim: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm_token = nn.LayerNorm(token_dim)
        self.attn_token = nn.MultiheadAttention(
            token_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_nbr = nn.LayerNorm(token_dim)
        self.attn_nbr = nn.MultiheadAttention(
            token_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * mlp_ratio, token_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        token_pos: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, k, n_tokens, token_dim = x.shape

        h = (self.norm_token(x) + token_pos).reshape(batch * k, n_tokens, token_dim)
        h = _chunked_mha(self.attn_token, h)
        x = x + h.reshape(batch, k, n_tokens, token_dim)

        h = self.norm_nbr(x).permute(0, 2, 1, 3).reshape(batch * n_tokens, k, token_dim)
        kpm = None
        if key_padding_mask is not None:
            # (batch, k) -> (batch * n_tokens, k). repeat_interleave is batch-major,
            # matching the reshape above, which lays rows out b0t0, b0t1, ..., b1t0.
            kpm = key_padding_mask.repeat_interleave(n_tokens, dim=0)
        h = _chunked_mha(self.attn_nbr, h, key_padding_mask=kpm)
        x = x + h.reshape(batch, n_tokens, k, token_dim).permute(0, 2, 1, 3)

        return x + self.mlp(self.norm_mlp(x))


class NeighborSetAttention(nn.Module):
    """Lightweight permutation-invariant set-attention encoder over spatial neighbors.

    A natively reimplemented, heavily shrunk version of the axial ("tabular")
    transformer used by the Arc stack model, sized for one A5000 and 20k-100k cells
    rather than for pretraining: 2 layers, ``token_dim`` 32, 4 heads, and a linear
    compression of the feature axis into a handful of gene-module tokens. Stack's
    pretrained weights are neither used nor usable here -- its vocabulary is human
    and this runs on mouse -- and stack is not a dependency.

    Selectable as the ``spatial_encoder="setattn"`` alternative to
    :class:`NeighborGATv2`. Unlike the GAT it reads the neighbors' log1p expression
    directly rather than their shared-encoder posterior means, so it owns its own
    tokenizer and does not run the shared encoder over the ``batch * k`` gathered
    rows.

    That does NOT make it cheaper. Measured on an A5000, batch 512, k 10, one
    forward+backward of the spatial branch: 3.1 ms (gat) vs 49.8 ms (setattn) at
    2000 genes, 10.6 vs 63.6 at 16253, 28.5 vs 99.5 at 50000 peaks. The tokenizer
    is a wider projection than the shared encoder's first layer (``n_tokens *
    token_dim`` = 512 against ``n_hidden`` = 128), and the axial stack issues many
    small attention kernels on top. It is still comfortably in "minutes" territory
    -- roughly 26 min for 400 epochs over 40k cells at 2000 HVGs -- but pick it for
    the inductive bias, not for speed. ``setattn_tokens=8`` and
    ``setattn_layers=1`` are the two knobs that buy time back.

    Self-exclusion
    --------------
    The centre cell's expression cannot reach this module's output. Three facts, in
    decreasing order of strength:

    1. ``build_neighbor_index`` never lists a cell as its own neighbor, so the
       gathered ``(batch, k, n_in)`` tensor holds only OTHER cells' rows.
    2. The one exception is padded slots, which the index parks on the centre
       cell's own row so the gather stays in range. :meth:`forward` zeroes every
       invalid slot before anything reads it, so past that line the tensor is
       provably not a function of the centre cell.
    3. The padding mask additionally drops those slots from the neighbor-axis
       softmax and from the pooling.

    With ``n_self=0`` (the default) the centre cell has no wire into this module at
    all, not even through attention logits -- strictly stronger than GATv2's
    ``add_self_loops=False``. With ``n_self>0`` the centre supplies a pooling QUERY
    only; the values summed stay neighbor-derived and the weights are renormalised
    to a convex combination, so with identical neighbors the output is that
    neighbor whatever the query says. That is exactly what ``add_self_loops=False``
    means.

    Parameters
    ----------
    n_in
        Feature width of one neighbor row (genes or peaks).
    n_out
        Width of the spatial latent. Two heads of this width are produced: the
        posterior mean and the raw (pre-exp) variance.
    n_tokens
        Number of learned gene-module tokens the feature axis is compressed into.
    token_dim
        Width of each token, and the transformer's ``d_model`` on both axes.
    n_layers
        Number of axial blocks.
    n_heads
        Attention heads, reduced to the largest divisor of ``token_dim`` if needed.
    mlp_ratio
        Expansion factor of the token-wise MLP.
    dropout
        Dropout inside the blocks. 0 by default, which keeps the module
        deterministic given its input and so keeps permutation invariance testable
        in train mode.
    n_batch
        Number of batch categories. When > 1, each neighbor's token stack gets a
        learned per-batch offset, mirroring the batch covariate the GAT path feeds
        to its shared encoder.
    n_self
        Width of the centre-cell feature vector used as a pooling query. ``0``
        disables it and pools by masked mean.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_tokens: int = 16,
        token_dim: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        n_batch: int = 0,
        n_self: int = 0,
    ) -> None:
        super().__init__()
        if min(n_tokens, token_dim, n_layers, mlp_ratio) < 1:
            raise ValueError(
                "n_tokens, token_dim, n_layers and mlp_ratio must all be >= 1, got "
                f"{n_tokens}, {token_dim}, {n_layers}, {mlp_ratio}"
            )
        self.n_in = n_in
        self.n_out = n_out
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.n_self = n_self
        self.n_heads = _largest_valid_head_count(token_dim, n_heads)

        d_pool = n_tokens * token_dim
        self.tokenize = nn.Linear(n_in, d_pool)
        # Token axis only -- see _AxialSetBlock. A neighbor-axis position would make
        # the module order-dependent and it would stop being a set function.
        self.token_pos = nn.Parameter(torch.empty(n_tokens, token_dim))
        self.batch_embedding = nn.Embedding(n_batch, token_dim) if n_batch > 1 else None
        self.blocks = nn.ModuleList(
            _AxialSetBlock(token_dim, self.n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        )
        self.pool_norm = nn.LayerNorm(d_pool)
        self.head_mean = nn.Linear(d_pool, n_out)
        self.head_var = nn.Linear(d_pool, n_out)
        self.self_query = nn.Linear(n_self, d_pool) if n_self > 0 else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.token_pos, mean=0.0, std=0.02)
        if self.batch_embedding is not None:
            nn.init.normal_(self.batch_embedding.weight, mean=0.0, std=0.02)
        # Variance head starts at exactly the prior, exp(0) = 1, so the spatial KL
        # begins near 0 instead of the model having to first walk a random initial
        # posterior back toward N(0, I).
        nn.init.zeros_(self.head_var.weight)
        nn.init.zeros_(self.head_var.bias)

    def forward(
        self,
        x_nbr: torch.Tensor,
        mask: torch.Tensor | None = None,
        h_self: torch.Tensor | None = None,
        batch_nbr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool a cell's k neighbors into posterior parameters for its spatial latent.

        Parameters
        ----------
        x_nbr
            ``(batch, k, n_in)`` neighbor features, already log1p-transformed.
        mask
            ``(batch, k)``, ``True`` for real neighbors. ``None`` treats all as real.
        h_self
            ``(batch, n_self)`` centre-cell features. Used only as a pooling query,
            and only when the module was built with ``n_self > 0``.
        batch_nbr
            ``(batch, k)`` batch code of each neighbor.

        Returns
        -------
        ``(mean, raw_var)``, each ``(batch, n_out)``. ``raw_var`` is pre-exp.
        """
        batch, k, _ = x_nbr.shape
        if mask is None:
            valid = torch.ones(batch, k, dtype=torch.bool, device=x_nbr.device)
        else:
            valid = mask.reshape(batch, k).bool()

        # SELF-EXCLUSION, enforced at the source. build_neighbor_index parks padded
        # slots on the CENTRE CELL's own row so the gather stays in range; leaving
        # them populated would hand a cell its own expression through the spatial
        # branch and collapse the intrinsic/niche distinction the model exists to
        # make. Past this line the tensor is a function of other cells only.
        x_nbr = x_nbr * valid.unsqueeze(-1).to(x_nbr.dtype)

        tokens = self.tokenize(x_nbr).view(batch, k, self.n_tokens, self.token_dim)
        if self.batch_embedding is not None and batch_nbr is not None:
            tokens = tokens + self.batch_embedding(
                batch_nbr.reshape(batch, k).long()
            ).unsqueeze(2)

        ignore = ~valid
        empty = ignore.all(dim=1)
        # torch returns NaN for a query row whose keys are ALL masked, which would
        # poison every gradient in the batch. Such a row has no neighbors at all (a
        # section with fewer than k+1 cells, or every edge trimmed as too long), so
        # un-mask it -- its slots are all-zero after the line above, so nothing real
        # enters -- and send its output to the prior below.
        attn_ignore = ignore & ~empty.unsqueeze(1)
        for block in self.blocks:
            tokens = block(tokens, self.token_pos, attn_ignore)

        flat = tokens.reshape(batch, k, self.n_tokens * self.token_dim)
        weight = valid.to(flat.dtype)
        if self.self_query is not None and h_self is not None:
            logits = (flat * self.self_query(h_self).unsqueeze(1)).sum(dim=-1)
            logits = logits / math.sqrt(flat.shape[-1])
            alpha = torch.softmax(logits.masked_fill(ignore, -1e9), dim=1) * weight
            # Renormalised so the pooled vector is exactly a convex combination of
            # the per-neighbor representations. That is what stops the centre cell's
            # query from contributing anything of its own.
            alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp(min=1e-12)
            pooled = (alpha.unsqueeze(-1) * flat).sum(dim=1)
        else:
            pooled = (flat * weight.unsqueeze(-1)).sum(dim=1) / weight.sum(
                dim=1, keepdim=True
            ).clamp(min=1.0)

        pooled = self.pool_norm(pooled)
        keep = (~empty).to(pooled.dtype).unsqueeze(-1)
        # A cell with no neighbors has no niche to report, so its posterior is the
        # prior exactly: mean 0 and exp(0) = 1 variance. KL contribution 0, no NaN.
        return self.head_mean(pooled) * keep, self.head_var(pooled) * keep


class NicheContextPool(nn.Module):
    """Reduce a cell's niche to ONE fixed-width vector fed to an encoder as INPUT.

    This is the building block of the ``niche_context`` design. The niche is treated as
    a CELL SET -- the cell's k spatial neighbors, optionally including the cell itself --
    and pooled, permutation-invariantly, into a single ``n_context`` vector. That vector
    is concatenated onto the cell's own log1p expression before the background or salient
    encoder reads it.

    Nothing here becomes a latent variable. The niche enters as encoder INPUT, so the
    model keeps its original two latents, its original decoder width, and its original
    contrastive zeroing of ``s`` on control cells. Contrast this with
    :class:`NeighborSetAttention`, which parameterises a SEPARATE third latent -- that is
    the SIMVI-style design, still reachable through ``n_spatial_latent``, and it is a
    different model.

    One instance is built per latent (``context_background`` and ``context_salient``)
    with independent weights, so the background encoder can be handed the niche summary
    that best explains shared tissue architecture while the salient encoder is handed the
    one that best separates the condition-specific microenvironment. A single shared
    pooler would have to compromise between the two; ``context_shared=True`` is available
    as the ablation that measures what that compromise costs.

    Parameters
    ----------
    n_in
        Feature width of one set member (genes or peaks).
    n_context
        Width of the pooled context vector.
    pool
        ``"setattn"`` (default) is the stack-style axial set-attention block: the feature
        axis is linearly compressed into ``n_tokens`` learned gene-module tokens of width
        ``token_dim``, then ``n_layers`` :class:`_AxialSetBlock` layers alternate
        attention over the tokens within a set member and across the set members, with a
        pre-norm residual and a token-wise MLP of expansion ``mlp_ratio``. There is
        deliberately NO positional encoding on the set axis, so the stack is
        permutation-EQUIVARIANT and the masked-mean pooling that follows makes the module
        permutation-INVARIANT: it reads the niche as a set, not as a list.

        ``"attention"`` is :class:`NeighborGATv2` over a single linear projection of the
        set -- one scalar attention weight per member, no cross-member mixing. Cheaper,
        and the constraint that the centre cell stays out of the value path is NOT
        imposed here: that existed only to keep SIMVI's intrinsic-vs-spatial split
        identifiable, and this design does not make that split.

        ``"mean"`` is a masked mean of a single linear projection. It exists as the
        ablation that says whether the attention earns its parameters: if ``"mean"``
        ties ``"setattn"`` on a benchmark, it does not.
    n_tokens, token_dim, n_layers, n_heads, mlp_ratio, dropout
        Shape of the ``"setattn"`` block. Defaults (16, 32, 2, 4, 4, 0.0) are small on
        purpose -- this is a per-dataset module, not a foundation model. ``n_heads`` is
        reduced to the largest divisor of ``token_dim`` if it does not divide.
    gat_heads
        Attention heads for ``pool="attention"`` (averaged, never concatenated).

    Notes
    -----
    No batch-covariate embedding is applied here, unlike :class:`NeighborSetAttention`.
    The context vector is consumed by ``z_encoder`` / ``s_encoder``, whose ``FCLayers``
    inject the one-hot batch covariate at every layer with ``inject_covariates=True``, so
    the batch correction the neighbors need is already downstream of this module. Adding
    it here too would only spend parameters -- and because neighbors never cross sections,
    every member of a given cell's set carries the same batch code anyway, so the
    embedding would contribute a per-row constant.
    """

    _POOLS = ("setattn", "attention", "mean")

    def __init__(
        self,
        n_in: int,
        n_context: int,
        pool: str = "setattn",
        n_tokens: int = 16,
        token_dim: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        gat_heads: int = 1,
    ) -> None:
        super().__init__()
        if pool not in self._POOLS:
            raise ValueError(f"pool must be one of {self._POOLS}, got {pool!r}")
        if n_context < 1:
            raise ValueError(f"n_context must be >= 1, got {n_context}")
        self.pool = pool
        self.n_in = n_in
        self.n_context = n_context

        if pool == "setattn":
            if min(n_tokens, token_dim, n_layers, mlp_ratio) < 1:
                raise ValueError(
                    "n_tokens, token_dim, n_layers and mlp_ratio must all be >= 1, got "
                    f"{n_tokens}, {token_dim}, {n_layers}, {mlp_ratio}"
                )
            self.n_tokens = n_tokens
            self.token_dim = token_dim
            self.n_heads = _largest_valid_head_count(token_dim, n_heads)
            d_pool = n_tokens * token_dim
            self.tokenize = nn.Linear(n_in, d_pool)
            # Token axis only. A set-axis position would make the module order-dependent
            # and it would stop being a set function -- see _AxialSetBlock.
            self.token_pos = nn.Parameter(torch.empty(n_tokens, token_dim))
            self.blocks = nn.ModuleList(
                _AxialSetBlock(token_dim, self.n_heads, mlp_ratio, dropout)
                for _ in range(n_layers)
            )
        elif pool == "attention":
            d_pool = n_context
            self.proj = nn.Linear(n_in, n_context)
            self.gat = NeighborGATv2(n_context, n_context, heads=gat_heads)
        else:
            d_pool = n_context
            self.proj = nn.Linear(n_in, n_context)

        self.pool_norm = nn.LayerNorm(d_pool)
        self.head = nn.Linear(d_pool, n_context)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.pool == "setattn":
            nn.init.normal_(self.token_pos, mean=0.0, std=0.02)

    def forward(
        self,
        x_self: torch.Tensor,
        x_set: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool one cell's niche set into its context vector.

        Parameters
        ----------
        x_self
            ``(batch, n_in)`` centre-cell features, already log1p-transformed. Used only
            as the attention QUERY under ``pool="attention"``; the other two pools ignore
            it. Whether the centre cell is part of the pooled SET is decided by the
            caller, which is what ``include_self`` controls.
        x_set
            ``(batch, m, n_in)`` niche set, already log1p-transformed. ``m`` is ``k`` or
            ``k + 1`` depending on ``include_self``.
        valid
            ``(batch, m)`` boolean, ``True`` for real members. ``None`` treats every slot
            as real. Padded slots contribute exactly zero, and a row with no valid member
            at all returns exactly zero rather than NaN.

        Returns
        -------
        ``(batch, n_context)`` context vector.
        """
        batch, m, _ = x_set.shape
        if valid is None:
            valid = torch.ones(batch, m, dtype=torch.bool, device=x_set.device)
        else:
            valid = valid.reshape(batch, m).bool()

        # Zeroed at the source, before anything reads them: build_neighbor_index parks
        # padded slots on the centre cell's own row so the gather stays in range, and a
        # padded slot must contribute exactly nothing regardless of which pool is used.
        x_set = x_set * valid.unsqueeze(-1).to(x_set.dtype)
        weight = valid.to(x_set.dtype)
        empty = ~valid.any(dim=1)

        if self.pool == "setattn":
            tokens = self.tokenize(x_set).view(batch, m, self.n_tokens, self.token_dim)
            ignore = ~valid
            # torch returns NaN for a query row whose keys are ALL masked, which would
            # poison every gradient in the batch. Such a row has no niche at all, so
            # un-mask it -- its slots are all-zero after the line above, so nothing real
            # enters -- and zero its output below.
            attn_ignore = ignore & ~empty.unsqueeze(1)
            for block in self.blocks:
                tokens = block(tokens, self.token_pos, attn_ignore)
            flat = tokens.reshape(batch, m, self.n_tokens * self.token_dim)
            pooled = (flat * weight.unsqueeze(-1)).sum(dim=1) / weight.sum(
                dim=1, keepdim=True
            ).clamp(min=1.0)
        elif self.pool == "attention":
            pooled = self.gat(self.proj(x_self), self.proj(x_set), valid)
        else:
            proj = self.proj(x_set) * weight.unsqueeze(-1)
            pooled = proj.sum(dim=1) / weight.sum(dim=1, keepdim=True).clamp(min=1.0)

        ctx = self.head(self.pool_norm(pooled))
        # A cell with no niche reports no niche: exactly zero, not a learned bias, so the
        # "padded slots contribute nothing" guarantee holds end to end.
        return ctx * (~empty).to(ctx.dtype).unsqueeze(-1)


class ContrastiveVAE(BaseModuleClass):
    """Variational inference for contrastive analysis of RNA-seq data.

    Implements the contrastiveVI model of :cite:p:`Weinberger23`.

    Parameters
    ----------
    n_input
        Number of input genes.
    n_batch
        Number of batches. If 0, no batch effect correction is performed.
    n_hidden
        Number of nodes per hidden layer.
    n_background_latent
        Dimensionality of the background latent space.
    n_salient_latent
        Dimensionality of the salient latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional
        distribution.
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterize prior on
        library size if not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterize prior
        on library size if not using observed library size.
    wasserstein_penalty
        Weight of the Wasserstein distance loss that further discourages shared
        variations from leaking into the salient latent space.
    me_reduction
        How the COVET reconstruction error is pooled over its entries, ``"mean"``
        (default, original behaviour) or ``"sum"``. See :func:`AOT_Distance`: only
        ``"sum"`` puts the COVET block on the same footing as the gene likelihood,
        and with ``"mean"`` the block is numerically inert at realistic feature
        counts. Left at ``"mean"`` here so existing models are unchanged;
        ``HarukaSpatial`` opts into ``"sum"``.
    n_spatial_latent
        Dimensionality of the SIMVI-style spatial (niche-induced) latent space. ``0``
        disables the spatial branch entirely and the module is numerically identical
        to the non-spatial model.
    spatial_attach
        Which encoder supplies the features the attention pools over, ``"background"``
        or ``"salient"``. Must not exceed that encoder's latent width.
    spatial_encoder
        Which neighbor encoder computes the spatial latent.

        ``"gat"`` (default, and the historical behaviour) is :class:`NeighborGATv2`:
        SIMVI's GATv2 over the neighbors' shared-encoder posterior means.

        ``"setattn"`` is :class:`NeighborSetAttention`, a small axial set-attention
        transformer that tokenizes the neighbors' expression itself. It is
        permutation-invariant over the k neighbors by construction and, at the
        default ``setattn_self_query=False``, gives the centre cell no path into its
        own spatial latent at all. It costs 3-16x more per step than the GAT (see
        that class for measurements), so choose it for the inductive bias rather
        than for throughput.

        Being the stronger encoder, it also competes harder for the reconstruction:
        at SIMVI's ``kl_spatial_weight=0.01`` it wins outright on the synthetic
        benchmark and leaves BOTH intrinsic latents collapsed onto their priors
        (probe accuracy 0.50 against 0.93 for the GAT). Raising
        ``kl_spatial_weight`` to 1.0 restores them. See
        ``HarukaSpatial.setup_model`` for the measured table and for when this
        encoder is and is not the right choice.
    n_attention_heads
        Number of GATv2 attention heads (averaged, never concatenated). Applies to
        ``spatial_encoder="gat"`` only; the set-attention encoder has its own
        ``setattn_heads``.
    setattn_tokens, setattn_token_dim, setattn_layers, setattn_heads
        Shape of the set-attention encoder: how many learned gene-module tokens the
        feature axis is compressed into, their width (also the attention ``d_model``
        on both axes), the number of axial blocks, and the head count. The defaults
        (16, 32, 2, 4) are deliberately small -- this is a lightweight per-dataset
        model, not a foundation model.
    setattn_mlp_ratio, setattn_dropout
        Token-wise MLP expansion and dropout inside the set-attention blocks.
    setattn_self_query
        Let the centre cell's attach-latent slice supply a QUERY for the pooling over
        neighbors. The values pooled stay neighbor-derived either way; this only
        decides whether the centre cell influences the weights, reproducing GATv2's
        ``add_self_loops=False``. ``False`` (default) is the stronger guarantee: the
        centre cell has no wire into the spatial branch whatsoever.
    kl_spatial_weight
        Weight on KL(q(z_sp) || N(0, I)). Deliberately small so the spatial latent
        stays expressive rather than collapsing onto the prior.
    lam_mi
        Weight on the independence penalty between the spatial latent and the
        intrinsic latents. ``5`` suits the default ``reg_to_use="mi"``. The ``"mmd"``
        path is scaled differently again; see ``mmd_bandwidth``.
    reg_to_use
        ``"mi"`` (default) for the closed-form Gaussian mutual information,
        ``"mmd"`` for a kernel MMD against N(0, I), or ``"sw"`` for a sliced
        Wasserstein distance against the same isotropic Gaussian.

        ``"sw"`` is stack's regularizer: 64 random unit projections, sort, mean
        squared difference. It tests the same hypothesis as ``"mmd"`` but costs
        O(n log n) instead of O(n^2), so it needs no row cap. The three are
        calibrated to completely different scales -- ``lam_mi`` is roughly 1000 for
        ``"mmd"``, 5 for ``"mi"``, 1 for ``"sw"`` -- and are not interchangeable at
        a shared weight.

        ``"mi"`` is the default because it is the only one of the two that retains
        discriminative power at the latent widths this model is run at. Measured
        two-sample t between maximally dependent and independent latents, n=512
        rows, spatial width 10:

        ============  ===========  ==============  ===============
        latent width  joint dim    t (mmd)         t (mi, debiased)
        ============  ===========  ==============  ===============
        16            26           56.6            > 500
        64            74           14.0            > 500
        128           138          5.5             > 500
        256           266          0.5             1157
        ============  ===========  ==============  ===============

        The MMD compares ``cat([z, z_sp])`` against N(0, I) in ``joint dim``
        dimensions, and that estimate is noise-dominated once the dimension grows
        past ~150 no matter how the bandwidth is set: at the 256 default it cannot
        distinguish an exact copy of z from an independent draw, so it silently
        stops enforcing the factorization it exists to enforce.
    mmd_bandwidth
        Gaussian kernel bandwidth for ``reg_to_use="mmd"``. ``"median"`` (default)
        uses the median of the squared pairwise distances, which adapts to the
        latent's scale and width. ``"simvi"` reproduces SIMVI's ``dim ** 2``
        exactly; that is faithful to the reference implementation but SIMVI
        calibrated it at ``dim`` around 40, and at ``dim`` = 266 it makes every
        kernel entry land in [0.989, 0.995], i.e. the kernel is the constant 1 and
        the MMD is 0 by construction. The two bandwidths differ in magnitude by
        23x (dim 20) to 86x (dim 266), so ``lam_mi`` is NOT transferable between
        them.
    mi_debias
        Subtract a null estimate, obtained by row-shuffling the spatial latent,
        from the mutual-information penalty. The plug-in Gaussian estimator has a
        finite-sample bias of roughly ``-d_y * log(1 - d_x / n)``, which at
        ``d_x`` = 256 and a 512-row batch is 7.0 nats of pure bias against a true
        value of 0, and diverges to 62 on a short final batch of 176 rows. The
        null shares that bias exactly, so subtracting it recovers a statistic that
        reads 0 under independence and stays comparable across ``latent_dim`` and
        ``batch_size``. It is detached, so the gradient is bit-identical to the
        un-debiased penalty; only the reported value changes.
    sw_n_proj
        Number of random unit projections for ``reg_to_use="sw"``. 64, as in stack.
    penalize_salient_spatial
        Also penalize dependence between the SALIENT and spatial latents. Off by
        default. SIMVI penalizes only intrinsic-vs-spatial, and that arm is well
        posed; salient-vs-spatial is not, because disease structure is spatially
        organized -- a lesion is a contiguous region, so a cell's neighbors are
        lesional exactly when the cell is, the spatial latent carries the lesion
        signal, and forcing independence deletes the lesion from the salient latent
        instead of disentangling anything. Measured on the synthetic benchmark:
        salient/lesion probe 0.529 with this on versus 0.952 with it off (chance
        0.5), while the spatial latent's own niche probe is also better with it off
        (0.708 vs 0.606). Enable it only if you specifically want the salient latent
        to exclude spatially organized condition effects.
    spatial_invariance_penalty
        Weight on the condition-invariance penalty applied to the BACKGROUND spatial
        latent. Read only under ``spatial_attach="both"``, where it is the symmetry
        breaker that stops z_sp from swallowing the lesion niche and starving s_sp;
        see :meth:`_spatial_invariance_penalty`. ``0.0`` ablates it.
    mask_salient_mean_on_background
        Force the salient posterior MEAN -- not merely the decode-path sample -- to
        ~0 on background-arm cells, via a large un-annealed penalty on
        ``||qs_m||^2`` over the background split. Off by default, so nothing changes
        unless asked. The contrastive trick zeroes the salient SAMPLE on the decode
        path only and never constrains the encoder output, so by default s_int is
        full-magnitude on controls (measured control/target ratio of mean |value|
        1.007). Turn this on when downstream code needs s_int to genuinely mean
        "signal absent in controls" -- notably ``spatial_attach="both"``, whose s_sp
        pooling assumes exactly that. See the rationale in :meth:`loss` for why this
        is a penalty on the encoder rather than a mask on its output.
    permutation_rate
        Fraction of genes corrupted during the denoising phase.
    mae_epochs
        Number of initial epochs to run with denoising corruption. ``0`` disables it.
    anneal_epochs
        Epochs over which the spatial KL and the independence penalty ramp to full
        weight.
    var_eps
        Floor added to the spatial posterior variance.
    mmd_max_samples
        Row cap on the MMD penalty, which is quadratic in the number of rows.
    niche_context
        Make BOTH existing latents niche-aware, instead of adding a third one. Each
        encoder is handed its own pooled summary of the cell's spatial neighborhood,
        concatenated onto the cell's own log1p expression::

            ctx_bg  = context_background(x_own, x_nbr, mask)   # (batch, n_context)
            ctx_sal = context_salient   (x_own, x_nbr, mask)   # (batch, n_context)
            qz_m, qz_v, z = z_encoder(cat([x_own, ctx_bg]),  batch_index)
            qs_m, qs_v, s = s_encoder(cat([x_own, ctx_sal]), batch_index)
            latent = cat([z, s])                              # unchanged

        The niche is therefore ENCODER INPUT, never a latent. The decoder stays
        ``DecoderSCVI(n_background_latent + n_salient_latent, n_input)``, ``decoder_me``
        stays on ``cat([z, s])``, and ``s`` is still zeroed on control cells, so the
        contrastive decomposition, the loss and the COVET head are all untouched.

        This is mutually exclusive with ``n_spatial_latent > 0``: that is the SIMVI-style
        design, which answers a different question (separating intrinsic state from
        niche-induced state into two latents) and brings its own independence penalty,
        spatial KL and denoising phase. Setting both raises rather than silently
        combining them. With ``niche_context=True`` every SIMVI term is unreachable:
        ``n_spatial_latent`` is 0, so the independence penalty, the spatial KL, the
        annealing and :meth:`_denoising_gene_mask` are all inert by construction.

        ``False`` (default) leaves the module bit-for-bit as it was: no module is
        constructed, no RNG is consumed, and the encoder widths are unchanged.
    n_context
        Width of EACH pooled niche vector. Both encoders widen to
        ``n_input + n_context``; the library encoder keeps reading the cell's own
        expression only, because library size is a property of the cell, not its niche.
    context_pool
        How the niche set is pooled -- ``"setattn"`` (default), ``"attention"`` or
        ``"mean"``. See :class:`NicheContextPool`. ``"mean"`` is the ablation that tells
        you whether the attention is earning its parameters.
    context_shared
        ``False`` (default) builds TWO independent poolers, one per latent. This is the
        point of the design: the background latent wants the niche context that is shared
        across conditions (tissue architecture) while the salient latent wants the one
        that distinguishes the lesion microenvironment, and a single pooler would have to
        represent a compromise between them. ``True`` builds one pooler and feeds both
        encoders the same vector, and exists only as the ablation that measures the cost
        of that compromise.
    include_self
        Let the cell's own row be a member of its own niche set. ``True`` by default.
        The exclusion that :class:`NeighborGATv2` and :class:`NeighborSetAttention`
        enforce exists only to keep SIMVI's intrinsic-vs-niche split identifiable -- if
        the cell could read itself through the spatial branch, "niche-induced" would stop
        meaning anything. No such split is being made here, and a niche that contains the
        cell is the natural object, so exclusion is offered but not required.
    mask_rate
        Stack-style RECTANGULAR masked-feature regularisation on the niche branch: a
        random subset of gene columns is zeroed, the SAME columns for every member of the
        set, resampled each training step. ``0.0`` (default) disables it and consumes no
        randomness. It applies to the pooler input only -- the cell's own expression as
        seen directly by the encoders, the library size and the reconstruction target are
        all read from the unmasked data -- so it regularises how the niche is summarised
        without degrading anything else.

        Stack draws the rate per step from ``U(0.2, 0.8)``; this is the deterministic
        analogue, one fixed rate, chosen so the regulariser is reproducible and the
        module stays testable. ``0.5`` is the midpoint of stack's band.
    """

    #: Weight on the background-arm salient-mean penalty enabled by
    #: ``mask_salient_mean_on_background``. Large and un-annealed on purpose -- see the
    #: rationale in ``loss``.
    _SALIENT_MEAN_MASK_WEIGHT = 1000.0

    def __init__(
        self,
        n_input: int,
        n_input_full: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_background_latent: int = 10,
        n_salient_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        use_observed_lib_size: bool = True,
        library_log_means: np.ndarray | None = None,
        library_log_vars: np.ndarray | None = None,
        wasserstein_penalty: float = 0,
        gene_likelihood = 'nb',
        me_weight = 0.3,
        me_reduction: str = "mean",
        n_spatial_latent: int = 0,
        spatial_attach: str = "background",
        spatial_encoder: str = "gat",
        n_attention_heads: int = 1,
        setattn_tokens: int = 16,
        setattn_token_dim: int = 32,
        setattn_layers: int = 2,
        setattn_heads: int = 4,
        setattn_mlp_ratio: int = 4,
        setattn_dropout: float = 0.0,
        setattn_self_query: bool = False,
        kl_spatial_weight: float = 0.01,
        lam_mi: float = 5.0,
        reg_to_use: str = "mi",
        mmd_bandwidth: str = "median",
        mi_debias: bool = True,
        penalize_salient_spatial: bool = False,
        spatial_invariance_penalty: float = 0.5,
        mask_salient_mean_on_background: bool = False,
        me_from_background_only: bool = False,
        sw_n_proj: int = 64,
        permutation_rate: float = 0.5,
        mae_epochs: int = 0,
        anneal_epochs: int = 50,
        var_eps: float = 1e-4,
        mmd_max_samples: int = 512,
        niche_context: bool = False,
        n_context: int = 128,
        context_pool: str = "setattn",
        context_shared: bool = False,
        include_self: bool = True,
        mask_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_input_full = n_input_full
        self.n_batch = n_batch
        self.n_hidden = n_hidden
        self.n_background_latent = n_background_latent
        self.n_salient_latent = n_salient_latent
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.latent_distribution = "normal"
        self.dispersion = "gene"
        self.px_r = torch.nn.Parameter(torch.randn(n_input))
        self.use_observed_lib_size = use_observed_lib_size
        self.wasserstein_penalty = wasserstein_penalty
        self.me_weight = me_weight
        if me_reduction not in ("mean", "sum"):
            raise ValueError(f"me_reduction must be 'mean' or 'sum', got {me_reduction!r}")
        self.me_reduction = me_reduction
        self.gene_likelihood = gene_likelihood

        # NICHE CONTEXT. Resolved before anything else spatial, because it decides the
        # encoder input width and it forecloses the entire SIMVI branch.
        if niche_context and n_spatial_latent > 0:
            raise ValueError(
                "niche_context=True and n_spatial_latent > 0 are two different designs "
                "and cannot be combined. niche_context makes the EXISTING background and "
                "salient latents niche-aware by feeding each encoder a pooled summary of "
                "the cell's neighborhood; n_spatial_latent adds a SEPARATE third latent "
                "(the SIMVI design) with its own independence penalty, spatial KL and "
                "denoising phase. Pick one: niche_context=True with n_spatial_latent=0, "
                "or niche_context=False with n_spatial_latent>0."
            )
        if niche_context:
            if context_pool not in NicheContextPool._POOLS:
                raise ValueError(
                    f"context_pool must be one of {NicheContextPool._POOLS}, got "
                    f"{context_pool!r}"
                )
            if n_context < 1:
                raise ValueError(f"n_context must be >= 1, got {n_context}")
            if not 0.0 <= mask_rate < 1.0:
                raise ValueError(f"mask_rate must be in [0, 1), got {mask_rate}")
        self.niche_context = niche_context
        self.n_context = n_context if niche_context else 0
        self.context_pool = context_pool
        self.context_shared = context_shared
        self.include_self = include_self
        self.mask_rate = mask_rate

        if spatial_attach not in ("background", "salient", "both"):
            raise ValueError(
                "spatial_attach must be 'background', 'salient' or 'both', got "
                f"{spatial_attach!r}"
            )
        if spatial_encoder not in ("gat", "setattn"):
            raise ValueError(
                f"spatial_encoder must be 'gat' or 'setattn', got {spatial_encoder!r}"
            )
        # The 2x2 decomposition is defined in terms of two independent GAT pairs, one
        # reading each intrinsic latent. The set-attention encoder tokenizes neighbor
        # EXPRESSION rather than a latent slice, so "two of them" would be two copies of
        # the same computation differing only by their self query -- not a background /
        # salient split. Refused rather than silently built.
        if spatial_attach == "both" and spatial_encoder != "gat":
            raise ValueError(
                "spatial_attach='both' requires spatial_encoder='gat'; the set-attention "
                "encoder reads neighbor expression, not the attach latent, so it has no "
                "background-versus-salient split to make."
            )
        if reg_to_use not in ("mmd", "mi", "sw"):
            raise ValueError(
                f"reg_to_use must be 'mmd', 'mi' or 'sw', got {reg_to_use!r}"
            )
        if mmd_bandwidth not in ("median", "simvi"):
            raise ValueError(
                f"mmd_bandwidth must be 'median' or 'simvi', got {mmd_bandwidth!r}"
            )
        if spatial_attach == "background":
            n_attach_latent = n_background_latent
        elif spatial_attach == "salient":
            n_attach_latent = n_salient_latent
        else:
            # "both" reads the trailing slice of BOTH intrinsic means, so the tighter
            # of the two widths is the binding constraint.
            n_attach_latent = min(n_background_latent, n_salient_latent)
        # The width limit exists because the spatial branch reads the TRAILING
        # n_spatial_latent dimensions of the attach encoder's posterior mean. The
        # set-attention encoder tokenizes the neighbors' expression itself and only
        # touches that slice when it is also given a self query, so outside that
        # case the limit does not apply and is not imposed.
        reads_attach_latent = spatial_encoder == "gat" or setattn_self_query
        if reads_attach_latent and n_spatial_latent > n_attach_latent:
            which = (
                "min(background, salient)" if spatial_attach == "both" else spatial_attach
            )
            raise ValueError(
                f"n_spatial_latent ({n_spatial_latent}) cannot exceed the "
                f"{which} latent width ({n_attach_latent}); the spatial branch "
                "reads the trailing n_spatial_latent dimensions of that latent."
            )
        # The MMD is taken over cat([z, z_sp]) against N(0, I), so its power is set by
        # that joint width. Measured t between maximally dependent and independent
        # latents at 512 rows: 14.0 at joint dim 74, 8.2 at 106, 5.5 at 138, 2.1 at
        # 202, 0.5 at 266. Past ~128 it stops being an independence test at all, and
        # no bandwidth repairs that -- it is the dimension, not the kernel.
        joint_dim = max(n_background_latent, n_salient_latent) + n_spatial_latent
        if n_spatial_latent > 0 and reg_to_use == "mmd" and joint_dim > _MMD_DIM_WARN:
            warnings.warn(
                f"reg_to_use='mmd' with a joint latent width of {joint_dim} "
                f"(> {_MMD_DIM_WARN}): the MMD independence penalty is "
                "noise-dominated at this dimension and will not measurably enforce "
                "independence between the spatial and intrinsic latents. Use "
                "reg_to_use='mi' (lam_mi=5), or reduce the latent width.",
                RuntimeWarning,
                stacklevel=3,
            )

        self.n_spatial_latent = n_spatial_latent
        self.spatial_attach = spatial_attach
        self.spatial_encoder = spatial_encoder
        self.n_attention_heads = n_attention_heads
        self.kl_spatial_weight = kl_spatial_weight
        self.lam_mi = lam_mi
        self.reg_to_use = reg_to_use
        self.mmd_bandwidth = mmd_bandwidth
        self.mi_debias = mi_debias
        self.penalize_salient_spatial = penalize_salient_spatial
        self.spatial_invariance_penalty = spatial_invariance_penalty
        self.mask_salient_mean_on_background = mask_salient_mean_on_background
        # COVET read from z ALONE. On cat([z,s]) the salient latent can satisfy the
        # niche target, which leaves z unconstrained and free to absorb the condition.
        self.me_from_background_only = me_from_background_only
        # Under "both" the decoder receives TWO spatial blocks, so every width that
        # counts spatial dimensions doubles. Everything else keys off this one number.
        self.n_spatial_blocks = 2 if (n_spatial_latent > 0 and spatial_attach == "both") else (
            1 if n_spatial_latent > 0 else 0
        )
        self.sw_n_proj = sw_n_proj
        # One-shot guards, so a degenerate configuration is reported once rather
        # than once per training step.
        self._warned_mi_conditioning = False
        self.permutation_rate = permutation_rate
        self.mae_epochs = mae_epochs
        self.anneal_epochs = anneal_epochs
        self.var_eps = var_eps
        self.mmd_max_samples = mmd_max_samples
        # Written each epoch by SpatialContrastiveTrainingPlan; the module is otherwise
        # never told which epoch it is in.
        self.current_epoch_ = 0

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )
            self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
            self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())

        cat_list = [n_batch]
        # With niche_context both intrinsic encoders read the cell's own expression AND
        # its pooled niche context. The library encoder deliberately does not: library
        # size is a property of the cell, not of its neighborhood.
        n_encoder_input = n_input + self.n_context
        # Background encoder.
        self.z_encoder = Encoder(
            n_encoder_input,
            n_background_latent,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=self.latent_distribution,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )
        # Salient encoder.
        self.s_encoder = Encoder(
            n_encoder_input,
            n_salient_latent,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=self.latent_distribution,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )
        # Library size encoder.
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )
        # Decoder from latent variable to distribution parameters in data space.
        n_total_latent = (
            n_background_latent
            + n_salient_latent
            + self.n_spatial_blocks * n_spatial_latent
        )
        self.decoder = DecoderSCVI(
            n_total_latent,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
        )
        g_out = int(np.sqrt((n_input_full - n_input)))
        g_out = int((g_out + 1) * g_out / 2)
        #print)
        self.decoder_me = FCLayers(
            n_in=(n_background_latent if me_from_background_only else n_total_latent),
            n_out= g_out,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
        )

        # Built last, and only when enabled, so that with niche_context == False the
        # parameter-initialisation RNG draws are byte-identical to the model without
        # this feature.
        if niche_context:
            pool_kwargs = {
                "n_in": n_input,
                "n_context": n_context,
                "pool": context_pool,
                "n_tokens": setattn_tokens,
                "token_dim": setattn_token_dim,
                "n_layers": setattn_layers,
                "n_heads": setattn_heads,
                "mlp_ratio": setattn_mlp_ratio,
                "dropout": setattn_dropout,
                "gat_heads": n_attention_heads,
            }
            # TWO independent poolers, one per latent -- see the class docstring for why
            # sharing them would be a compromise rather than an economy. They share the
            # neighbor GATHER (pure data movement; gathering twice would be waste) but
            # never share weights, so perturbing one cannot move the other's latent.
            self.context_background = NicheContextPool(**pool_kwargs)
            # Plain None rather than a second reference to the same module: aliasing it
            # would duplicate every tensor in state_dict.
            self.context_salient = (
                None if context_shared else NicheContextPool(**pool_kwargs)
            )
            # A cell's neighbors are essentially never in its own minibatch, so the full
            # expression matrix has to be resident and gathered by global row id.
            # Non-persistent: excluded from state_dict, re-attached after load().
            self.register_buffer("nbr_bank", None, persistent=False)
            self.register_buffer("nbr_batch", None, persistent=False)

        if n_spatial_latent > 0:
            if spatial_encoder == "gat":
                self.gat_mean = NeighborGATv2(
                    n_spatial_latent, n_spatial_latent, heads=n_attention_heads
                )
                self.gat_var = NeighborGATv2(
                    n_spatial_latent, n_spatial_latent, heads=n_attention_heads
                )
                # 2x2: a SECOND, fully independent GAT pair reading the salient latent.
                # Built after the first so that with spatial_attach != "both" the
                # parameter-init RNG draws are byte-identical to before this feature.
                if spatial_attach == "both":
                    self.gat_mean_salient = NeighborGATv2(
                        n_spatial_latent, n_spatial_latent, heads=n_attention_heads
                    )
                    self.gat_var_salient = NeighborGATv2(
                        n_spatial_latent, n_spatial_latent, heads=n_attention_heads
                    )
            else:
                # One trunk, two heads: running the whole transformer twice to get a
                # mean and a variance, as the GAT path does, would double the cost for
                # nothing.
                self.set_attn = NeighborSetAttention(
                    n_in=n_input,
                    n_out=n_spatial_latent,
                    n_tokens=setattn_tokens,
                    token_dim=setattn_token_dim,
                    n_layers=setattn_layers,
                    n_heads=setattn_heads,
                    mlp_ratio=setattn_mlp_ratio,
                    dropout=setattn_dropout,
                    n_batch=n_batch,
                    n_self=n_spatial_latent if setattn_self_query else 0,
                )
            # A cell's neighbors are essentially never in its own minibatch, so the
            # full expression matrix has to be resident and gathered by global row id.
            # Non-persistent: excluded from state_dict, re-attached after load().
            self.register_buffer("nbr_bank", None, persistent=False)
            self.register_buffer("nbr_batch", None, persistent=False)

    def attach_neighbor_bank(
        self, expression: np.ndarray | torch.Tensor, batch_index: np.ndarray | torch.Tensor
    ) -> None:
        """Install the gene-expression matrix the spatial branch gathers neighbors from.

        Parameters
        ----------
        expression
            ``(n_cells, n_input)`` gene block of the full registered matrix, in the
            same row order the neighbor indices refer to.
        batch_index
            ``(n_cells,)`` or ``(n_cells, 1)`` batch codes, so neighbors are encoded
            under their own batch covariate.
        """
        if self.n_spatial_latent == 0 and not self.niche_context:
            raise ValueError(
                "attach_neighbor_bank requires n_spatial_latent > 0 or niche_context=True"
            )
        expression = torch.as_tensor(np.asarray(expression), dtype=torch.float32)
        if expression.ndim != 2 or expression.shape[1] != self.n_input:
            raise ValueError(
                f"expression must be (n_cells, {self.n_input}), got {tuple(expression.shape)}"
            )
        batch_index = torch.as_tensor(np.asarray(batch_index)).reshape(-1).long()
        if batch_index.shape[0] != expression.shape[0]:
            raise ValueError(
                f"batch_index has {batch_index.shape[0]} rows but expression has "
                f"{expression.shape[0]}"
            )
        device = next(self.parameters()).device
        self.nbr_bank = expression.to(device)
        self.nbr_batch = batch_index.to(device)

    @auto_move_data
    def _compute_local_library_params(
        self, batch_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes local library parameters.

        Compute two tensors of shape (batch_index.shape[0], 1) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the batch the cell corresponds to.
        """
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            F.one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_means
        )
        local_library_log_vars = F.linear(
            F.one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_vars
        )
        return local_library_log_means, local_library_log_vars

    @staticmethod
    def _get_min_batch_size(concat_tensors: dict[str, dict[str, torch.Tensor]]) -> int:
        return min(
            concat_tensors["background"][REGISTRY_KEYS.X_KEY].shape[0],
            concat_tensors["target"][REGISTRY_KEYS.X_KEY].shape[0],
        )

    @staticmethod
    def _reduce_tensors_to_min_batch_size(
        tensors: dict[str, torch.Tensor], min_batch_size: int
    ) -> None:
        for name, tensor in tensors.items():
            tensors[name] = tensor[:min_batch_size, :]

    def _get_inference_input_from_concat_tensors(
        self, concat_tensors: dict[str, dict[str, torch.Tensor]], index: str
    ) -> dict[str, torch.Tensor]:
        tensors = concat_tensors[index]
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        input_dict = {"x": x, "batch_index": batch_index}
        if self.n_spatial_latent > 0 or self.niche_context:
            neighbor_idx = tensors.get(SPATIAL_REGISTRY_KEYS.NEIGHBOR_IDX_KEY, None)
            if neighbor_idx is None:
                need = (
                    "niche_context=True" if self.niche_context else "n_spatial_latent > 0"
                )
                raise ValueError(
                    f"{need} but no neighbor index is registered. Pass "
                    "neighbor_idx_key to setup_anndata after building the graph with "
                    "ContrastiveVI.build_neighbor_index."
                )
            input_dict["neighbor_idx"] = neighbor_idx
            neighbor_mask = tensors.get(SPATIAL_REGISTRY_KEYS.NEIGHBOR_MASK_KEY, None)
            if neighbor_mask is not None:
                input_dict["neighbor_mask"] = neighbor_mask
        return input_dict

    def _get_inference_input(
        self, concat_tensors: dict[str, dict[str, torch.Tensor]]
    ) -> dict[str, dict[str, torch.Tensor]]:
        background = self._get_inference_input_from_concat_tensors(concat_tensors, "background")
        target = self._get_inference_input_from_concat_tensors(concat_tensors, "target")
        # Ensure batch sizes are the same.
        min_batch_size = self._get_min_batch_size(concat_tensors)
        self._reduce_tensors_to_min_batch_size(background, min_batch_size)
        self._reduce_tensors_to_min_batch_size(target, min_batch_size)
        return {"background": background, "target": target}

    @staticmethod
    def _get_generative_input_from_concat_tensors(
        concat_tensors: dict[str, dict[str, torch.Tensor]], index: str
    ) -> dict[str, torch.Tensor]:
        tensors = concat_tensors[index]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        input_dict = {"batch_index": batch_index}
        return input_dict

    def _get_generative_input_from_inference_outputs(
        self, inference_outputs: dict[str, dict[str, torch.Tensor]], data_source: str
    ) -> dict[str, torch.Tensor]:
        z = inference_outputs[data_source]["z"]
        s = inference_outputs[data_source]["s"]
        library = inference_outputs[data_source]["library"]
        out = {"z": z, "s": s, "library": library}
        if self.n_spatial_latent > 0:
            out["z_sp"] = inference_outputs[data_source]["z_sp"]
            if self.spatial_attach == "both":
                out["s_sp"] = inference_outputs[data_source]["s_sp"]
        return out

    def _get_generative_input(
        self,
        concat_tensors: dict[str, dict[str, torch.Tensor]],
        inference_outputs: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, dict[str, torch.Tensor]]:
        background_tensor_input = self._get_generative_input_from_concat_tensors(
            concat_tensors, "background"
        )
        target_tensor_input = self._get_generative_input_from_concat_tensors(
            concat_tensors, "target"
        )
        # Ensure batch sizes are the same.
        min_batch_size = self._get_min_batch_size(concat_tensors)
        self._reduce_tensors_to_min_batch_size(background_tensor_input, min_batch_size)
        self._reduce_tensors_to_min_batch_size(target_tensor_input, min_batch_size)

        background_inference_outputs = self._get_generative_input_from_inference_outputs(
            inference_outputs, "background"
        )
        target_inference_outputs = self._get_generative_input_from_inference_outputs(
            inference_outputs, "target"
        )
        background = {**background_tensor_input, **background_inference_outputs}
        target = {**target_tensor_input, **target_inference_outputs}
        return {"background": background, "target": target}

    @staticmethod
    def _reshape_tensor_for_samples(tensor: torch.Tensor, n_samples: int):
        return tensor.unsqueeze(0).expand((n_samples, tensor.size(0), tensor.size(1)))

    def _denoising_gene_mask(self, x_: torch.Tensor) -> torch.Tensor | None:
        """Gene subset to corrupt this step, or None outside the denoising phase.

        Consumes no randomness when disabled, which is what keeps the disabled
        configuration byte-identical to the non-spatial model.
        """
        if (
            self.n_spatial_latent == 0
            or self.mae_epochs <= 0
            or not self.training
            or self.current_epoch_ >= self.mae_epochs
        ):
            return None
        return torch.rand(x_.shape[1], device=x_.device) < self.permutation_rate

    @staticmethod
    def _permute_genes(values: torch.Tensor, gene_mask: torch.Tensor) -> torch.Tensor:
        """Shuffle the masked genes' values across rows, preserving each gene's marginal.

        Breaks the cell-gene association so the spatial branch cannot recover a cell's
        own expression from its own row and pass it off as neighbor-induced.
        """
        sub = values[:, gene_mask]
        perm = torch.argsort(torch.rand(sub.shape, device=values.device), dim=0)
        out = values.clone()
        out[:, gene_mask] = sub.gather(0, perm)
        return out

    def _rectangular_gene_mask(self, n_genes: int, device) -> torch.Tensor | None:
        """Gene columns to zero in the niche branch this step, or None when disabled.

        Stack's "rectangular" masking: ONE column subset, shared by every member of every
        set, so the pooler is denied a feature block wholesale rather than being handed a
        differently-corrupted view of each neighbor. Consumes no randomness when
        ``mask_rate`` is 0, which is what keeps the disabled configuration reproducible.
        """
        if self.mask_rate <= 0.0 or not self.training:
            return None
        return torch.rand(n_genes, device=device) < self.mask_rate

    def _niche_contexts(
        self,
        x_own: torch.Tensor,
        neighbor_idx: torch.Tensor | None,
        neighbor_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool each cell's niche into the two context vectors its encoders read.

        Parameters
        ----------
        x_own
            ``(batch, n_input)`` centre-cell expression, ALREADY log1p-transformed.
        neighbor_idx
            ``(batch, k)`` global row positions into the resident neighbor bank.
        neighbor_mask
            ``(batch, k)`` validity mask, or ``None`` to treat every slot as valid.

        Returns
        -------
        ``(ctx_bg, ctx_sal)``, each ``(batch, n_context)``. Under
        ``context_shared=True`` these are the SAME tensor, computed once.
        """
        if neighbor_idx is None:
            raise ValueError(
                "niche_context=True but no neighbor index was supplied to inference."
            )
        if self.nbr_bank is None:
            raise ValueError(
                "The neighbor expression bank is not attached. It is excluded from "
                "state_dict, so call attach_neighbor_bank (or ContrastiveVI."
                "attach_neighbor_bank) after loading a saved model."
            )

        idx = neighbor_idx.long()
        n_cells, k = idx.shape

        # THE NICHE AS A CELL SET. One gather, shared by both poolers.
        x_nbr = self.nbr_bank[idx.reshape(-1)]
        x_nbr = torch.nan_to_num(x_nbr, nan=0.0, posinf=0.0, neginf=0.0)
        x_nbr = torch.log(1 + x_nbr.clamp(min=0)).reshape(n_cells, k, self.n_input)

        if neighbor_mask is None:
            valid = torch.ones(n_cells, k, dtype=torch.bool, device=x_nbr.device)
        else:
            valid = neighbor_mask.reshape(n_cells, k) > 0.5

        gene_mask = self._rectangular_gene_mask(self.n_input, x_nbr.device)
        if gene_mask is not None:
            # Rebound locally. The caller's x_ is untouched, so the encoders' own-
            # expression block, the library size and the reconstruction target all still
            # see uncorrupted data -- only the niche summary is regularised.
            x_own = x_own.masked_fill(gene_mask, 0.0)
            x_nbr = x_nbr.masked_fill(gene_mask, 0.0)

        if self.include_self:
            # The cell joins its own niche. Excluding it is what SIMVI needs to keep
            # "niche-induced" identifiable against "intrinsic"; this design draws no such
            # line, and a niche containing the cell is the natural object.
            x_set = torch.cat([x_own.unsqueeze(1), x_nbr], dim=1)
            valid = torch.cat(
                [torch.ones(n_cells, 1, dtype=torch.bool, device=valid.device), valid],
                dim=1,
            )
        else:
            x_set = x_nbr

        ctx_bg = self.context_background(x_own, x_set, valid)
        if self.context_salient is None:
            # context_shared: one block feeding both encoders, computed once.
            return ctx_bg, ctx_bg
        return ctx_bg, self.context_salient(x_own, x_set, valid)

    def _spatial_pool(
        self,
        arm: str,
        attach_mean: torch.Tensor,
        nbr_x: torch.Tensor,
        flat: torch.Tensor,
        mask: torch.Tensor | None,
        n_cells: int,
        k: int,
        second_pair: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool one arm's neighbor latents into a spatial posterior.

        ``arm`` names which intrinsic encoder supplies the neighbor representation.
        ``second_pair`` selects the SECOND GAT pair, which exists only under
        ``spatial_attach="both"``; the single-latent configurations all use the first
        pair regardless of which encoder they read, which is what keeps their state
        dicts and their numerics unchanged.

        Under "both" this runs twice over the SAME gathered neighbor expression -- the
        gather is pure data movement, so doing it once and pooling it twice is the only
        sensible split.
        """
        h_self = attach_mean[:, -self.n_spatial_latent :]
        nbr_batch = self.nbr_batch[flat].unsqueeze(-1)
        encoder = self.s_encoder if arm == "salient" else self.z_encoder
        if second_pair:
            gat_mean, gat_var = self.gat_mean_salient, self.gat_var_salient
        else:
            gat_mean, gat_var = self.gat_mean, self.gat_var
        nbr_mean = encoder(nbr_x, nbr_batch)[0]
        h_nbr = nbr_mean[:, -self.n_spatial_latent :].reshape(
            n_cells, k, self.n_spatial_latent
        )
        qsp_m = gat_mean(h_self, h_nbr, mask)
        qsp_v = torch.exp(gat_var(h_self, h_nbr, mask)) + self.var_eps
        return qsp_m, qsp_v, Normal(qsp_m, qsp_v.sqrt()).rsample()

    def _spatial_inference(
        self,
        attach_mean: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor | None,
        gene_mask: torch.Tensor | None,
        salient_mean: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if neighbor_idx is None:
            raise ValueError(
                "n_spatial_latent > 0 but no neighbor index was supplied to inference."
            )
        if self.nbr_bank is None:
            raise ValueError(
                "The neighbor expression bank is not attached. It is excluded from "
                "state_dict, so call attach_neighbor_bank (or ContrastiveVI."
                "attach_neighbor_bank) after loading a saved model."
            )

        idx = neighbor_idx.long()
        n_cells, k = idx.shape
        flat = idx.reshape(-1)

        nbr_x = self.nbr_bank[flat]
        nbr_x = torch.nan_to_num(nbr_x, nan=0.0, posinf=0.0, neginf=0.0)
        if gene_mask is not None:
            nbr_x = self._permute_genes(nbr_x, gene_mask)
        nbr_x = torch.log(1 + nbr_x.clamp(min=0))

        mask = None if neighbor_mask is None else (neighbor_mask > 0.5)

        if self.spatial_encoder == "setattn":
            # The set-attention encoder tokenizes the neighbors' expression itself,
            # so the shared encoder is not run over the batch * k gathered rows here.
            qsp_m, raw_v = self.set_attn(
                nbr_x.reshape(n_cells, k, self.n_input),
                mask,
                attach_mean[:, -self.n_spatial_latent :],
                self.nbr_batch[flat].reshape(n_cells, k),
            )
            qsp_v = torch.exp(raw_v) + self.var_eps
            return qsp_m, qsp_v, Normal(qsp_m, qsp_v.sqrt()).rsample()

        if self.spatial_attach != "both":
            arm = "background" if self.spatial_attach == "background" else "salient"
            return self._spatial_pool(arm, attach_mean, nbr_x, flat, mask, n_cells, k)

        # 2x2. z_sp pools the neighbors' BACKGROUND latents (niche structure that
        # exists in both arms); s_sp pools their SALIENT latents (niche structure
        # specific to the diseased arm). No mask forces s_sp to zero on controls:
        # s_int is already driven to ~0 there by the Wasserstein term, so the salient
        # encoder returns ~0 for every neighbor of a control cell and s_sp follows.
        qsp_m, qsp_v, z_sp = self._spatial_pool(
            "background", attach_mean, nbr_x, flat, mask, n_cells, k
        )
        qssp_m, qssp_v, s_sp = self._spatial_pool(
            "salient", salient_mean, nbr_x, flat, mask, n_cells, k, second_pair=True
        )
        return qsp_m, qsp_v, z_sp, qssp_m, qssp_v, s_sp

    @auto_move_data
    def _generic_inference(
        self,
        x: torch.Tensor,
        index: int,
        batch_index: torch.Tensor,
        n_samples: int = 1,
        neighbor_idx: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x_ = x[:, :index]
        x_ = torch.nan_to_num(x_, nan=0.0, posinf=0.0, neginf=0.0)
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1).clamp(min=1e-6)).unsqueeze(1)

        # Corruption applies to the encoder input only; the library size and the
        # reconstruction target are both read from the uncorrupted x.
        gene_mask = self._denoising_gene_mask(x_)
        if gene_mask is not None:
            x_ = self._permute_genes(x_, gene_mask)

        x_ = torch.log(1 + x_.clamp(min=0))

        # NICHE CONTEXT: each encoder is handed the cell's own expression plus its OWN
        # pooled summary of the cell's neighborhood. Two separate poolers, so the two
        # contexts are different vectors and each latent gets the niche summary that
        # serves it. This adds no latent -- the niche enters as encoder input only.
        h_bg = h_sal = x_
        if self.niche_context:
            ctx_bg, ctx_sal = self._niche_contexts(x_, neighbor_idx, neighbor_mask)
            h_bg = torch.cat([x_, ctx_bg], dim=-1)
            h_sal = torch.cat([x_, ctx_sal], dim=-1)

        qz_m, qz_v, z = self.z_encoder(h_bg, batch_index)
        qs_m, qs_v, s = self.s_encoder(h_sal, batch_index)

        ql_m, ql_v = None, None
        if not self.use_observed_lib_size:
            ql_m, ql_v, library_encoded = self.l_encoder(x_[:, :self.n_input], batch_index)
            library = library_encoded

        qsp_m = qsp_v = z_sp = None
        qssp_m = qssp_v = s_sp = None
        if self.n_spatial_latent > 0:
            attach_mean = qz_m if self.spatial_attach != "salient" else qs_m
            spatial_out = self._spatial_inference(
                attach_mean, neighbor_idx, neighbor_mask, gene_mask, salient_mean=qs_m
            )
            if self.spatial_attach == "both":
                qsp_m, qsp_v, z_sp, qssp_m, qssp_v, s_sp = spatial_out
            else:
                qsp_m, qsp_v, z_sp = spatial_out

        if n_samples > 1:
            qz_m = self._reshape_tensor_for_samples(qz_m, n_samples)
            qz_v = self._reshape_tensor_for_samples(qz_v, n_samples)
            z = self._reshape_tensor_for_samples(z, n_samples)
            qs_m = self._reshape_tensor_for_samples(qs_m, n_samples)
            qs_v = self._reshape_tensor_for_samples(qs_v, n_samples)
            s = self._reshape_tensor_for_samples(s, n_samples)
            if z_sp is not None:
                qsp_m = self._reshape_tensor_for_samples(qsp_m, n_samples)
                qsp_v = self._reshape_tensor_for_samples(qsp_v, n_samples)
                z_sp = self._reshape_tensor_for_samples(z_sp, n_samples)
            if s_sp is not None:
                qssp_m = self._reshape_tensor_for_samples(qssp_m, n_samples)
                qssp_v = self._reshape_tensor_for_samples(qssp_v, n_samples)
                s_sp = self._reshape_tensor_for_samples(s_sp, n_samples)

            if self.use_observed_lib_size:
                library = self._reshape_tensor_for_samples(library, n_samples)
            else:
                ql_m = self._reshape_tensor_for_samples(ql_m, n_samples)
                ql_v = self._reshape_tensor_for_samples(ql_v, n_samples)
                library = Normal(ql_m, ql_v.sqrt()).sample()

        outputs = {
            "z": z,
            "qz_m": qz_m,
            "qz_v": qz_v,
            "s": s,
            "qs_m": qs_m,
            "qs_v": qs_v,
            "library": library,
            "ql_m": ql_m,
            "ql_v": ql_v,
        }
        if self.n_spatial_latent > 0:
            outputs["z_sp"] = z_sp
            outputs["qsp_m"] = qsp_m
            outputs["qsp_v"] = qsp_v
            if self.spatial_attach == "both":
                outputs["s_sp"] = s_sp
                outputs["qssp_m"] = qssp_m
                outputs["qssp_v"] = qssp_v
        return outputs

    @auto_move_data
    def inference(
        self,
        background: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        n_samples: int = 1,
    ) -> dict[str, dict[str, torch.Tensor]]:
        background_batch_size = background["x"].shape[0]
        target_batch_size = target["x"].shape[0]
        inference_input = {}
        for key in background.keys():
            inference_input[key] = torch.cat([background[key], target[key]], dim=0)
        inference_input['index'] = self.n_input
        outputs = self._generic_inference(**inference_input, n_samples=n_samples)
        batch_size_dim = 0 if n_samples == 1 else 1
        background_outputs, target_outputs = {}, {}
        for key in outputs.keys():
            if outputs[key] is not None:
                background_tensor, target_tensor = torch.split(
                    outputs[key],
                    [background_batch_size, target_batch_size],
                    dim=batch_size_dim,
                )
            else:
                background_tensor, target_tensor = None, None
            background_outputs[key] = background_tensor
            target_outputs[key] = target_tensor
        background_outputs["s"] = torch.zeros_like(background_outputs["s"])
        # z_sp is deliberately NOT zeroed here. Zeroing s is what defines salient as
        # "what target cells have and background cells do not". Niche-induced variation
        # is not condition-specific: control tissue has real spatial structure too, and
        # zeroing it would delete the healthy-niche baseline that the diseased niche is
        # only interpretable against, train the attention on half the cells, and put all
        # the spatial reconstruction pressure on the target arm.
        return {"background": background_outputs, "target": target_outputs}

    @auto_move_data
    def _generic_generative(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
        z_sp: torch.Tensor | None = None,
        s_sp: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.n_spatial_latent > 0:
            zeros = lambda: torch.zeros(  # noqa: E731
                *z.shape[:-1], self.n_spatial_latent, device=z.device, dtype=z.dtype
            )
            if z_sp is None:
                z_sp = zeros()
            if self.spatial_attach == "both":
                if s_sp is None:
                    s_sp = zeros()
                # Each intrinsic block sits next to its own spatial block.
                latent = torch.cat([z, z_sp, s, s_sp], dim=-1)
            else:
                latent = torch.cat([z, s, z_sp], dim=-1)
        else:
            latent = torch.cat([z, s], dim=-1)
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            latent,
            library,
            batch_index,
        )
        px_rate = torch.nan_to_num(px_rate, nan=1e-6, posinf=1e6, neginf=1e-6).clamp(min=1e-6)
        px_scale = torch.nan_to_num(px_scale, nan=1.0, posinf=1e4, neginf=1e-6).clamp(min=1e-6)
        if self.n_input_full > self.n_input:
            me_pred = self.decoder_me(
                z if self.me_from_background_only else latent,
                batch_index,
            )
        else:
            me_pred = torch.empty(latent.shape[0], 0, device=latent.device, dtype=latent.dtype)
        
        px_r = torch.exp(self.px_r)
        return {
            'me_pred': me_pred,
            "px_scale": px_scale,
            "px_r": px_r,
            "px_rate": px_rate,
            "px_dropout": px_dropout,
        }

    @auto_move_data
    def generative(
        self,
        background: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> dict[str, dict[str, torch.Tensor]]:
        latent_z_shape = background["z"].shape
        batch_size_dim = 0 if len(latent_z_shape) == 2 else 1
        background_batch_size = background["z"].shape[batch_size_dim]
        target_batch_size = target["z"].shape[batch_size_dim]
        generative_input = {}
        latent_keys = ["z", "s", "library"]
        if self.n_spatial_latent > 0:
            latent_keys.append("z_sp")
            if self.spatial_attach == "both":
                latent_keys.append("s_sp")
        for key in latent_keys:
            generative_input[key] = torch.cat([background[key], target[key]], dim=batch_size_dim)
        generative_input["batch_index"] = torch.cat(
            [background["batch_index"], target["batch_index"]], dim=0
        )
        outputs = self._generic_generative(**generative_input)
        background_outputs, target_outputs = {}, {}
        for key in ["px_scale", "px_rate", "px_dropout", 'me_pred']:
            if outputs[key] is not None:
                background_tensor, target_tensor = torch.split(
                    outputs[key],
                    [background_batch_size, target_batch_size],
                    dim=batch_size_dim,
                )
            else:
                background_tensor, target_tensor = None, None
            background_outputs[key] = background_tensor
            target_outputs[key] = target_tensor
        background_outputs["px_r"] = outputs["px_r"]
        target_outputs["px_r"] = outputs["px_r"]
        return {"background": background_outputs, "target": target_outputs}

    @staticmethod
    def reconstruction_loss(
        x: torch.Tensor,
        index: int,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
        px_scale: torch.Tensor,
        gene_likelihood: str
    ) -> torch.Tensor:
        """Computes likelihood loss for zero-inflated negative binomial distribution.

        Parameters
        ----------
        xadata.obsm['spatial']
            Input data.
        px_rate
            Mean of distribution.
        px_r
            Inverse dispersion.
        px_dropout
            Logits scale of zero inflation probability.

        Returns
        -------
        Negative log likelihood (reconstruction loss) for each data point. If number of
        latent samples == 1, the tensor has shape (batch_size, ). If number of latent
        samples > 1, the tensor has shape (n_samples, batch_size).
        """
        x_obs = x[:, :index]
        if gene_likelihood in ("zinb", "nb", "poisson"):
            x_obs = x_obs.round().clamp(min=0).long()
        if gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout, scale=px_scale)
        elif gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif gene_likelihood == "poisson":
            px = Poisson(rate=px_rate, scale=px_scale, validate_args=False)
        elif gene_likelihood == "normal":
            px = Normal(px_rate, px_r, normal_mu=px_scale)

        recon_loss = (
            -px
            .log_prob(x_obs)
            .sum(dim=-1)
        )
        return recon_loss


    @staticmethod
    def me_reconstruction_loss(
        x: torch.Tensor,
        index: int,
        me_pred: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Computes likelihood loss for zero-inflated negative binomial distribution.

        Parameters
        ----------
        x
            Input data.
        px_rate
            Mean of distribution.
        px_r
            Inverse dispersion.
        px_dropout
            Logits scale of zero inflation probability.

        Returns
        -------
        Negative log likelihood (reconstruction loss) for each data point. If number of
        latent samples == 1, the tensor has shape (batch_size, ). If number of latent
        samples > 1, the tensor has shape (n_samples, batch_size).
        """
        #print(grammian_cov(me_pred))
        if me_pred.shape[1] == 0 or index >= x.shape[1]:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        recon_loss = -AOT_Distance(grammian_cov(me_pred), x[:, index:], reduction)
        #recon_loss = torch.nn.MSELoss()(me_pred, x[:, index:])
        return recon_loss


    @staticmethod
    def latent_kl_divergence(
        variational_mean: torch.Tensor,
        variational_var: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_var: torch.Tensor,
    ) -> torch.Tensor:
        """Computes KL divergence between a variational posterior and prior Gaussian.

        Parameters
        ----------
        variational_mean
            Mean of the variational posterior Gaussian.
        variational_var
            Variance of the variational posterior Gaussian.
        prior_mean
            Mean of the prior Gaussian.
        prior_var
            Variance of the prior Gaussian.

        Returns
        -------
        KL divergence for each data point. If number of latent samples == 1, the tensor
        has shape (batch_size, ). If number of latent samples > 1, the tensor has shape
        (n_samples, batch_size).
        """
        return kl(
            Normal(variational_mean, variational_var.sqrt()),
            Normal(prior_mean, prior_var.sqrt()),
        ).sum(dim=-1)

    def library_kl_divergence(
        self,
        batch_index: torch.Tensor,
        variational_library_mean: torch.Tensor,
        variational_library_var: torch.Tensor,
        library: torch.Tensor,
    ) -> torch.Tensor:
        """Computes KL divergence between library size variational posterior and prior.

        Both the variational posterior and prior are Log-Normal.

        Parameters
        ----------
        batch_index
            Batch indices for batch-specific library size mean and variance.
        variational_library_mean
            Mean of variational Log-Normal.
        variational_library_var
            Variance of variational Log-Normal.
        library
            Sampled library size.

        Returns
        -------
        KL divergence for each data point. If number of latent samples == 1, the tensor
        has shape (batch_size, ). If number of latent samples > 1, the tensor has shape
        (n_samples, batch_size).
        """
        if not self.use_observed_lib_size:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)

            kl_library = kl(
                Normal(variational_library_mean, variational_library_var.sqrt()),
                Normal(local_library_log_means, local_library_log_vars.sqrt()),
            )
        else:
            kl_library = torch.zeros_like(library)
        return kl_library.sum(dim=-1)

    def _anneal_weight(self) -> float:
        """SIMVI's min(1, epoch / anneal_epochs) on the spatial terms only.

        Epochs are 1-indexed here to match SIMVI, so the first epoch already carries a
        non-zero weight.
        """
        if self.anneal_epochs <= 0:
            return 1.0
        return float(min(1.0, (self.current_epoch_ + 1) / self.anneal_epochs))

    @staticmethod
    def _sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x2 = x.pow(2).sum(dim=1, keepdim=True)
        y2 = y.pow(2).sum(dim=1, keepdim=True)
        return (x2 + y2.transpose(0, 1) - 2.0 * (x @ y.transpose(0, 1))).clamp(min=0.0)

    def _median_bandwidth(self, sample: torch.Tensor) -> torch.Tensor:
        """Median of the squared pairwise distances, detached.

        Detached because the bandwidth is a property of the current cloud, not a
        quantity to optimise: leaving it attached lets the encoder shrink the
        penalty by inflating its own scale, which is the degenerate solution.
        """
        n = sample.shape[0]
        m = min(n, 256)
        if m < 2:
            return torch.ones((), device=sample.device, dtype=sample.dtype)
        sub = sample[torch.randperm(n, device=sample.device)[:m]]
        sq = self._sq_dists(sub, sub)
        iu = torch.triu_indices(m, m, offset=1, device=sq.device)
        return sq[iu[0], iu[1]].median().detach().clamp(min=1e-8)

    def _mmd_kernel(
        self, x: torch.Tensor, y: torch.Tensor, bandwidth: torch.Tensor
    ) -> torch.Tensor:
        return torch.exp(-self._sq_dists(x, y) / bandwidth)

    def _mmd_penalty(self, sample: torch.Tensor) -> torch.Tensor:
        ref = torch.randn_like(sample)
        if self.mmd_bandwidth == "simvi":
            # SIMVI's exp(-||x-y||^2 / dim^2): it divides by the dimensionality
            # twice. Faithful to the reference implementation, but calibrated there
            # at dim ~= 40. At dim = 266 every kernel entry lands in [0.989, 0.995],
            # so the kernel is the constant 1 and the statistic is 0 by construction.
            # Kept only for exact SIMVI reproduction.
            dim = sample.shape[1]
            bw = torch.tensor(float(dim * dim), device=sample.device, dtype=sample.dtype)
        else:
            # One bandwidth for all three terms, taken from the sample cloud, so the
            # three kernel matrices remain comparable.
            bw = self._median_bandwidth(sample)
        k = self._mmd_kernel
        return (
            k(sample, sample, bw).mean()
            + k(ref, ref, bw).mean()
            - 2.0 * k(sample, ref, bw).mean()
        )

    @staticmethod
    def _second_moment(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Uncentred, as in SIMVI. Both latents are KL-pulled toward zero mean, and
        # lam_mi is calibrated against this form.
        return a.transpose(0, 1) @ b / max(a.shape[0] - 1, 1)

    def _mi_raw(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c_xx = self._second_moment(x, x)
        c_yy = self._second_moment(y, y)
        c_yx = self._second_moment(y, x)
        c_xy = self._second_moment(x, y)
        eye_x = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
        eye_y = torch.eye(y.shape[1], device=y.device, dtype=y.dtype)
        c_xx_inv = torch.inverse(c_xx + eye_x * 1e-3)
        resid = c_yy - torch.linalg.multi_dot([c_yx, c_xx_inv, c_xy])
        # Deviation from SIMVI: it ridges only C_xx, so both logdets diverge to -inf if
        # the spatial latent collapses to a low-rank subspace.
        return -torch.logdet(resid + eye_y * 1e-6) + torch.logdet(c_yy + eye_y * 1e-6)

    def _mi_penalty(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Closed-form Gaussian MI, optionally debiased against a shuffled null.

        The plug-in estimator regresses y on x from n rows, so it carries a
        finite-sample bias of about ``-d_y * log(1 - d_x / n)`` even when the two
        are exactly independent: 0.4 nats at d_x=20/n=512, 7.0 at d_x=256/n=512,
        and 62 at d_x=256/n=176 (a short final batch). Row-shuffling y destroys
        every dependence while preserving n, d_x, d_y and both marginals, so the
        null carries the same bias and subtracting it leaves ~0 under
        independence (measured 0.02 at d_x=256/n=512, 0.12 at n=176) while
        retaining essentially all of the signal (62.0 of 69.0).
        """
        raw = self._mi_raw(x, y)
        if not self.mi_debias:
            return raw
        n = x.shape[0]
        if n < 3:
            return raw
        if n <= x.shape[1] and not self._warned_mi_conditioning:
            self._warned_mi_conditioning = True
            warnings.warn(
                f"The mutual-information penalty is regressing a {x.shape[1]}-dim "
                f"latent from only {n} rows. The covariance is rank-deficient at "
                "this ratio; the debiasing keeps the reported value honest but the "
                "term is noisy. Raise batch_size above the latent width, or pass "
                "datasplitter_kwargs=dict(drop_last=True) to avoid short batches.",
                RuntimeWarning,
                stacklevel=4,
            )
        # Detached: the null is a per-step constant offset, so the gradient into the
        # intrinsic encoder is bit-identical to the un-debiased penalty (verified:
        # max |grad difference| exactly 0). Only the reported value changes. Letting
        # gradient through would ask the encoder to MAXIMISE its spurious dependence
        # on a shuffled latent, which is meaningless.
        with torch.no_grad():
            perm = torch.randperm(n, device=y.device)
            null = self._mi_raw(x.detach(), y[perm])
        return raw - null

    def _sliced_wasserstein_penalty(self, sample: torch.Tensor) -> torch.Tensor:
        """Sliced 2-Wasserstein between the joint latent and an isotropic Gaussian.

        The cheap alternative to :meth:`_mmd_penalty`, taken from stack: draw
        ``sw_n_proj`` random unit directions, project both clouds, sort each
        projection, and average the squared differences of the order statistics.
        That is O(n log n) per projection against MMD's O(n^2), so it needs no row
        cap and its cost does not grow with batch size the way the kernel does.

        It is zero only when every one-dimensional projection of the joint matches
        N(0, 1), which forces the joint itself to be standard normal and therefore
        its two blocks independent -- the same argument the MMD-against-N(0, I)
        penalty rests on, so this is a drop-in for it.

        The sample is centred first, as stack does, so the statistic measures shape
        rather than location; the latents' own KL terms are what pull their means to
        zero. NOTE the scale differs by orders of magnitude from both other options,
        so ``lam_mi`` is not transferable: ``"mmd"`` wants 1000, ``"mi"`` wants 5,
        and this wants roughly 1.
        """
        n, dim = sample.shape
        if n < 2:
            return sample.sum() * 0.0
        proj = torch.randn(self.sw_n_proj, dim, device=sample.device, dtype=sample.dtype)
        proj = proj / proj.norm(dim=1, keepdim=True).clamp(min=1e-12)
        centred = sample - sample.mean(dim=0, keepdim=True)
        reference = torch.randn_like(sample)
        x_sorted = (centred @ proj.t()).sort(dim=0).values
        y_sorted = (reference @ proj.t()).sort(dim=0).values
        return ((x_sorted - y_sorted) ** 2).mean()

    def _pair_penalty(self, x: torch.Tensor, y_detached: torch.Tensor) -> torch.Tensor:
        if self.reg_to_use == "mi":
            return self._mi_penalty(x, y_detached)
        sample = torch.cat([x, y_detached], dim=1)
        if self.reg_to_use == "sw":
            # Deliberately uncapped: being linearithmic in the rows rather than
            # quadratic is the entire reason this option exists.
            return self._sliced_wasserstein_penalty(sample)
        if sample.shape[0] > self.mmd_max_samples:
            keep = torch.randperm(sample.shape[0], device=sample.device)[: self.mmd_max_samples]
            sample = sample[keep]
        return self._mmd_penalty(sample)

    def _independence_penalty(
        self, inference_outputs: dict[str, dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        background = inference_outputs["background"]
        target = inference_outputs["target"]

        def flat(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(-1, t.shape[-1])

        # INTENTIONAL: detaching the spatial latent is load-bearing, not a backprop bug.
        # The spatial latent has a hard structural definition (attention-pooled from the
        # neighbors, self excluded from the value path) and that definition carries the
        # scientific meaning. If gradient were allowed in, the cheapest way to satisfy
        # the penalty is not to decorrelate but to degrade z_sp toward isotropic noise,
        # which scores perfectly and destroys the decomposition. With the detach the
        # penalty reads one-directionally: z_sp is given, and the intrinsic latents must
        # carve out variation it does not already explain. Do not "fix" this.
        z = torch.cat([flat(background["z"]), flat(target["z"])], dim=0)
        z_sp = torch.cat([flat(background["z_sp"]), flat(target["z_sp"])], dim=0).detach()
        penalty = self._pair_penalty(z, z_sp)

        # The salient-vs-spatial arm is OFF by default, and that is a deliberate
        # scientific decision rather than an oversight.
        #
        # SIMVI only ever penalises INTRINSIC vs spatial, and that arm is well posed:
        # a cell's own state and its neighbours' influence are different things.
        # Salient vs spatial is not. Salient is the condition-specific signal, and
        # disease structure is spatially organised almost by definition -- a lesion is
        # a contiguous region, so a cell's neighbours are lesional exactly when the
        # cell is, and the spatial latent therefore carries the lesion signal itself.
        # Forcing s _||_ z_sp then does not disentangle anything; it deletes the
        # lesion from the salient latent, which is the one thing Haruka exists to
        # find. Measured on the synthetic benchmark, where the lesion is a disc:
        # salient/lesion probe 0.529 with this arm on versus 0.952 with it off
        # (chance 0.5), and the spatial latent's own niche probe is BETTER without it
        # (0.708 vs 0.606) -- the arm costs the salient latent everything and buys the
        # spatial latent nothing.
        #
        # This was invisible until the independence penalty was repaired: with the
        # old dim^2-bandwidth MMD the term delivered no dependence-carrying gradient,
        # so the arm was inert and harmless. Turning it on only makes sense if you
        # specifically want salient to capture condition effects that are NOT
        # spatially organised.
        if self.penalize_salient_spatial:
            # s is identically zero on every background cell, so this arm is restricted
            # to the target arm; otherwise it would measure the zero block.
            penalty = penalty + self._pair_penalty(
                flat(target["s"]), flat(target["z_sp"]).detach()
            )
        return penalty

    def _spatial_invariance_penalty(
        self, inference_outputs: dict[str, dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Force the BACKGROUND spatial latent to be condition-invariant.

        THE SYMMETRY BREAKER for ``spatial_attach="both"``. z_sp and s_sp are both
        spatial and therefore compete for the same niche signal, and the competition is
        rigged: z_sp receives gradient from BOTH arms while s_sp only ever sees the
        diseased one, so left alone z_sp absorbs the lesion niche and s_sp starves.

        Penalising the discrepancy between the control-arm and target-arm distributions
        of z_sp closes that route: anything z_sp encodes must look the same in healthy
        and diseased tissue, so condition-specific niche signal has nowhere to go but
        s_sp. The statistic is the Gaussian-W2 proxy -- squared difference of the arm
        means plus squared difference of the arm variances, summed over dimensions.

        Both arms are attached (unlike the independence penalty, which detaches z_sp on
        purpose): the point here is precisely to reshape z_sp, not to route around it.
        """
        def flat(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(-1, t.shape[-1])

        bg = flat(inference_outputs["background"]["z_sp"])
        tg = flat(inference_outputs["target"]["z_sp"])
        if bg.shape[0] < 2 or tg.shape[0] < 2:
            return bg.sum() * 0.0
        mean_gap = (bg.mean(dim=0) - tg.mean(dim=0)).pow(2).sum()
        var_gap = (bg.var(dim=0, unbiased=False) - tg.var(dim=0, unbiased=False)).pow(2).sum()
        return mean_gap + var_gap

    def _generic_loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        generative_outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        qs_m = inference_outputs["qs_m"]
        qs_v = inference_outputs["qs_v"]
        library = inference_outputs["library"]
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]
        px_rate = generative_outputs["px_rate"]
        px_scale = generative_outputs["px_scale"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]
        me_pred = generative_outputs["me_pred"]

        prior_z_m = torch.zeros_like(qz_m)
        prior_z_v = torch.ones_like(qz_v)
        prior_s_m = torch.zeros_like(qs_m)
        prior_s_v = torch.ones_like(qs_v)

        recon_loss = self.reconstruction_loss(x, self.n_input, px_rate, px_r, px_dropout, px_scale, self.gene_likelihood)
        #print(me_pred, x, self.n_input)
        pred_me_loss = self.me_reconstruction_loss(
            x, self.n_input, me_pred, self.me_reduction
        )
        kl_z = self.latent_kl_divergence(qz_m, qz_v, prior_z_m, prior_z_v)
        kl_s = self.latent_kl_divergence(qs_m, qs_v, prior_s_m, prior_s_v)
        kl_library = self.library_kl_divergence(batch_index, ql_m, ql_v, library)
        losses = {
            "recon_loss": recon_loss,
            "kl_z": kl_z,
            "kl_s": kl_s,
            "kl_library": kl_library,
            'me_recon_loss': pred_me_loss
        }
        if self.n_spatial_latent > 0:
            qsp_m = inference_outputs["qsp_m"]
            qsp_v = inference_outputs["qsp_v"]
            losses["kl_sp"] = self.latent_kl_divergence(
                qsp_m, qsp_v, torch.zeros_like(qsp_m), torch.ones_like(qsp_v)
            )
            if self.spatial_attach == "both":
                qssp_m = inference_outputs["qssp_m"]
                qssp_v = inference_outputs["qssp_v"]
                # Folded into the same kl_sp term, so it carries the identical
                # kl_spatial_weight down-weight the background spatial block gets.
                losses["kl_sp"] = losses["kl_sp"] + self.latent_kl_divergence(
                    qssp_m, qssp_v, torch.zeros_like(qssp_m), torch.ones_like(qssp_v)
                )
        return losses

    def loss(
        self,
        concat_tensors: dict[str, dict[str, torch.Tensor]],
        inference_outputs: dict[str, dict[str, torch.Tensor]],
        generative_outputs: dict[str, dict[str, torch.Tensor]],
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Computes loss terms for contrastiveVI.

        Parameters
        ----------
        concat_tensors
            Tuple of data mini-batch. The first element contains background data
            mini-batch. The second element contains target data mini-batch.
        inference_outputs
            Dictionary of inference step outputs. The keys are "background" and
            "target" for the corresponding outputs.
        generative_outputs
            Dictionary of generative step outputs. The keys are "background" and
            "target" for the corresponding outputs.
        kl_weight
            Importance weight for KL divergence of background and salient latent
            variables, relative to KL divergence of library size.

        Returns
        -------
        An scvi.module.base.LossOutput instance that records the following:
        loss
            One-dimensional tensor for overall loss used for optimization.
        reconstruction_loss
            Reconstruction loss with shape (n_samples, batch_size) if number of latent
            samples > 1, or (batch_size, ) if number of latent samples == 1.
        kl_local
            KL divergence term with shape (n_samples, batch_size) if number of latent
            samples > 1, or (batch_size, ) if number of latent samples == 1.
        """
        background_tensors = concat_tensors["background"]
        target_tensors = concat_tensors["target"]
        # Ensure batch sizes are the same.
        min_batch_size = self._get_min_batch_size(concat_tensors)
        self._reduce_tensors_to_min_batch_size(background_tensors, min_batch_size)
        self._reduce_tensors_to_min_batch_size(target_tensors, min_batch_size)

        background_losses = self._generic_loss(
            background_tensors,
            inference_outputs["background"],
            generative_outputs["background"],
        )
        target_losses = self._generic_loss(
            target_tensors,
            inference_outputs["target"],
            generative_outputs["target"],
        )
        reconst_loss = background_losses["recon_loss"] + target_losses["recon_loss"] + self.me_weight  * (background_losses["me_recon_loss"] + target_losses["me_recon_loss"])
        kl_divergence_z = background_losses["kl_z"] + target_losses["kl_z"]
        kl_divergence_s = target_losses["kl_s"]
        kl_divergence_l = background_losses["kl_library"] + target_losses["kl_library"]

        wasserstein_loss = torch.norm(
            inference_outputs["background"]["qs_m"], dim=-1
        ) ** 2 + torch.sum(inference_outputs["background"]["qs_v"], dim=-1)

        kl_local_for_warmup = kl_divergence_z + kl_divergence_s
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = (
            kl_weight * (self.wasserstein_penalty * wasserstein_loss + kl_local_for_warmup)
            + kl_local_no_warmup
        )

        kl_local = {
            "kl_divergence_l": kl_divergence_l,
            "kl_divergence_z": kl_divergence_z,
            "kl_divergence_s": kl_divergence_s,
        }
        extra_metrics = {"wasserstein_loss_sum": torch.sum(wasserstein_loss)}

        independence_penalty = None
        invariance = None
        if self.n_spatial_latent > 0:
            anneal = self._anneal_weight()
            kl_divergence_sp = background_losses["kl_sp"] + target_losses["kl_sp"]
            weighted_kl_local = (
                weighted_kl_local + anneal * self.kl_spatial_weight * kl_divergence_sp
            )
            kl_local["kl_divergence_sp"] = kl_divergence_sp
            independence_penalty = self._independence_penalty(inference_outputs)
            extra_metrics["independence_penalty"] = independence_penalty.detach()
            extra_metrics["spatial_anneal_weight"] = anneal
            if self.spatial_attach == "both" and self.spatial_invariance_penalty > 0:
                invariance = self._spatial_invariance_penalty(inference_outputs)
                extra_metrics["spatial_invariance"] = invariance.detach()

        # OPT-IN: force the salient posterior MEAN to ~0 on background-arm cells, so
        # that s_int genuinely means "signal absent in controls".
        #
        # WHY A PENALTY AND NOT AN EXPLICIT MASK. Zeroing `qs_m` on the background
        # split after the encoder runs would satisfy the letter of the requirement and
        # miss the point. The salient-spatial pooling path in `_spatial_pool` re-runs
        # `self.s_encoder` over the raw NEIGHBOR expression; it never reads `qs_m`. A
        # post-encoder mask therefore leaves the encoder FUNCTION unconstrained on
        # control-arm expression, and s_sp for a control cell -- a pooled s_encoder
        # output over that cell's (control) neighbors -- would be completely
        # unaffected. Only a penalty that shapes the encoder function itself
        # generalises to the pooling path, which is the premise the 2x2 needs.
        #
        # The weight is deliberately large and NOT multiplied by `kl_weight`: the
        # existing annealed `wasserstein_penalty` is exactly the too-weak mechanism
        # this is meant to replace, so annealing it would reintroduce the failure.
        # Only the mean is penalised; adding the variance term would collapse the
        # salient posterior to a point mass on the background arm.
        salient_mean_mask = None
        if self.mask_salient_mean_on_background:
            salient_mean_mask = torch.norm(
                inference_outputs["background"]["qs_m"], dim=-1
            ) ** 2
            extra_metrics["salient_mean_mask"] = torch.mean(salient_mean_mask).detach()

        reconst_loss = torch.nan_to_num(reconst_loss, nan=1e6, posinf=1e6, neginf=0.0)
        reconst_loss = torch.clamp(reconst_loss, max=1e6)
        loss = torch.mean(reconst_loss + weighted_kl_local)
        if salient_mean_mask is not None:
            loss = loss + self._SALIENT_MEAN_MASK_WEIGHT * torch.mean(salient_mean_mask)
        if independence_penalty is not None:
            loss = loss + anneal * self.lam_mi * independence_penalty
        if invariance is not None:
            loss = loss + anneal * self.spatial_invariance_penalty * invariance

        # LossOutput internally sums the reconst_loss, kl_local, and kl_global
        # terms before logging, so we do the same for the `wasserstein_loss` term.
        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_local,
            extra_metrics=extra_metrics,
        )

    def sample(self):
        raise NotImplementedError("No sampling method for contrastiveVI")
