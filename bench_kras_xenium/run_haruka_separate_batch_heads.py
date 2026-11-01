#!/usr/bin/env python3
"""Run shared-decoder Haruka with one batch key and three head-specific projectors.

The assay decoder keeps its native scVI categorical batch projector.  The
composition and neighbor-mean heads receive distinct learned additive batch
embeddings, so technical effects need not share a geometry across objectives.
Latent z/s encoders remain batch-blind through decoder-only conditioning.
"""
from __future__ import annotations
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))

import json
import sys
from pathlib import Path

import torch

torch.set_float32_matmul_precision("high")

BASE_DIR = Path(f"{_HR}/bench_mouse_brain_scviva")
sys.path.insert(0, str(BASE_DIR))
import run_haruka_shared_decoder as base  # noqa: E402


class SeparateBatchHeadNicheDecoder(base.SharedNicheDecoder):
    """Shared latent trunk with independent composition/neighbor batch maps."""

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
        # Build both task heads at the original latent width. Batch effects are
        # projected separately into that width and added before each head.
        super().__init__(
            latent_dim,
            n_cell_types,
            neighbor_embedding_dim,
            hidden,
            n_layers,
            n_batches=0,
            batch_embedding_dim=0,
            neighbor_expression_mode=neighbor_expression_mode,
        )
        self.batch_embedding = None
        self.composition_batch_embedding = None
        self.neighbor_mean_batch_embedding = None
        if batch_embedding_dim > 0:
            if n_batches < 1:
                raise ValueError("Separate batch heads require n_batches > 0")
            self.composition_batch_embedding = torch.nn.Sequential(
                torch.nn.Embedding(int(n_batches), int(batch_embedding_dim)),
                torch.nn.Linear(int(batch_embedding_dim), int(latent_dim), bias=False),
            )
            self.neighbor_mean_batch_embedding = torch.nn.Sequential(
                torch.nn.Embedding(int(n_batches), int(batch_embedding_dim)),
                torch.nn.Linear(int(batch_embedding_dim), int(latent_dim), bias=False),
            )

    def forward(self, latent, batch_index=None):
        if self.composition_batch_embedding is None:
            composition_latent = neighbor_latent = latent
        else:
            if batch_index is None:
                raise ValueError("Batch-aware niche decoding requires batch_index")
            code = batch_index.reshape(-1).long()
            composition_latent = latent + self.composition_batch_embedding(code)
            neighbor_latent = latent + self.neighbor_mean_batch_embedding(code)
        composition_logits = self.composition(composition_latent)
        if self.neighbor_expression_mode == "composition_gated":
            composition = torch.softmax(composition_logits, dim=-1)
            profiles = self.neighbor_mean(
                torch.cat([neighbor_latent, composition], dim=-1)
            ).reshape(-1, self.n_cell_types, self.neighbor_embedding_dim)
            neighbor_expression = (profiles * composition.unsqueeze(-1)).reshape(latent.shape[0], -1)
        else:
            neighbor_expression = self.neighbor_mean(neighbor_latent)
        return torch.cat([composition_logits, neighbor_expression], dim=-1)


def argument_value(flag: str, default: str | None = None) -> str | None:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except ValueError:
        return default


if __name__ == "__main__":
    base.SharedNicheDecoder = SeparateBatchHeadNicheDecoder
    base.main()

    # Record the monkey-patched architecture in the generated run metadata.
    out_root = Path(argument_value("--out_root", "results_shared_decoder"))
    data_tag = Path(argument_value("--data", "data.h5ad")).stem
    candidates = sorted((out_root / data_tag).glob("*/meta.json"), key=lambda p: p.stat().st_mtime)
    if candidates:
        meta_path = candidates[-1]
        meta = json.loads(meta_path.read_text())
        meta.update({
            "decoder_batch_parameterization": {
                "gene": "native scVI categorical decoder projector",
                "composition": "independent embedding(8) -> linear(latent width)",
                "neighbor_mean": "independent embedding(8) -> linear(latent width)",
                "shared_batch_key": meta.get("batch_key"),
            },
            "separate_decoder_batch_projectors": True,
        })
        meta_path.write_text(json.dumps(meta, indent=2))
