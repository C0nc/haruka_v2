"""Label-free linear cleanup of paired Haruka intrinsic/spatial slots.

This is a diagnostic for identifiability, not a truth-guided transform. It removes
only the component of one learned slot linearly predictable from its paired GAT
slot, using all cells and no simulator labels. If this improves selectivity, the
remaining error is shared latent subspace rather than missing signal.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os

import numpy as np


SLOTS = ("background", "salient", "spatial", "salient_spatial")


def residualize(target: np.ndarray, nuisance: np.ndarray, alpha: float,
                chunk_rows: int = 8192) -> np.ndarray:
    """Return target minus its ridge-linear prediction from nuisance.

    Prediction is chunked so the same transform is practical for CINEMA-OT's
    900-column spatial slot; no full float64 target or prediction copy is made.
    """
    x = np.asarray(nuisance, dtype=np.float64)
    y = np.asarray(target)
    xm, xs = x.mean(0), x.std(0)
    ym = y.mean(0, dtype=np.float64)
    xs[xs < 1e-8] = 1.0
    xz = (x - xm) / xs
    gram = xz.T @ xz
    scale = max(1.0, np.trace(gram) / max(1, gram.shape[0]))
    rhs = xz.T @ y
    rhs -= np.outer(xz.sum(0), ym)
    coef = np.linalg.solve(
        gram + alpha * scale * np.eye(gram.shape[0]), rhs
    )
    cleaned = np.empty(y.shape, dtype=np.float32)
    for start in range(0, len(y), chunk_rows):
        stop = min(len(y), start + chunk_rows)
        cleaned[start:stop] = y[start:stop] - (ym + xz[start:stop] @ coef)
    return cleaned


def transform(in_dir: str, out_dir: str, pairs: list[tuple[str, str]], alpha: float):
    with open(os.path.join(in_dir, "meta.json")) as fh:
        meta = json.load(fh)
    n_blocks = int(meta.get("n_layer", 3)) + 1
    arrays = {
        name: np.load(os.path.join(in_dir, f"{name}.npy"), mmap_mode="r")
        for name in SLOTS
    }
    own = {name: arr[:, : arr.shape[1] // n_blocks] for name, arr in arrays.items()}
    cleaned_own = {}

    for target, nuisance in pairs:
        if target not in own or nuisance not in own:
            raise KeyError(f"unknown orthogonalization pair {target}:{nuisance}")
        if len(own[target]) != len(own[nuisance]):
            raise ValueError(f"row mismatch for {target}:{nuisance}")
        width = own[target].shape[1]
        cleaned = residualize(own[target], own[nuisance], alpha)
        cleaned_own[target] = cleaned
        own[target] = cleaned

    for arr in arrays.values():
        try:
            arr._mmap.madvise(mmap.MADV_DONTNEED)
        except (AttributeError, ValueError):
            pass

    os.makedirs(out_dir, exist_ok=True)
    for name, arr in arrays.items():
        replacement = cleaned_own.get(name)
        dst = np.lib.format.open_memmap(
            os.path.join(out_dir, f"{name}.npy"), mode="w+",
            dtype=np.float32, shape=arr.shape,
        )
        # With n_layer=0 the cleaned own block is the whole array. Do not first
        # page the original wide target into memory only to overwrite it.
        if replacement is not None and replacement.shape == arr.shape:
            for start in range(0, len(arr), 8192):
                stop = min(len(arr), start + 8192)
                dst[start:stop] = replacement[start:stop]
        else:
            for start in range(0, len(arr), 8192):
                stop = min(len(arr), start + 8192)
                dst[start:stop] = arr[start:stop]
            if replacement is not None:
                dst[:, : replacement.shape[1]] = replacement
        del dst
    obs_path = os.path.join(in_dir, "obs_names.npy")
    if os.path.exists(obs_path):
        np.save(os.path.join(out_dir, "obs_names.npy"), np.load(obs_path, allow_pickle=True),
                allow_pickle=True)
    meta.update(
        postprocess="label_free_ridge_orthogonalization",
        orthogonalize_pairs=[list(pair) for pair in pairs],
        orthogonalize_alpha=float(alpha),
        source_embedding_dir=in_dir,
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument(
        "--pairs", nargs="+", default=["background:spatial", "salient_spatial:spatial"],
        help="target:nuisance pairs applied in order",
    )
    args = ap.parse_args()
    pairs = [tuple(spec.split(":", 1)) for spec in args.pairs]
    transform(args.in_dir, args.out_dir, pairs, args.alpha)


if __name__ == "__main__":
    main()
