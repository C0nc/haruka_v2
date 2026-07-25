#!/usr/bin/env python3
"""Train the chunked axial contrastive model on the synthetic benchmark.

Evaluated with the same kNN probe and ARI used by run_haruka_spatial_example.py, so the
numbers sit directly beside Haruka's:

    Haruka background   layer 0.960   lesion 0.046   niche 0.522
    Haruka salient      layer 0.041   lesion 0.968   niche 0.507

The open question is `niche`: Haruka's MLP encoder sees one cell at a time and is at chance
there by construction. Chunked attention lets a cell see its neighbours, so it is the one
column where this model could differ.
"""

import argparse
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from run_haruka_spatial_example import make_synthetic
from stack_contrast import ChunkLoader, StackContrast, spatial_chunks

TRUTHS = ("layer", "lesion", "niche")


def probe(X, y, seed=0, k=15):
    """Balanced accuracy of a kNN probe, 3-fold CV. Chance is 1/n_classes."""
    X = StandardScaler().fit_transform(np.nan_to_num(X))
    y = np.asarray(y).astype(str)
    if len(np.unique(y)) < 2:
        return float("nan")
    out = []
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y):
        m = KNeighborsClassifier(n_neighbors=k).fit(X[tr], y[tr])
        out.append(balanced_accuracy_score(y[te], m.predict(X[te])))
    return float(np.mean(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--chunk_size", type=int, default=32)
    p.add_argument("--batch_chunks", type=int, default=8)
    p.add_argument("--n_latent", type=int, default=32)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_tokens", type=int, default=16)
    p.add_argument("--token_dim", type=int, default=32)
    p.add_argument("--sw_weight", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle_chunks", action="store_true",
                   help="ablation: randomise chunk membership, destroying spatial context")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    out = make_synthetic(seed=args.seed)
    rna = out[0] if isinstance(out, tuple) else out
    counts = np.asarray(rna.X, dtype=np.float32)
    coords = np.column_stack([rna.obs["array_col"], rna.obs["array_row"]]).astype(float) \
        if "array_col" in rna.obs else np.asarray(rna.obsm["spatial"], dtype=float)
    group = rna.obs["slice_id"].astype(str).values
    is_control = (rna.obs["condition"].astype(str).values == "control")
    print(f"cells {rna.n_obs} | genes {rna.n_vars} | control {is_control.sum()} | "
          f"perturbed {(~is_control).sum()} | device {dev}")

    chunks = spatial_chunks(coords, group, chunk_size=args.chunk_size, seed=args.seed)
    if args.shuffle_chunks:
        flat = np.concatenate(chunks)
        np.random.default_rng(args.seed).shuffle(flat)
        chunks = [flat[i:i + args.chunk_size] for i in range(0, len(flat), args.chunk_size)]
        print("ABLATION: chunk membership randomised (no spatial context)")
    print(f"chunks {len(chunks)} | size {args.chunk_size}")

    model = StackContrast(
        n_genes=rna.n_vars, n_latent=args.n_latent, n_tokens=args.n_tokens,
        token_dim=args.token_dim, n_layers=args.n_layers, sw_weight=args.sw_weight,
    ).to(dev)
    n_par = sum(p_.numel() for p_ in model.parameters())
    print(f"parameters {n_par:,}")

    loader = ChunkLoader(counts, is_control, chunks, batch_chunks=args.batch_chunks, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    t0 = time.time()
    model.train()
    for ep in range(args.epochs):
        tot = nb = 0.0
        for X, C, P, _ in loader:
            opt.zero_grad()
            o = model(X, C, P)
            o["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(o["loss"]); nb += 1
        if ep % 50 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:4d}  loss {tot / max(nb, 1):10.2f}  recon {float(o['recon']):10.2f}"
                  f"  sw {float(o['sw']):.4f}")
    print(f"trained in {time.time() - t0:.1f}s")

    # inference over every cell, in the same chunking
    model.eval()
    Z = np.zeros((rna.n_obs, args.n_latent), np.float32)
    S = np.zeros((rna.n_obs, args.n_latent), np.float32)
    with torch.no_grad():
        for X, C, P, I in ChunkLoader(counts, is_control, chunks,
                                      batch_chunks=args.batch_chunks, device=dev, shuffle=False):
            z, s = model.embed(X, C, P)
            keep = (~P).cpu().numpy()
            idx = I.cpu().numpy()
            Z[idx[keep]] = z.cpu().numpy()[keep]
            S[idx[keep]] = s.cpu().numpy()[keep]

    tgt = ~is_control  # evaluate on the perturbed arm, as Haruka does
    print("\n" + "=" * 62)
    print("kNN probe, balanced accuracy (perturbed arm, 3-fold CV)")
    print("=" * 62)
    hdr = f"{'':12s}" + "".join(f"{t:>10}" for t in TRUTHS)
    print(hdr)
    for name, E in (("background", Z), ("salient", S)):
        print(f"{name:12s}" + "".join(
            f"{probe(E[tgt], rna.obs[t].values[tgt], args.seed):>10.3f}" for t in TRUTHS))
    print(f"{'chance':12s}" + "".join(
        f"{1.0 / max(len(np.unique(rna.obs[t].values[tgt])), 1):>10.3f}" for t in TRUTHS))

    print("\n" + "=" * 62)
    print("ARI of KMeans domains vs ground truth")
    print("=" * 62)
    print(hdr)
    for name, E, k in (("background", Z, 3), ("salient", S, 2)):
        lab = KMeans(k, n_init=10, random_state=args.seed).fit_predict(
            StandardScaler().fit_transform(E[tgt]))
        print(f"{name:12s}" + "".join(
            f"{adjusted_rand_score(rna.obs[t].values[tgt].astype(str), lab):>10.3f}"
            for t in TRUTHS))

    print("\nreference -- Haruka on the same data:")
    print("background     0.960     0.046     0.522   (probe)")
    print("salient        0.041     0.968     0.507   (probe)")


if __name__ == "__main__":
    main()
