#!/usr/bin/env python3
"""End-to-end example for the spatially aware Haruka pipeline.

Runs out of the box on synthetic data, so it needs no external files::

    python run_haruka_spatial_example.py

The synthetic sections are built so that each of the three latents has a
distinct ground truth to recover, which is what makes the run a check rather
than a smoke test:

  layer   3 horizontal bands present in BOTH sections -> background
  lesion  a disc present only in the condition section -> salient
  niche   a cell has a 'sender' cell among its spatial neighbors -> spatial

The niche label is the decisive one. Senders are scattered at random,
independently of layer and lesion, and they express a ligand program. Crucially,
a cell exposed to a sender expresses NOTHING of its own in response, so its own
expression row carries exactly zero information about its niche label. The label
is knowable only by looking at other cells. A latent computed from a cell's own
expression therefore cannot recover it above chance no matter how well trained,
while the neighbor-attention branch can. That asymmetry is the whole claim, and
the probe table below is what tests it.

To run on the real arcseq data instead, point --real at the directory that
run_spatial_multiomics_arcseq.py uses::

    python run_haruka_spatial_example.py --real \
        --data_dir /mnt/HDD2B/Users/yancui/mouse_atlas/process_data \
        --out_dir  /mnt/HDD2B/Users/yancui/Haruka/outputs/arcseq_spatial

That path reuses load_and_prepare() from run_spatial_multiomics_arcseq.py, so it
expects the same four files (arcseq_{P21S3,LPC21S1}_{rna,atac}_adata.h5ad), and
treats P21S3 as control and LPC21S1 as the condition.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import anndata as ad
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from haruka_spatial import HarukaMultiomicsSpatial


def _programs(mask_list, n_feat, base, amp, rng):
    """Poisson counts from a base rate plus one additive block per program.

    Each entry is ``(cell_mask, feature_block)`` or ``(cell_mask, feature_block,
    amplitude)``. The ligand block is given a larger amplitude than the rest:
    only ~8% of cells are senders and the attention pools a weighted mean over
    k neighbors, so one sender among k is diluted k-fold and needs to be
    unmistakable to survive the pooling.
    """
    n = mask_list[0][0].shape[0]
    rate = np.full((n, n_feat), base, dtype=np.float64)
    for entry in mask_list:
        mask, block = entry[0], entry[1]
        block_amp = entry[2] if len(entry) > 2 else amp
        if mask.any():
            rate[np.ix_(np.flatnonzero(mask), block)] += block_amp
    lib = rng.gamma(shape=10.0, scale=0.1, size=(n, 1))
    return rng.poisson(rate * lib).astype(np.float32)


def make_synthetic(n_x=30, n_y=20, n_genes=60, n_peaks=80, sender_rate=0.08, seed=0):
    """Two spatial sections (control + condition) with paired RNA and ATAC."""
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(n_x), np.arange(n_y))
    coords = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    n_per = coords.shape[0]

    rna_parts, atac_parts, obs_parts, coord_parts = [], [], [], []
    for slice_id, condition in (("control_s1", "control"), ("disease_s1", "disease")):
        y = coords[:, 1]
        layer = np.digitize(y, [n_y / 3.0, 2.0 * n_y / 3.0])

        lesion = np.zeros(n_per, dtype=bool)
        if condition == "disease":
            centre = np.array([n_x * 0.65, n_y * 0.5])
            lesion = np.linalg.norm(coords - centre, axis=1) < min(n_x, n_y) * 0.28

        sender = rng.random(n_per) < sender_rate
        # niche is defined purely by the 8 surrounding grid positions. The cell's own
        # sender status is deliberately NOT part of the label and senders are drawn
        # i.i.d., so nothing in a cell's own row predicts its niche label.
        # Zero-padded rather than wrapped, so the label's neighborhood is exactly the
        # 8 surrounding positions. On a regular grid those are also the 8 Euclidean
        # nearest neighbors, so running the model at k=8 makes the graph it attends
        # over identical to the one that generated the label.
        pad = np.pad(sender.reshape(n_y, n_x).astype(np.int16), 1)
        neigh = np.zeros((n_y, n_x), dtype=np.int16)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                if dy == 1 and dx == 1:
                    continue
                neigh += pad[dy : dy + n_y, dx : dx + n_x]
        niche = neigh.ravel() > 0

        # No 'response' program: exposure to a sender changes nothing about the
        # exposed cell's own expression. The only trace of the niche is in the
        # neighbors' ligand block.
        rna = _programs(
            [
                (layer == 0, np.arange(0, 4)),
                (layer == 1, np.arange(4, 8)),
                (layer == 2, np.arange(8, 12)),
                (sender, np.arange(12, 24), 12.0),
                (lesion, np.arange(24, 36)),
            ],
            n_genes,
            base=0.5,
            amp=3.0,
            rng=rng,
        )
        atac = _programs(
            [
                (layer == 0, np.arange(0, 6)),
                (layer == 1, np.arange(6, 12)),
                (layer == 2, np.arange(12, 18)),
                (sender, np.arange(18, 36), 8.0),
                (lesion, np.arange(36, 54)),
            ],
            n_peaks,
            base=0.3,
            amp=2.0,
            rng=rng,
        )

        obs = {
            "condition": np.repeat(condition, n_per),
            "slice_id": np.repeat(slice_id, n_per),
            "layer": layer.astype(str),
            "lesion": lesion.astype(int).astype(str),
            "niche": niche.astype(int).astype(str),
            "sender": sender.astype(int).astype(str),
        }
        names = np.array([f"{slice_id}_{i:05d}" for i in range(n_per)])
        rna_parts.append(rna)
        atac_parts.append(atac)
        obs_parts.append((names, obs))
        coord_parts.append(coords.copy())

    import pandas as pd

    names = np.concatenate([p[0] for p in obs_parts])
    obs = pd.DataFrame(
        {k: np.concatenate([p[1][k] for p in obs_parts]) for k in obs_parts[0][1]},
        index=names,
    )
    for col in obs.columns:
        obs[col] = obs[col].astype("category")
    spatial = np.concatenate(coord_parts, axis=0)

    def build(blocks, prefix):
        X = np.concatenate(blocks, axis=0)
        a = ad.AnnData(X=X, obs=obs.copy())
        a.var_names = [f"{prefix}{i}" for i in range(X.shape[1])]
        a.obsm["spatial"] = spatial.copy()
        a.layers["counts"] = X.copy()
        a.layers["log"] = np.log1p(X)
        return a

    return build(rna_parts, "gene"), build(atac_parts, "peak")


TRUTHS = ("layer", "lesion", "niche")


def _probe(X, y, seed):
    """Balanced accuracy of a kNN probe, 3-fold stratified CV.

    Asks only 'is this information present in this latent', which is the question
    the factorization makes a claim about. A clustering-based score cannot answer
    it: with k fixed, whichever signal dominates variance takes the partition and
    every other signal reads as ~0 whether or not the latent encodes it.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y)) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    return float(
        cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy", n_jobs=1).mean()
    )


def probe_table(adata, seed):
    """Is each ground truth recoverable from each raw latent?"""
    truths = [t for t in TRUTHS if t in adata.obs.columns]
    reps = [
        (k, f"{k}_rep") for k in ("background", "salient", "spatial") if f"{k}_rep" in adata.obsm
    ]
    if not truths or not reps:
        return None
    rows = [f"{'':<12}" + "".join(f"{t:>10}" for t in truths)]
    for kind, key in reps:
        X = np.asarray(adata.obsm[key])
        cells = "".join(
            f"{_probe(X, np.asarray(adata.obs[t]).astype(str), seed):>10.3f}" for t in truths
        )
        rows.append(f"{kind:<12}{cells}")
    chance = "".join(
        f"{1.0 / max(len(np.unique(adata.obs[t])), 1):>10.3f}" for t in truths
    )
    rows.append(f"{'chance':<12}{chance}")
    return "\n".join(rows)


def ari_table(adata, tag):
    """ARI of each domain clustering against each ground-truth label."""
    truths = [t for t in TRUTHS if t in adata.obs.columns]
    kinds = [
        k
        for k in ("background", "salient", "spatial")
        if f"cluster_haruka_{k}{tag}" in adata.obs.columns
    ]
    if not truths or not kinds:
        return None
    rows = [f"{'':<12}" + "".join(f"{t:>10}" for t in truths)]
    for kind in kinds:
        pred = np.asarray(adata.obs[f"cluster_haruka_{kind}{tag}"]).astype(int)
        cells = "".join(
            f"{adjusted_rand_score(np.asarray(adata.obs[t]).astype(str), pred):>10.3f}"
            for t in truths
        )
        rows.append(f"{kind:<12}{cells}")
    return "\n".join(rows)


# The three independence penalties are calibrated to scales that differ by orders
# of magnitude, so a single --lam_mi default would silently switch the penalty off
# (or blow the loss up) when --reg_to_use changes.
_LAM_MI_DEFAULT = {"mi": 5.0, "mmd": 1000.0, "sw": 1.0}


def spatial_kwargs(args):
    """Spatial-branch settings shared by the synthetic and real paths.

    These land in ``run_rna`` / ``run_atac`` ``**kwargs``, which routes anything
    ``HarukaSpatial.setup_model`` accepts to the model and everything else to the
    trainer.
    """
    return dict(
        spatial_encoder=args.spatial_encoder,
        reg_to_use=args.reg_to_use,
        lam_mi=_LAM_MI_DEFAULT[args.reg_to_use] if args.lam_mi is None else args.lam_mi,
        kl_spatial_weight=args.kl_spatial_weight,
    )


def run_synthetic(args):
    print("Building synthetic paired RNA/ATAC sections ...")
    rna, atac = make_synthetic(
        n_x=args.n_x, n_y=args.n_y, n_genes=args.n_genes, n_peaks=args.n_peaks, seed=args.seed
    )
    print(f"  RNA {rna.shape}  ATAC {atac.shape}")
    for label in ("layer", "lesion", "niche"):
        counts = rna.obs[label].value_counts().to_dict()
        print(f"  {label:<7} {counts}")

    multi = HarukaMultiomicsSpatial(
        rna,
        atac,
        cov_gene_num=args.cov_num,
        cov_peak_num=args.cov_num,
        seed=args.seed,
        k_spatial=args.k_spatial,
    )

    common = dict(
        latent_dim=args.latent_dim,
        n_layer=args.n_layer,
        salient_cluster_num=2,
        background_cluster_number=3,
        spatial_cluster_num=2,
        condition_key="condition",
        condition_label="disease",
        batch_key="slice_id",
        max_epochs=args.max_epochs,
        n_spatial_latent=args.n_spatial_latent,
        cluster_method=args.cluster_method,
        plan_kwargs=dict(lr=args.lr),
        **spatial_kwargs(args),
    )

    print("\n=== RNA ===")
    multi.run_rna(gene_likelihood="poisson", **common)
    print("\n=== ATAC ===")
    multi.run_atac(**common)
    print("\n=== integrated ===")
    multi.concat_embeddings_and_cluster(
        multi_salient_cluster_num=2,
        multi_background_cluster_number=3,
        multi_spatial_cluster_num=2,
        cluster_method=args.cluster_method,
    )

    print("\n" + "=" * 68)
    print("Balanced accuracy of a kNN probe: is each ground truth present")
    print("in each latent? (condition-arm cells only, 3-fold CV)")
    print("=" * 68 + "\n")
    for name, adata in (("RNA", multi.output_rna), ("ATAC", multi.output_atac)):
        table = probe_table(adata, args.seed)
        if table:
            print(f"-- {name}")
            print(table)
            print()

    print("=" * 68)
    print("Adjusted Rand index of each domain clustering vs each ground truth")
    print("=" * 68 + "\n")
    for name, adata, tag in (
        ("RNA", multi.output_rna, ""),
        ("ATAC", multi.output_atac, ""),
        ("integrated", multi.output_multi, "_multi"),
    ):
        table = ari_table(adata, tag)
        if table:
            print(f"-- {name}")
            print(table)
            print()

    print("Cluster assignments written to obs:")
    for col in sorted(c for c in multi.output_multi.obs.columns if c.startswith("cluster_haruka")):
        n_used = len(np.unique(np.asarray(multi.output_multi.obs[col]).astype(int)))
        print(f"  {col:<38} {n_used} clusters used")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        multi.output_rna.write_h5ad(os.path.join(args.out_dir, "spatial_rna_output.h5ad"))
        multi.output_atac.write_h5ad(os.path.join(args.out_dir, "spatial_atac_output.h5ad"))
        multi.output_multi.write_h5ad(
            os.path.join(args.out_dir, "spatial_multiomics_output.h5ad")
        )
        print("\nWrote outputs to", args.out_dir)
    return multi


def sanitize_nan(adata, name):
    """Replace NaN with 0 in X and every layer.

    load_and_prepare concatenates the two sections with join='outer', and the
    sections do not share a feature set (16253 vs 16670 genes; 52550 vs 51741
    peaks). anndata fills the introduced columns of a DENSE X with NaN, so every
    cell ends up with a NaN somewhere: measured 3.8% of RNA entries and 17.3% of
    ATAC entries, which makes every row sum NaN and therefore every library size
    NaN.

    This is not cosmetic. Training the stock Haruka on that data leaves 22 of 39
    module parameters NaN, and it goes unnoticed because the encoders read
    nan_to_num(x) and so still emit finite embeddings. Filling with 0 is correct
    here: a gene absent from a section's feature list was not measured there, and
    0 is what the count likelihood already means by 'not observed'.
    """
    X = np.asarray(adata.X)
    n_nan = int(np.isnan(X).sum())
    if n_nan:
        print(
            f"  {name}: filling {n_nan} NaN entries ({np.isnan(X).mean():.2%} of X, "
            f"{int(np.isnan(X).any(1).sum())}/{adata.n_obs} cells affected) with 0"
        )
        adata.X = np.nan_to_num(X, nan=0.0)
    for key in list(adata.layers):
        L = np.asarray(adata.layers[key])
        if np.isnan(L).any():
            adata.layers[key] = np.nan_to_num(L, nan=0.0)
    return adata


def run_real(args):
    from run_spatial_multiomics_arcseq import load_and_prepare

    data_dir = Path(args.data_dir)
    files = {
        "rna_control": data_dir / "arcseq_P21S3_rna_adata.h5ad",
        "rna_condition": data_dir / "arcseq_LPC21S1_rna_adata.h5ad",
        "atac_control": data_dir / "arcseq_P21S3_atac_adata.h5ad",
        "atac_condition": data_dir / "arcseq_LPC21S1_atac_adata.h5ad",
    }
    for f in files.values():
        if not f.exists():
            raise FileNotFoundError(f"Data file not found: {f}")

    rna, atac = load_and_prepare(
        str(files["rna_control"]),
        str(files["rna_condition"]),
        str(files["atac_control"]),
        str(files["atac_condition"]),
        condition_label="LPC21S1",
        control_label="P21S3",
        max_cells_per_group=args.max_cells,
    )
    print("RNA shape:", rna.shape, "ATAC shape:", atac.shape)
    sanitize_nan(rna, "RNA")
    sanitize_nan(atac, "ATAC")

    multi = HarukaMultiomicsSpatial(
        rna,
        atac,
        cov_gene_num=args.cov_num_real,
        cov_peak_num=args.cov_num_real,
        seed=args.seed,
        k_spatial=args.k_spatial,
    )
    common = dict(
        latent_dim=args.latent_dim,
        n_layer=args.n_layer,
        salient_cluster_num=4,
        background_cluster_number=4,
        spatial_cluster_num=4,
        condition_key="condition",
        condition_label="LPC21S1",
        batch_key="slice_id",
        max_epochs=args.max_epochs,
        n_spatial_latent=args.n_spatial_latent,
        cluster_method=args.cluster_method,
        plan_kwargs=dict(lr=args.lr),
        **spatial_kwargs(args),
    )
    multi.run_rna(gene_likelihood="poisson", **common)
    multi.run_atac(**common)
    multi.concat_embeddings_and_cluster(
        multi_salient_cluster_num=4,
        multi_background_cluster_number=4,
        multi_spatial_cluster_num=4,
        cluster_method=args.cluster_method,
    )

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        multi.output_rna.write_h5ad(os.path.join(args.out_dir, "spatial_rna_output.h5ad"))
        multi.output_atac.write_h5ad(os.path.join(args.out_dir, "spatial_atac_output.h5ad"))
        multi.output_multi.write_h5ad(
            os.path.join(args.out_dir, "spatial_multiomics_output.h5ad")
        )
        print("Wrote outputs to", args.out_dir)
    return multi


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", action="store_true", help="Run on arcseq data instead of synthetic")
    p.add_argument("--data_dir", default="/mnt/HDD2B/Users/yancui/mouse_atlas/process_data")
    p.add_argument("--out_dir", default=None, help="Write h5ad outputs here (optional)")
    p.add_argument("--max_cells", type=int, default=None, help="--real only: subset per group")
    p.add_argument("--n_x", type=int, default=30)
    p.add_argument("--n_y", type=int, default=20)
    p.add_argument("--n_genes", type=int, default=60)
    p.add_argument("--n_peaks", type=int, default=80)
    p.add_argument("--cov_num", type=int, default=8, help="synthetic COVET HVG count")
    p.add_argument("--cov_num_real", type=int, default=32, help="--real COVET HVG count")
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--n_spatial_latent", type=int, default=10, help="0 disables the spatial branch")
    p.add_argument(
        "--spatial_encoder",
        default="gat",
        choices=["gat", "setattn"],
        help="'gat' is SIMVI's GATv2 over the neighbors' encoder means; 'setattn' is the "
        "lightweight stack-style axial set-attention encoder over their expression",
    )
    p.add_argument(
        "--reg_to_use",
        default="mi",
        choices=["mi", "mmd", "sw"],
        help="independence penalty; 'sw' is the O(n log n) sliced Wasserstein",
    )
    p.add_argument(
        "--lam_mi",
        type=float,
        default=None,
        help="weight on the independence penalty. Defaults per --reg_to_use "
        "(mi: 5, mmd: 1000, sw: 1), which are NOT interchangeable scales",
    )
    p.add_argument(
        "--kl_spatial_weight",
        type=float,
        default=0.01,
        help="KL weight on the spatial latent. SIMVI's 0.01 is a 100x down-weight "
        "that keeps the latent expressive; raise it toward 1.0 if the spatial latent "
        "out-competes the intrinsic ones for the reconstruction and collapses them",
    )
    # 8 on the synthetic grid because the 8 Euclidean nearest neighbors there are
    # exactly the 8 surrounding positions that define the niche label, so the graph
    # the model attends over is the one that generated the label.
    #
    # On real tissue the class default of 10 (SIMVI's value) is fine: real arrays
    # are lattices whose distance shells hold 4, 4 and 4 sites, and the graph
    # builder refuses to admit a partial shell, so k=10 resolves to the two
    # complete shells there and to a genuine 10 on continuous imaging coordinates.
    p.add_argument("--k_spatial", type=int, default=8)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--max_epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--cluster_method",
        default="kmeans",
        choices=["kmeans", "minibatch_kmeans", "gmm"],
        help="'gmm' (scaled + PCA-reduced) is slower but better on anisotropic domains",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    if args.real:
        run_real(args)
    else:
        run_synthetic(args)
    print("\nDone.")


if __name__ == "__main__":
    main()
