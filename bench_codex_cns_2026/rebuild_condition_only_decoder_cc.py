#!/usr/bin/env python3
"""Build condition-only 3-hop CC on the DECODER niche objects, not the raw latents.

The shipped `rebuild_condition_only_salient_cc.py` aggregates the raw 32-d salient
latent `s`, so s-CC is a spatial low-pass filter on `s`.  The model also emits the
objects the report actually describes:

    spatial.npy          (110-d) = D_niche(z, 0)            background niche prediction
    salient_spatial.npy  (110-d) = D_niche(z, s) - D_niche(z, 0)   counterfactual shift
    (110 = 11 cell-type composition + 99 cell-type-specific neighbour-mean)

This aggregates those instead, so the spatial slots hold a niche *shift* rather
than a smoothed latent.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import anndata as ad, numpy as np, pandas as pd

ROOT = Path('/home/yancui/haruka_v2')
sys.path.insert(0, str(ROOT / 'bench_merfish'))
from common import save_embeddings                      # noqa: E402
from run_baselines_2x2 import _cc_aggregate             # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--condition-key', default='condition')
    p.add_argument('--condition-value', default='disease')
    p.add_argument('--slice-key', default='donor_id')
    args = p.parse_args()

    source = ad.read_h5ad(args.data, backed='r')
    obs = source.obs[[args.condition_key, args.slice_key]].copy()
    obs.index = obs.index.astype(str)
    spatial_xy = np.asarray(source.obsm['spatial'], dtype=np.float32)
    source.file.close()

    names = pd.Index(np.load(args.input_dir / 'obs_names.npy', allow_pickle=True).astype(str))
    rows = obs.index.get_indexer(names)
    if (rows < 0).any():
        raise ValueError('latent rows do not align with the source data')
    keep = obs.iloc[rows][args.condition_key].astype(str).eq(args.condition_value).to_numpy()
    kept_rows, kept_names = rows[keep], names.to_numpy()[keep]

    bg = np.load(args.input_dir / 'spatial.npy').astype(np.float32)[keep]
    sal = np.load(args.input_dir / 'salient_spatial.npy').astype(np.float32)[keep]
    print(f'per-cell decoder objects: background {bg.shape}, salient shift {sal.shape}', flush=True)

    graph = ad.AnnData(
        X=np.zeros((keep.sum(), 1), dtype=np.float32),
        obs=pd.DataFrame({'slice_id': obs.iloc[kept_rows][args.slice_key].astype(str).to_numpy()},
                         index=kept_names))
    graph.obs['slice_id'] = pd.Categorical(graph.obs['slice_id'])
    graph.obsm['spatial'] = spatial_xy[kept_rows]

    bg_cc = _cc_aggregate(graph, bg, 'decoder_background')
    sal_cc = _cc_aggregate(graph, sal, 'decoder_salient')
    print(f'aggregated: background {bg_cc.shape}, salient {sal_cc.shape}', flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_embeddings(
        args.output_dir, background=bg, salient=sal, spatial=bg_cc, salient_spatial=sal_cc,
        obs_names=kept_names,
        method='decoder_niche_cellcharter_condition_only',
        note=('intrinsic slots hold the per-cell decoder niche objects '
              '(background D_niche(z,0); salient D_niche(z,s)-D_niche(z,0)); '
              'spatial slots hold their symmetric three-hop CC aggregation, own block removed'),
        base_dim=int(bg.shape[1]), n_hops=3, runtime_s=0.0, n_layer=0,
        source_result=str(args.input_dir.resolve()),
        condition_key=args.condition_key, condition_value=args.condition_value,
        sample_key=args.slice_key)
    np.save(args.output_dir / 'source_rows.npy', kept_rows.astype(np.int64))
    print('done', flush=True)


if __name__ == '__main__':
    main()
