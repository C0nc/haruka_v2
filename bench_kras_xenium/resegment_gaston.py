#!/usr/bin/env python3
"""Redo GASTON domain segmentation properly from saved isodepth.

`get_isodepth_labels(..., num_domains=L)` throws KeyError when the DP optimum
needs fewer than L pieces: find_segments_from_dp backtracks past bucket 0 while
segments remain. So run the DP directly and take the largest L that resolves,
instead of silently falling back to quantile bins.
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, warnings; warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from gaston import dp_related

D = Path('/home/yancui/haruka_v2/bench_kras_xenium/gaston_scc')
CC = ('/home/yancui/haruka_v2/results_real_compgated_nb_final/mouselung_kras_xenium_clean_noC3/'
      'haruka_z_s_cellcharter_cc3_condition_only/')

names = pd.Index(np.load(CC + 'obs_names.npy', allow_pickle=True).astype(str))
h = ad.read_h5ad('/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad', backed='r')
rows = np.load(CC + 'source_rows.npy')
cid_of_row = h.obs['cell_id'].iloc[rows].astype(str).to_numpy(); h.file.close()
pos = pd.Series(np.arange(len(cid_of_row)), index=cid_of_row)
rep = np.load(CC + 'salient_spatial.npy')

out = {}
for f in sorted(glob.glob(str(D / 'gaston_JB_*.npz'))):
    sec = Path(f).stem.replace('gaston_', '')
    d = np.load(f, allow_pickle=True)
    iso = d['isodepth']; cids = d['cell_id'].astype(str)
    idx = pos.reindex(cids).to_numpy().astype(int)
    A = PCA(n_components=15, random_state=0).fit_transform(StandardScaler().fit_transform(rep[idx]))
    best = None
    for L in range(6, 1, -1):
        for nb in (50, 30, 20):
            try:
                be = np.linspace(iso.min(), iso.max() + 0.01, nb + 1)
                em, sm = dp_related.dp_bucketized(A.T, be, L, xcoords=iso)
                segs = dp_related.find_segments_from_dp(em, sm, L)
                bl = np.digitize(iso, be)
                lab = np.zeros(len(iso))
                for c, seg in enumerate(segs):
                    for s_ in seg:
                        lab[np.where(bl == s_ + 1)[0]] = c
                if len(np.unique(lab)) == L:
                    best = (L, nb, lab); break
            except Exception:
                continue
        if best: break
    if best is None:
        print(f'{sec}: DP resolved no L in 2..6'); continue
    L, nb, lab = best
    np.savez(D / f'gaston_{sec}_resegmented.npz', isodepth=iso, labels=lab,
             cell_id=cids, xy=d['xy'], n_domains=L, num_buckets=nb)
    sizes = pd.Series(lab).value_counts().sort_index()
    out[sec] = dict(n_domains=int(L), num_buckets=int(nb),
                    domain_sizes={int(k): int(v) for k, v in sizes.items()})
    print(f'{sec}: DP resolved {L} domains (num_buckets={nb}); sizes '
          f'{[int(v) for v in sizes.values]}', flush=True)

(D / 'resegment_summary.json').write_text(json.dumps(out, indent=2) + '\n')
print('\nwrote', D / 'resegment_summary.json')
