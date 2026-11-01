#!/usr/bin/env python3
"""Expression baselines for the whole-slice plaque test: the 34 input proteins, per cell and
3-hop kNN-aggregated (same aggregate() as the CC mechanism test), on the same disease cells."""
import sys, warnings
from pathlib import Path
import numpy as np, anndata as ad
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/yancui/haruka_v2/bench_codex_cns_2026')
from evaluate_channelnorm_ab import aggregate
from evaluate_wholeslice import held_out_auroc, ROOT

for v in sys.argv[1:]:
    a = ad.read_h5ad(Path(__file__).resolve().parent / f'codex_wholeslice_{v}.h5ad')
    src = np.load(ROOT / f'codex_wholeslice_{v}' / 'haruka_z_s_cellcharter_cc3_condition_only' / 'source_rows.npy')
    obs = a.obs.iloc[src]; X = np.asarray(a.X[src], dtype=np.float32)
    donor = obs.donor_id.astype(str).to_numpy(); region = obs.region.astype(str).to_numpy()
    ct = obs.cell_type.astype(str).to_numpy(); dist = obs.dist_dense_abeta_um.to_numpy(float)
    Xa = aggregate(X, a.obsm['spatial'][src].astype(float), donor)
    allm = np.isfinite(dist); gm = allm & (region == 'GM')
    print(f'== {v} expression baselines ==')
    for r in (10, 30):
        y = (dist <= r).astype(int)
        for k, x in [('protein (34)', X), ('protein 3-hop agg', Xa)]:
            print(f'  <= {r} um {k:18s} all {held_out_auroc(x, y, donor, allm)[0]:.3f}   GM only {held_out_auroc(x, y, donor, gm)[0]:.3f}')
    y = (dist <= 30).astype(int)
    for t in sorted(set(ct)):
        print(f'  <= 30 um GM {t:16s} protein {held_out_auroc(X, y, donor, gm & (ct == t))[0]:.3f}')
