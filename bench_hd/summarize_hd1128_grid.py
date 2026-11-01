#!/usr/bin/env python3
"""One table over the HD MM-1128 fits: s scale, donor / cell-type share of z and s, and the HD-state tests (median over cell types)."""
import numpy as np, pandas as pd
from pathlib import Path
B = Path('/home/yancui/haruka_v2/bench_hd')
FITS = [('sym', 'batch', 'two-way 10'), ('oneway', 'batch', 'one-way 10'), ('batch_ow1', 'batch', 'one-way 1'), ('batch_ow30', 'batch', 'one-way 30'),
        ('batch_ow100', 'batch', 'one-way 100'), ('batch_tw1', 'batch', 'two-way 1'), ('donor_ow10', 'donor', 'one-way 10'), ('donor_ow30', 'donor', 'one-way 30')]
rows = []
for tag, bk, pen in FITS:
    d = B / ('hd1128_zs' if tag == 'sym' else f'hd1128_zs_{tag}')
    if not (d / 'gene_r2.csv').exists(): continue
    vp = pd.read_csv(d / 'variance_partition.csv', index_col=0); c = pd.read_csv(d / 'continuous_gt.csv'); inc = pd.read_csv(d / 'inclusions.csv'); g = pd.read_csv(d / 'gene_r2.csv')
    sc = np.load(d / 's_scale.npy') if (d / 's_scale.npy').exists() else [np.nan] * 4
    r = dict(batch_key=bk, penalty=pen, s_norm_ctrl=sc[0], s_norm_HD=sc[1], s_sd=sc[2], z_sd=sc[3],
             donor_z=vp.loc['slice_id', 'z'], donor_s=vp.loc['slice_id', 's'], celltype_z=vp.loc['cell_type', 'z'], celltype_s=vp.loc['cell_type', 's'])
    for gt in ['CAG', 'm_gamma', 'lipo']:
        t = c[c['gt'] == gt]; r[f'{gt}_z'] = t.z.median(); r[f'{gt}_s'] = t.s.median()
    for lab, k in [('intranuclear (Agg 1) vs none', 'agg1'), ('nearest extranuclear (Agg 0) vs none', 'agg0')]:
        t = inc[inc.test == lab]; r[f'{k}_z'] = t.z.median(); r[f'{k}_s'] = t.s.median()
    r['genes_s_gt_z'] = (g.s_minus_z > 0).mean()
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(B / 'hd1128_grid_summary.csv', index=False)
pd.set_option('display.width', 300); print(T.round(3).to_string(index=False))
