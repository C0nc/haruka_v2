#!/usr/bin/env python3
"""Figure 2 (maps): what each method actually recovers, in tissue.

Reproduces the scorer EXACTLY (raw spatial embedding -> KMeans, k = #GT classes,
random_state 0, n_init 20) and asserts every ARI matches the published value
before drawing.  Predicted clusters are Hungarian-matched to ground truth only to
pick colours, so a region drawn the same colour as the truth is the cluster that
best overlaps it; the ARI is computed on the raw labels, not the matched ones.
"""
from __future__ import annotations
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari
from scipy.optimize import linear_sum_assignment
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch
import matplotlib.patheffects as pe

B = Path(f'{_HR}/bench_mouse_brain_scviva')
OUT = Path(f'{_HR}/figures_paper')
DATA = B / 'data_composition_pairs_all'      # verified: reproduces both bg and salient ARIs exactly
BG_COL = ['#0072B2','#E69F00','#009E73','#CC79A7','#56B4E9','#D55E00','#882255']  # validated, all-pairs
SAL_COL = ['#2a78d6','#eb6834','#1baf7a']                                          # validated, all-pairs
CTRL = '#dfe4e9'
INK, INK2, MUTED = '#16202b', '#4d5966', '#8a96a3'

def emb_dir(method, sec):
    if method == 'default': return B/f'results_shared_full_niche_compgated_lambda10_cc/mouse_brain_{sec}_paired/haruka_z_s_cellcharter_seed0'
    if method == 'gated25': return B/f'results_shared_full_niche_compgated_lambda25_cc/mouse_brain_{sec}_paired/haruka_z_s_cellcharter_seed0'
    if method == 'indep25': return B/f'results_shared_full_niche_lambda25_cc/mouse_brain_{sec}_paired/haruka_z_s_cellcharter_seed0'
    root = 'results_direct_kmeans' if sec == 'A8' else 'results_cross_section'
    leaf = {'cinema':('cinema_minibatch','cinemaot_minibatch_b2048_e3_seed0'),
            'expr':('baselines','pca_2x2_seed0'), 'scvi':('baselines','cellcharter_2x2_seed0'),
            'cvi':('baselines','contrastivevi_2x2_seed0')}[method]
    return B/f'{root}/{leaf[0]}/mouse_brain_{sec}_paired/{leaf[1]}'

METHODS = [('default','Haruka (default, λ=10)'), ('gated25','Haruka gated λ=25'),
           ('indep25','Haruka independent λ=25'), ('cinema','CINEMA-OT + CellCharter'),
           ('expr','Expression + aggregation'), ('scvi','scVI + CellCharter'),
           ('cvi','ContrastiveVI + CellCharter')]

def match(truth, pred, classes):
    """Map each predicted cluster to the GT class it overlaps most (for colour only)."""
    pc = np.unique(pred)
    M = np.array([[np.sum((pred == p) & (truth == c)) for c in classes] for p in pc])
    r, c = linear_sum_assignment(-M)
    lut = {pc[i]: classes[j] for i, j in zip(r, c)}
    return np.array([lut.get(p, classes[0]) for p in pred])

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument('--section', default='B5'); a_ = ap.parse_args()
    sec = a_.section
    cache = OUT / f'fig2_maps_cache_{sec}.npz'
    a = ad.read_h5ad(DATA / f'mouse_brain_{sec}_paired.h5ad', backed='r')
    allnames = pd.Index(a.obs_names.astype(str))
    obs_all, xy_all = a.obs, np.asarray(a.obsm['spatial'], dtype=float); a.file.close()

    if cache.exists():
        Z = dict(np.load(cache, allow_pickle=True)); print('loaded cache', flush=True)
    else:
        Z = {}
        for key, _ in METHODS:
            d = emb_dir(key, sec)
            names = np.load(d/'obs_names.npy', allow_pickle=True).astype(str)
            pos = allnames.get_indexer(names); assert (pos >= 0).all()
            o = obs_all.iloc[pos]
            yb = o.gt_background_subregion.astype(str).to_numpy()
            ys = o.gt_sal_domain.astype(str).to_numpy()
            dis = o.condition.astype(str).to_numpy() == 'disease'
            pb = KMeans(len(np.unique(yb)), random_state=0, n_init=20).fit_predict(np.load(d/'spatial.npy'))
            ps = KMeans(len(np.unique(ys[dis])), random_state=0, n_init=20).fit_predict(np.load(d/'salient_spatial.npy')[dis])
            ref = json.load(open(d/'direct_kmeans_ari.json'))['scores']
            rb, rs = ari(yb, pb), ari(ys[dis], ps)
            # KMeans is not bit-reproducible across BLAS thread counts; require agreement to 1e-3
            db, ds = rb - ref['background_subregion']['ari'], rs - ref['salient_domain']['ari']
            assert abs(db) < 1e-3 and abs(ds) < 1e-3, (key, rb, rs, ref)
            Z[f'{key}_pos'] = pos; Z[f'{key}_pb'] = pb; Z[f'{key}_ps'] = ps; Z[f'{key}_dis'] = dis
            Z[f'{key}_ari'] = np.array([rb, rs])
            print(f'  {key:8s} bg {rb:.4f} (d {db:+.1e})  sal {rs:.4f} (d {ds:+.1e})  match published', flush=True)
        np.savez(cache, **Z)

    cond = obs_all.condition.astype(str).to_numpy()
    gtb = obs_all.gt_background_subregion.astype(str).to_numpy()
    gts = obs_all.gt_sal_domain.astype(str).to_numpy()
    bcls = sorted(np.unique(gtb)); scls = sorted(np.unique(gts[cond == 'disease']))
    FIXED = {'CC/ACO':'#0072B2','CTX_L1/MEN':'#E69F00','CTX_L2/3':'#009E73','CTX_L4/5/6':'#882255',
             'STR_CP/ACB':'#56B4E9','STR_LS/NDB':'#D55E00','VEN':'#CC79A7'}   # same validated 7, re-assigned
    bmap = {c: FIXED.get(c, BG_COL[i]) for i, c in enumerate(bcls)}; smap = {c: SAL_COL[i] for i, c in enumerate(scls)}
    ctl, dis = cond == 'control', cond == 'disease'

    rows = [('gt', 'Ground truth')] + METHODS
    RH = 2.62; FH = RH * len(rows) + .95
    fig = plt.figure(figsize=(11.6, FH))
    gs = gridspec.GridSpec(len(rows), 3, wspace=.03, hspace=.16, left=.16, right=.995,
                           top=1 - .42 / FH, bottom=.5 / FH)
    heads = ['Background subregions · control arm', 'Background subregions · disease arm',
             'Salient domains · disease arm']

    def draw(ax, xy, colors, back=None):
        if back is not None:
            ax.scatter(back[:, 0], back[:, 1], s=.25, c=CTRL, linewidths=0, rasterized=True)
        ax.scatter(xy[:, 0], xy[:, 1], s=.25, c=colors, linewidths=0, rasterized=True)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)

    for r, (key, label) in enumerate(rows):
        if key == 'gt':
            colb = np.array([bmap[c] for c in gtb]); cols = np.array([smap.get(c, CTRL) for c in gts])
            pbc, pbd, psd = colb[ctl], colb[dis], cols[dis]
            sub = ''
        else:
            pos = Z[f'{key}_pos']; dm = Z[f'{key}_dis']
            yb = gtb[pos]; mb = match(yb, Z[f'{key}_pb'], bcls)
            ys = gts[pos][dm]; ms = match(ys, Z[f'{key}_ps'], scls)
            cb_ = np.array([bmap[c] for c in mb])
            cnd = cond[pos]
            # place each predicted label back at its cell's coordinate
            full_b = np.full(len(gtb), CTRL, dtype=object); full_b[pos] = cb_   # unscored cells -> neutral
            full_s = np.full(len(gtb), CTRL, dtype=object)
            full_s[pos[dm]] = np.array([smap[c] for c in ms])
            pbc, pbd, psd = full_b[ctl], full_b[dis], full_s[dis]
            rb, rs = Z[f'{key}_ari']
            sub = f'bg ARI {rb:.3f}   ·   salient ARI {rs:.3f}'
        for c, (xy, colors) in enumerate([(xy_all[ctl], pbc), (xy_all[dis], pbd), (xy_all[dis], psd)]):
            ax = fig.add_subplot(gs[r, c]); draw(ax, xy, list(colors))
            if c == 0: row_ax = ax
            if r == 0: ax.set_title(heads[c], fontsize=9.6, color=INK, pad=6)
            if key == 'gt' and c == 2:
                for cl in scls:
                    m = gts[dis] == cl
                    pts = xy[m]; med = np.median(pts, 0); lx, ly = pts[np.argmin(((pts - med) ** 2).sum(1))]
                    ax.text(lx, ly, cl.replace('domain_', 'D'),
                            fontsize=8, ha='center', va='center', color='white', fontweight='bold',
                            path_effects=[pe.withStroke(linewidth=2, foreground='#111')])
        p0 = row_ax.get_position()
        if key == 'gt':
            fig.text(p0.x0 - .006, p0.y1 - .002, 'Ground truth', ha='right', va='top', fontsize=9,
                     color=INK, fontweight='bold')
            yy = p0.y1 - .024
            for cl in bcls:                       # region key beside the map it describes
                fig.patches.append(plt.Rectangle((p0.x0 - .118, yy - .0035), .012, .007,
                                   transform=fig.transFigure, color=bmap[cl], zorder=5))
                fig.text(p0.x0 - .102, yy, cl, ha='left', va='center', fontsize=7.4, color=INK)
                yy -= .0128
        else:
            fig.text(p0.x0 - .006, (p0.y0 + p0.y1) / 2, label + f'\n{sub}', ha='right', va='center',
                     fontsize=8.6, color=INK, fontweight='bold' if key == 'default' else 'normal',
                     linespacing=1.5)

    leg = [Patch(color=smap[c], label=c.replace('domain_', 'salient domain D')) for c in scls] + \
          [Patch(color=CTRL, label='control arm (no salient domain)')]
    fig.legend(handles=leg, ncol=4, frameon=False, fontsize=8, loc='lower center', bbox_to_anchor=(.58, 0))
    fig.suptitle(f'Figure 2 — Recovered domains per method, section {sec} (median section for Haruka)',
                 x=.02, y=.999, ha='left', fontsize=12.5, color=INK, fontweight='bold')
    p = OUT / f'fig2_benchmark_maps_{sec}.png'
    fig.savefig(p, dpi=220, bbox_inches='tight'); fig.savefig(str(p).replace('.png', '.pdf'), bbox_inches='tight')
    print('wrote', p, flush=True)

if __name__ == '__main__':
    main()
