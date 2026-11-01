#!/usr/bin/env python3
"""Figure 2 — benchmark on the paired mouse-brain simulation.

Claim: Haruka recovers background anatomy and condition-specific salient domains
together; methods that model only one of the two fail on the other.

Panels (each also saved as its own SVG in this folder):
  a  ground truth, section B5 (median section for Haruka): background subregions,
     salient domains (disease arm)
  b  recovered maps, one SVG per method (main figure shows 4; all 5 are written)
  c  background-subregion ARI, n = 5 section pairs
  d  salient-domain ARI, n = 5 section pairs
  e  paired per-section difference, Haruka minus each comparator

Scoring (identical for every method): raw saved spatial embedding -> KMeans,
k = number of planted classes, random_state 0, n_init 20.  Cached clusterings were
checked against the published ARIs (max |deviation| 4.4e-4) before caching.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
from scipy.optimize import linear_sum_assignment
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paperstyle as ps  # noqa: E402
ps.setup()

ROOT = Path('/home/yancui/haruka_v2')
DATA = ROOT / 'bench_mouse_brain_scviva/data_composition_pairs_all/mouse_brain_B5_paired.h5ad'
CACHE = ROOT / 'figures_paper/fig2_maps_cache_B5.npz'
PERSEC = ROOT / 'figures_paper/fig2_benchmark_per_section.csv'
OUT = HERE

METHODS = [  # key, csv name, short label, family
    ('default', 'Haruka λ=10 composition-gated', 'Haruka', 'haruka'),
    ('cinema', 'Mini-batch CINEMA-OT + CellCharter', 'CINEMA-OT + CC', 'baseline'),
    ('expr', 'Expression + neighborhood aggregation', 'Expr. + aggregation', 'baseline'),
    ('scvi', 'scVI + CellCharter', 'scVI + CC', 'baseline'),
    ('cvi', 'ContrastiveVI z/s + CellCharter', 'ContrastiveVI + CC', 'baseline'),
]
MAIN_MAPS = ['default', 'cinema', 'expr', 'scvi']   # one per failure mode; cvi -> supplement
LAB = {k: lab for k, _, lab, _ in METHODS}
FAM = {k: fam for k, _, _, fam in METHODS}
CSVNAME = {k: n for k, n, _, _ in METHODS}

# ---------------------------------------------------------------- data
a = ad.read_h5ad(DATA, backed='r')
obs, XY = a.obs, np.asarray(a.obsm['spatial'], dtype=float)
a.file.close()
COND = obs.condition.astype(str).to_numpy()
GTB = obs.gt_background_subregion.astype(str).to_numpy()
GTS = obs.gt_sal_domain.astype(str).to_numpy()
DIS = COND == 'disease'
BCLS = sorted(np.unique(GTB))
SCLS = sorted(np.unique(GTS[DIS]))
assert set(BCLS) == set(ps.REGION), BCLS
DCOL = {c: ps.DOMAIN[i] for i, c in enumerate(SCLS)}
Z = dict(np.load(CACHE, allow_pickle=True))
D = pd.read_csv(PERSEC)
D = D[D.method.isin([n for _, n, _, _ in METHODS])].reset_index(drop=True)
assert (D.groupby('method').size() == 5).all() and D.method.nunique() == len(METHODS)


def match(truth, pred, classes):
    """Colour each predicted cluster as the truth class it overlaps most (colour only)."""
    pc = np.unique(pred)
    M = np.array([[np.sum((pred == p) & (truth == c)) for c in classes] for p in pc])
    r, c = linear_sum_assignment(-M)
    lut = {pc[i]: classes[j] for i, j in zip(r, c)}
    return np.array([lut.get(p, classes[0]) for p in pred])


def method_colours(key):
    """Per-cell colours on the DISEASE arm for one method (background, salient)."""
    pos, dm = Z[f'{key}_pos'], Z[f'{key}_dis']
    bg = np.full(len(GTB), ps.NO_DOMAIN, dtype=object)
    bg[pos] = [ps.REGION[c] for c in match(GTB[pos], Z[f'{key}_pb'], BCLS)]
    sal = np.full(len(GTB), ps.NO_DOMAIN, dtype=object)
    sal[pos[dm]] = [DCOL[c] for c in match(GTS[pos][dm], Z[f'{key}_ps'], SCLS)]
    return bg[DIS], sal[DIS], Z[f'{key}_ari']


# ---------------------------------------------------------------- panel drawers
def draw_map(ax, colours):
    xy = XY[DIS]
    ax.scatter(xy[:, 0], xy[:, 1], s=0.12, c=list(colours), linewidths=0, rasterized=True)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


def draw_gt(ax_b, ax_s):
    draw_map(ax_b, [ps.REGION[c] for c in GTB[DIS]])
    draw_map(ax_s, [DCOL[c] for c in GTS[DIS]])
    xy = XY[DIS]
    for c in SCLS:                                   # direct labels on large domains
        pts = xy[GTS[DIS] == c]; med = np.median(pts, 0)
        lx, ly = pts[np.argmin(((pts - med) ** 2).sum(1))]
        ax_s.text(lx, ly, c.replace('domain_', 'D'), fontsize=6.5, fontweight='bold', color='white',
                  ha='center', va='center', path_effects=[pe.withStroke(linewidth=1.4, foreground=ps.INK)])


def draw_method(ax_b, ax_s, key):
    b, s, (rb, rs) = method_colours(key)
    draw_map(ax_b, b); draw_map(ax_s, s)
    ax_b.text(.5, -.04, f'ARI {rb:.2f}', transform=ax_b.transAxes, ha='center', va='top', fontsize=6)
    ax_s.text(.5, -.04, f'ARI {rs:.2f}', transform=ax_s.transAxes, ha='center', va='top', fontsize=6)


def region_key(ax):
    ax.axis('off')
    for i, c in enumerate(BCLS):
        y = 1 - i / (len(BCLS) - 1)
        ax.add_patch(plt.Rectangle((0, y - .045), .1, .09, color=ps.REGION[c], transform=ax.transAxes,
                                   clip_on=False))
        ax.text(.16, y, c, transform=ax.transAxes, va='center', fontsize=6)


def draw_box(ax, metric, ylabel):
    rng = np.random.default_rng(0)
    for i, (k, *_ ) in enumerate(METHODS):
        v = D.loc[D.method == CSVNAME[k], metric].to_numpy()
        col = ps.METHOD[FAM[k]]
        bp = ax.boxplot(v, positions=[i], widths=.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color=ps.INK, lw=.8), whiskerprops=dict(color=ps.INK, lw=.6),
                        capprops=dict(color=ps.INK, lw=.6), boxprops=dict(lw=.6))
        bp['boxes'][0].set(facecolor=col, edgecolor=ps.INK, alpha=.85)
        ax.scatter(i + rng.uniform(-.13, .13, len(v)), v, s=5, color='white', edgecolor=ps.INK,
                   linewidth=.45, zorder=3)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([LAB[k] for k, *_ in METHODS], rotation=40, ha='right', rotation_mode='anchor')
    ax.set_ylim(0, 1); ax.set_yticks([0, .25, .5, .75, 1]); ax.set_ylabel(ylabel)
    ax.set_xlim(-.6, len(METHODS) - .4)


def draw_paired(ax):
    ref = D[D.method == CSVNAME['default']].set_index('section')
    comps = [k for k, *_ in METHODS if k != 'default']
    for i, k in enumerate(comps):
        other = D[D.method == CSVNAME[k]].set_index('section').loc[ref.index]
        for off, metric, mk in [(-.24, 'bg', 'o'), (.24, 'sal', '^')]:   # wide enough that counts never overlap
            d = (ref[metric] - other[metric]).to_numpy()
            x = i + off + np.linspace(-.05, .05, len(d))
            ax.scatter(x, d, s=8, marker=mk, linewidths=.35, edgecolors=ps.INK, zorder=3,   # outline keeps light 'worse' points visible
                       c=[ps.DELTA['up'] if v > 0 else ps.DELTA['down'] for v in d])
            ax.plot([i + off - .09, i + off + .09], [d.mean()] * 2, color=ps.INK, lw=.8, zorder=4)
            ax.text(i + off, max(d.max(), 0) + .04, f'{int((d > 0).sum())}/5', ha='center',
                    va='bottom', fontsize=5.6)
    ax.axhline(0, color=ps.INK, lw=.6)
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels([LAB[k] for k in comps], rotation=40, ha='right', rotation_mode='anchor')
    ax.set_ylabel('Haruka − comparator (ARI)'); ax.set_ylim(-.2, 1.05); ax.set_xlim(-.6, len(comps) - .4)
    ax.legend(handles=[Line2D([], [], marker='o', ls='', ms=3, color=ps.NEUTRAL['dark'], label='background'),
                       Line2D([], [], marker='^', ls='', ms=3, color=ps.NEUTRAL['dark'], label='salient'),
                       Line2D([], [], marker='s', ls='', ms=3, color=ps.DELTA['up'], label='Haruka better'),
                       Line2D([], [], marker='s', ls='', ms=3, color=ps.DELTA['down'], label='Haruka worse')],
              loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=4, handletextpad=.15,
              columnspacing=.7, borderaxespad=.2)


# ---------------------------------------------------------------- standalone subpanels
def standalone():
    written = []
    # a: ground truth
    fig = plt.figure(figsize=(52 * ps.MM, 62 * ps.MM))
    g = gridspec.GridSpec(2, 2, width_ratios=[1, .62], wspace=.05, hspace=.08)
    ab, as_ = fig.add_subplot(g[0, 0]), fig.add_subplot(g[1, 0]); draw_gt(ab, as_)
    region_key(fig.add_subplot(g[0, 1]))
    ab.set_title('Background subregions', fontsize=6.5); as_.set_title('Salient domains', fontsize=6.5)
    written += ps.save(fig, OUT / 'fig2a_ground_truth'); plt.close(fig)
    # b: one column per method
    for k, *_ in METHODS:
        fig = plt.figure(figsize=(32 * ps.MM, 62 * ps.MM))
        g = gridspec.GridSpec(2, 1, hspace=.22)
        ab, as_ = fig.add_subplot(g[0]), fig.add_subplot(g[1]); draw_method(ab, as_, k)
        ab.set_title(LAB[k], fontsize=6.5, fontweight='bold' if k == 'default' else 'normal')
        written += ps.save(fig, OUT / f'fig2b_{k}'); plt.close(fig)
    # c, d, e
    for name, fn in [('fig2c_background_ari', lambda ax: draw_box(ax, 'bg', 'Background subregion ARI')),
                     ('fig2d_salient_ari', lambda ax: draw_box(ax, 'sal', 'Salient domain ARI')),
                     ('fig2e_paired_difference', draw_paired)]:
        fig, ax = plt.subplots(figsize=(56 * ps.MM, 46 * ps.MM)); fn(ax)
        written += ps.save(fig, OUT / name); plt.close(fig)
    return written


# ---------------------------------------------------------------- composite (preview of the page)
def composite():
    fig = plt.figure(figsize=(183 * ps.MM, 126 * ps.MM))
    top = gridspec.GridSpec(2, 6, left=.075, right=.995, top=.955, bottom=.47,
                            width_ratios=[1, .5, 1, 1, 1, 1], wspace=.06, hspace=.1)
    ab, as_ = fig.add_subplot(top[0, 0]), fig.add_subplot(top[1, 0]); draw_gt(ab, as_)
    ab.set_title('Ground truth', fontsize=6.5, fontweight='bold')
    ps.panel_label(ab, 'a', x=-.3, y=1.08)
    fig.text(.07, (.955 + .5) / 2 + .115, 'Background\nsubregions', ha='right', va='center', fontsize=6.5)
    fig.text(.07, (.955 + .5) / 2 - .115, 'Salient\ndomains', ha='right', va='center', fontsize=6.5)
    region_key(fig.add_subplot(top[0, 1]))
    for j, k in enumerate(MAIN_MAPS):
        ab, as_ = fig.add_subplot(top[0, j + 2]), fig.add_subplot(top[1, j + 2]); draw_method(ab, as_, k)
        ab.set_title(LAB[k], fontsize=6.5, fontweight='bold' if k == 'default' else 'normal')
        if j == 0: ps.panel_label(ab, 'b', x=-.12, y=1.08)
    bot = gridspec.GridSpec(1, 3, left=.065, right=.995, top=.36, bottom=.08, wspace=.36)
    for idx, (letter, fn) in enumerate([('c', lambda ax: draw_box(ax, 'bg', 'Background subregion ARI')),
                                        ('d', lambda ax: draw_box(ax, 'sal', 'Salient domain ARI')),
                                        ('e', draw_paired)]):
        ax = fig.add_subplot(bot[0, idx]); fn(ax); ps.panel_label(ax, letter, x=-.27, y=1.07)
    return ps.save(fig, OUT / 'fig2_composite')


if __name__ == '__main__':
    files = standalone() + composite()
    D.to_csv(OUT / 'fig2_source_data_per_section.csv', index=False)
    rows = []
    for k, *_ in METHODS:
        rb, rs = Z[f'{k}_ari']; rows.append(dict(method=LAB[k], section='B5', background_ari=rb, salient_ari=rs))
    pd.DataFrame(rows).to_csv(OUT / 'fig2b_source_data_maps_B5.csv', index=False)
    bad = []
    for f in files:
        if f.endswith('.svg'):
            ok, sizes = ps.check_fonts(f)
            print(f'{Path(f).name:32s} fonts {sizes}  {"OK" if ok else "BELOW MINIMUM"}')
            if not ok: bad.append(f)
    print('\nALL SVGs >= %.1f pt' % ps.MIN_FONT_PT if not bad else f'\nFONT VIOLATIONS: {bad}')
