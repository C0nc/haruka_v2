#!/usr/bin/env python3
"""Exercise figure blocks, in the layout of the AD block (each panel its own SVG, 46 x 46 mm):
  row 1  UMAPs of all cells (both arms): u1 z cell type | u2 s cell type | u3 z brain region | u4 s brain region
  row 2  UMAPs of vascular cells (Endothelial, Pericyte, VLMC, VSMC): v1 z coloured by observed clock age |
         v2 s coloured by observed clock age | v3 z coloured by condition | v4 s coloured by condition
  row 3  tissue, control OC4 and exercise OE3 (same slide A1): m1 cell type | m2 observed clock age (vascular) |
         m3 held-out clock prediction from z | m4 the same from s   (vascular readouts in SD units, blue = younger)
Fit: weights 10 / 100, batch = run. Held-out predictions from make_exercise_clock_shift.py.
UMAP: 15 neighbours, min_dist 0.3, random_state 0; 40,000 cells (row 1) and 30,000 vascular cells (row 2).
"""
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
_HW = _os.environ.get('HARUKA_WORK') or ('/project/zhihuanglab/yancui' if _os.path.isdir('/project/zhihuanglab/yancui') else _HR + '/work')
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import umap
warnings.filterwarnings('ignore')
sys.path.insert(0, f'{_HR}/figures_paper'); import paperstyle as ps
ps.setup()
OUT = Path(f'{_HR}/figures_paper/exercise')
MD = sorted(Path(f'{_HW}/haruka_exercise/results_w10_s100/exercise_old_full').glob('*/'))[0]
DIV = LinearSegmentedColormap.from_list('div', ps.DIVERGING); SZ = 46 * ps.MM
VAS = ['Endothelial', 'Pericyte', 'VLMC', 'VSMC']
GROUP = {'Neuron-Inhibitory': 'Neuron', 'Neuron-Excitatory': 'Neuron', 'Neuron-MSN': 'Neuron', 'Neuroblast': 'Neuron',
         'Astrocyte': 'Astrocyte', 'Oligodendrocyte': 'Oligodendrocyte', 'OPC': 'Oligodendrocyte', 'Microglia': 'Microglia',
         'Macrophage': 'Microglia', **{v: 'Vascular' for v in VAS}}
GCOL = {**ps.MAJOR, 'Other': ps.NEUTRAL['mid']}
VCOL = {'Endothelial': '#7B524F', 'Pericyte': '#C48A62', 'VLMC': '#93A9C6', 'VSMC': '#82A86F'}

h = ad.read_h5ad(f'{_HW}/haruka_exercise/data/exercise_old_full.h5ad', backed='r'); o = h.obs
names = np.asarray(h.obs_names).astype(str)
assert (np.load(MD / 'obs_names.npy', allow_pickle=True).astype(str) == names).all()
ct = o.celltype.astype(str).to_numpy(); grp = np.array([GROUP.get(c, 'Other') for c in ct])
sub = o.subregion.astype(str).to_numpy(); mouse = o.mouse_id.astype(str).to_numpy()
ex = (o.condition.astype(str) == 'exercise').to_numpy(); xy = np.asarray(h.obsm['spatial'], float)[:, :2]
Z = np.load(MD / 'background.npy'); S = np.load(MD / 'salient.npy')
D = pd.read_csv(OUT / 'oof_clock_predictions_w10_s100.csv.gz').set_index('cell_id')
R = {k: pd.Series(D[k]).reindex(names).to_numpy() for k in ['observed', 'z', 's']}

def panel(fname, letter, title, col=None):
    fig, ax = plt.subplots(figsize=(SZ, SZ)); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.set_title(title, pad=2, color=col or ps.INK, fontweight='bold' if col else 'normal')
    if letter: ps.panel_label(ax, letter, x=-0.06, y=1.06)
    return fig, ax
def legend(ax, cols, ncol=3):
    hs = [plt.Line2D([], [], marker='o', ls='', ms=3, color=v, label=k) for k, v in cols.items()]
    ax.legend(handles=hs, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=ncol, handletextpad=0.1, columnspacing=0.6, fontsize=5.8)
def cbar(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.03); cb.set_label(label, fontsize=6)
    cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.4)
def emb(X, seed=0):
    return umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=seed).fit_transform(X)

rng = np.random.default_rng(0)
# ---- row 1: all cells
i1 = np.sort(rng.choice(len(ct), 40000, replace=False))
EZ, ES = emb(Z[i1]), emb(S[i1])
reg_order = ['CTX_L1/MEN', 'CTX_L2/3', 'CTX_L4/5/6', 'STR_CP/ACB', 'STR_LS/NDB', 'CC/ACO', 'VEN']
for E, lat, L in [(EZ, 'z', 'a'), (ES, 's', 'b')]:
    fig, ax = panel(f'exb_{L}_{lat}_umap_celltype', L, f'{lat}: cell type', ps.LATENT[lat])
    perm = rng.permutation(len(i1))
    ax.scatter(E[perm, 0], E[perm, 1], s=0.3, c=[GCOL[g] for g in grp[i1][perm]], lw=0, rasterized=True)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2'); legend(ax, GCOL); ps.save(fig, OUT / f'exb_{L}_{lat}_umap_celltype'); plt.close(fig)
for E, lat, L in [(EZ, 'z', 'c'), (ES, 's', 'd')]:
    fig, ax = panel(f'exb_{L}', L, f'{lat}: brain region', ps.LATENT[lat]); perm = rng.permutation(len(i1))
    ax.scatter(E[perm, 0], E[perm, 1], s=0.3, c=[ps.REGION[r] for r in sub[i1][perm]], lw=0, rasterized=True)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2'); legend(ax, {r.replace('_', ' '): ps.REGION[r] for r in reg_order}, ncol=3)
    ps.save(fig, OUT / f'exb_{L}_{lat}_umap_region'); plt.close(fig)

# ---- row 2: vascular cells
vm = np.flatnonzero(np.isin(ct, VAS) & np.isfinite(R['observed']))
i2 = np.sort(rng.choice(vm, min(30000, len(vm)), replace=False))
VZ, VS = emb(Z[i2]), emb(S[i2])
oz = R['observed'][i2]; lim = np.nanpercentile(np.abs(oz), 97)
for E, lat, L in [(VZ, 'z', 'e'), (VS, 's', 'f')]:
    fig, ax = panel('', L, f'{lat}: observed clock age (vascular)', ps.LATENT[lat]); perm = rng.permutation(len(i2))
    im = ax.scatter(E[perm, 0], E[perm, 1], s=0.5, c=oz[perm], cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), lw=0, rasterized=True)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2'); cbar(fig, ax, im, 'clock age, region-adjusted (months)')
    ps.save(fig, OUT / f'exb_{L}_{lat}_umap_vascular_clock'); plt.close(fig)
for E, lat, L in [(VZ, 'z', 'g'), (VS, 's', 'h')]:
    fig, ax = panel('', L, f'{lat}: exercise vs control (vascular)', ps.LATENT[lat]); perm = rng.permutation(len(i2))
    cc = np.where(ex[i2][perm], ps.CONDITION['disease'], ps.CONDITION['control'])
    ax.scatter(E[perm, 0], E[perm, 1], s=0.5, c=cc, lw=0, rasterized=True)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2'); legend(ax, {'control': ps.CONDITION['control'], 'exercise': ps.CONDITION['disease']}, ncol=2)
    ps.save(fig, OUT / f'exb_{L}_{lat}_umap_vascular_condition'); plt.close(fig)

# ---- row 3: tissue, OC4 (control) + OE3 (exercise), same slide A1
pair = ['OC4', 'OE3']; sel = np.isin(mouse, pair)
P = xy.copy();
# place the two sections side by side with a gap (each section's own coordinates)
x0 = 0.0
for m in pair:
    mm = mouse == m; P[mm, 0] = xy[mm, 0] - xy[mm, 0].min() + x0; P[mm, 1] = xy[mm, 1] - xy[mm, 1].min()
    x0 = P[mm, 0].max() + 600
vs = sel & np.isin(ct, VAS) & np.isfinite(R['observed'])
def tissue(fname, letter, title, col=None):
    fig, ax = plt.subplots(figsize=(2 * SZ, SZ)); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.set_title(title, pad=14, color=col or ps.INK, fontweight='bold' if col else 'normal'); ps.panel_label(ax, letter, x=-0.03, y=1.12)
    for m, lab in zip(pair, ['control (OC4)', 'exercise (OE3)']):
        mm = mouse == m; ax.text(P[mm, 0].mean(), P[sel, 1].max() + 80, lab, ha='center', va='bottom', fontsize=6)
    ax.plot([P[sel, 0].min(), P[sel, 0].min() + 1000], [P[sel, 1].min() - 250] * 2, color=ps.INK, lw=1)
    ax.text(P[sel, 0].min() + 500, P[sel, 1].min() - 300, '1 mm', ha='center', va='top', fontsize=6)
    return fig, ax
fig, ax = tissue('', 'i', 'Cell type')
ax.scatter(P[sel, 0], P[sel, 1], s=0.15, c=[GCOL[g] for g in grp[sel]], lw=0, rasterized=True)
legend(ax, GCOL, ncol=6); ps.save(fig, OUT / 'exb_i_tissue_celltype'); plt.close(fig)
for k, L, title, col in [('observed', 'j', 'Observed clock age (vascular cells)', None), ('z', 'k', 'z: held-out clock prediction', ps.LATENT['z']),
                         ('s', 'l', 's: held-out clock prediction', ps.LATENT['s'])]:
    v = R[k][vs]; v = (v - np.nanmean(R[k][np.isin(ct, VAS)])) / np.nanstd(R[k][np.isin(ct, VAS)])
    fig, ax = tissue('', L, title, col)
    ax.scatter(P[sel & ~np.isin(ct, VAS), 0], P[sel & ~np.isin(ct, VAS), 1], s=0.1, color=ps.NEUTRAL['light'], lw=0, rasterized=True)
    im = ax.scatter(P[vs, 0], P[vs, 1], s=0.9, c=v, cmap=DIV, norm=TwoSlopeNorm(0, -2, 2), lw=0, rasterized=True)
    cbar(fig, ax, im, 'clock age, region-adjusted (SD; blue = younger)')
    for m in pair:
        mm = vs & (mouse == m); ax.text(P[mouse == m, 0].mean(), P[sel, 1].min() - 300, f'mean {np.nanmean((R[k][mm] - np.nanmean(R[k][np.isin(ct, VAS)])) / np.nanstd(R[k][np.isin(ct, VAS)])):+.2f} SD',
                                        ha='center', va='top', fontsize=6)
    ps.save(fig, OUT / f'exb_{L}_tissue_{k}'); plt.close(fig)
bad = [f.name for f in sorted(OUT.glob('exb_*.svg')) if not ps.check_fonts(f)[0]]
print('panels:', len(list(OUT.glob('exb_*.svg'))), '| font failures:', bad)
