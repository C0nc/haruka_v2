#!/usr/bin/env python3
"""PEtracer M2-1 figure in the layout of the AD block (each panel its own SVG).
Haruka fit: control = the section's non-tumour cells, condition = malignant cells inside tumours (buffer = halo,
neighbour-only). Lineage (clone) is never seen by the model.
  a z UMAP, cell type (all trained cells)      b s UMAP, cell type
  c z UMAP, clone (malignant condition cells)   d s UMAP, clone
  e tissue: cell type                           f tissue: ground-truth clone
  g tissue: held-out clone from z               h tissue: held-out clone from s
  i main lineage results: z vs s vs scVI (lineage_test_summary.csv)
Held-out clone: multinomial logistic (C = 0.1, balanced), GroupKFold(5) over 150 um spatial blocks, as in lineage_test.py.
UMAP: 15 neighbours, min_dist 0.3, random_state 0.
"""
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
import sys, warnings
import numpy as np, pandas as pd, anndata as ad, umap
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
sys.path.insert(0, f'{_HR}/figures_paper'); import paperstyle as ps
ps.setup(); MM = ps.MM; SZ = 46 * MM
OUT = Path(f'{_HR}/figures_paper/petracer'); OUT.mkdir(exist_ok=True)
D = Path(f'{_HR}/bench_petracer_lineage/m2_lineage')
MD = D / 'fits/petracer_M2-1_lineage/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
a = ad.read_h5ad(D / 'petracer_M2-1_lineage.h5ad', backed='r'); o = a.obs
assert (np.load(MD / 'obs_names.npy', allow_pickle=True).astype(str) == np.asarray(a.obs_names).astype(str)).all()
xy = np.asarray(a.obsm['spatial'], float)[:, :2]; ct = o.cell_type.astype(str).to_numpy(); cond = o.condition.astype(str).to_numpy()
clone = o.lin_clone.astype(str).to_numpy()
Z = np.load(MD / 'background.npy'); S = np.load(MD / 'salient.npy')
GRP = {'Malignant': 'Malignant', 'CD11c macrophage': 'Macrophage', 'ARG1 macrophage': 'Macrophage', 'ALOX15 macrophage': 'Macrophage',
       'Exhausted CD8 T cell': 'T / NK', 'CD4 T cell': 'T / NK', 'Treg': 'T / NK', 'NK': 'T / NK', 'B cell': 'B cell',
       'Neutrophil': 'Other myeloid', 'cDC': 'Other myeloid', 'pDC': 'Other myeloid', 'Endothelial': 'Endothelial',
       'Cancer fibroblast': 'Fibroblast', 'Alveolar fibroblast 1': 'Fibroblast', 'Alveolar fibroblast 2': 'Fibroblast',
       'AT1/AT2': 'Epithelial', 'Club cell': 'Epithelial'}
GCOL = {'Malignant': '#7B524F', 'Macrophage': '#A67D94', 'T / NK': '#A7C6FE', 'B cell': '#5D82A8', 'Other myeloid': '#E8C488',
        'Endothelial': '#C48A62', 'Fibroblast': '#82A86F', 'Epithelial': '#C8A9DB'}
CLONES = ['1', '2-1', '2-2', '4', '5', '6']
CCOL = {'1': '#AE874D', '2-1': '#39698E', '2-2': '#61ABC6', '4': '#3C7564', '5': '#B1868E', '6': '#7F5387'}
grp = np.array([GRP.get(c, 'Other myeloid') for c in ct])
mal = (ct == 'Malignant') & (cond == 'disease'); lab = mal & np.isin(clone, CLONES)
train = cond != 'halo'

def ax_off(ax):
    ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
def legend(ax, cols, ncol, title=None):
    hs = [plt.Line2D([], [], marker='o', ls='', ms=3, color=v, label=k) for k, v in cols.items()]
    ax.legend(handles=hs, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=ncol, handletextpad=0.1, columnspacing=0.5,
              fontsize=5.8, title=title, title_fontsize=6)
def panel(name, L, title, col=None, w=SZ):
    fig, ax = plt.subplots(figsize=(w, SZ)); ax_off(ax)
    ax.set_title(title, pad=2, color=col or ps.INK, fontweight='bold' if col else 'normal'); ps.panel_label(ax, L, x=-0.06, y=1.06)
    return fig, ax

rng = np.random.default_rng(0)
# held-out clone prediction (spatial blocks)
li = np.flatnonzero(lab); blocks = (np.floor(xy[li, 0] / 150).astype(int) * 10000 + np.floor(xy[li, 1] / 150).astype(int))
pred = {}
for k, M in [('z', Z), ('s', S)]:
    p = np.empty(len(li), object)
    for tr, te in GroupKFold(5).split(li, clone[li], blocks):
        sc = StandardScaler().fit(M[li[tr]])
        p[te] = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced').fit(sc.transform(M[li[tr]]), clone[li[tr]]).predict(sc.transform(M[li[te]]))
    pred[k] = p; print(k, 'held-out clone balanced acc', round(balanced_accuracy_score(clone[li], p.astype(str)), 3))

# UMAPs: condition (tumour-arm) cells only, one embedding per latent
ia = np.sort(rng.choice(np.flatnonzero(cond == 'disease'), 40000, replace=False)); pm = rng.permutation(len(ia))
for k, M, La, Lc in [('z', Z, 'a', 'c'), ('s', S, 'b', 'd')]:
    e = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(M[ia])[pm]; ii = ia[pm]
    fig, ax = panel('', La, f'{k}: cell type (condition cells)', ps.LATENT[k])
    ax.scatter(e[:, 0], e[:, 1], s=0.3, c=[GCOL[g] for g in grp[ii]], lw=0, rasterized=True); legend(ax, GCOL, 3)
    ps.save(fig, OUT / f'pt_{La}_{k}_umap_celltype'); plt.close(fig)
    fig, ax = panel('', Lc, f'{k}: lineage clone (condition cells)', ps.LATENT[k]); lb = lab[ii]
    ax.scatter(e[~lb, 0], e[~lb, 1], s=0.3, color='#DCD8D3', lw=0, rasterized=True)
    ax.scatter(e[lb, 0], e[lb, 1], s=0.4, c=[CCOL[c] for c in clone[ii][lb]], lw=0, rasterized=True); legend(ax, CCOL, 6, 'clone (grey = non-malignant)')
    ps.save(fig, OUT / f'pt_{Lc}_{k}_umap_clone'); plt.close(fig)

# tissue
def tissue(L, title, col=None):
    fig, ax = plt.subplots(figsize=(SZ, SZ)); ax_off(ax); ax.set_aspect('equal')
    ax.set_title(title, pad=2, color=col or ps.INK, fontweight='bold' if col else 'normal'); ps.panel_label(ax, L, x=-0.06, y=1.06)
    x0, y0 = xy[:, 0].min(), xy[:, 1].min(); ax.plot([x0, x0 + 1000], [y0 - 150] * 2, color=ps.INK, lw=1)
    ax.text(x0 + 500, y0 - 200, '1 mm', ha='center', va='top', fontsize=6)
    return fig, ax
fig, ax = tissue('e', 'Cell type')
ax.scatter(xy[:, 0], xy[:, 1], s=0.08, c=[GCOL[g] for g in grp], lw=0, rasterized=True); legend(ax, GCOL, 3)
ps.save(fig, OUT / 'pt_e_tissue_celltype'); plt.close(fig)
for L, title, col, labels in [('f', 'Ground-truth clone', None, clone[li]), ('g', 'z: held-out clone', ps.LATENT['z'], pred['z'].astype(str)),
                              ('h', 's: held-out clone', ps.LATENT['s'], pred['s'].astype(str))]:
    if L != 'f':
        title += f'\nbalanced acc {balanced_accuracy_score(clone[li], labels):.2f}'
    fig, ax = tissue(L, title, col)
    ax.scatter(xy[:, 0], xy[:, 1], s=0.06, color='#DCD8D3', lw=0, rasterized=True)
    ax.scatter(xy[li, 0], xy[li, 1], s=0.25, c=[CCOL[c] for c in labels], lw=0, rasterized=True)
    if L == 'f': legend(ax, CCOL, 6, 'clone')
    ps.save(fig, OUT / f'pt_{L}_tissue_clone'); plt.close(fig)

# niche reconstruction: observed k=20 neighbour fraction vs Haruka D([z,0]) (background) vs D([z,s]) (with s)
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
import json
types = json.load(open(MD / 'meta.json'))['cell_type_categories']
pbg = np.load(MD / 'background_composition.npy'); pfull = np.load(MD / 'condition_composition.npy')
code = pd.Series(range(len(types)), index=types).reindex(ct).to_numpy()
nn = NearestNeighbors(n_neighbors=21).fit(xy); _, ind = nn.kneighbors(xy); nb = code[ind[:, 1:]]
obs = np.stack([(nb == j).mean(1) for j in range(len(types))], 1)
cm = cond == 'disease'
gain = []
for j, tname in enumerate(types):
    if obs[cm, j].mean() < 0.02: continue
    gain.append((tname, pearsonr(obs[cm, j], pbg[cm, j])[0], pearsonr(obs[cm, j], pfull[cm, j])[0]))
G = pd.DataFrame(gain, columns=['type', 'r_z_background', 'r_with_s']); G['gain'] = G.r_with_s - G.r_z_background
G.sort_values('gain', ascending=False).to_csv(OUT / 'pt_niche_reconstruction.csv', index=False); print(G.round(3).to_string(index=False))
top = G.sort_values('gain', ascending=False).type.iloc[:2].tolist()
from matplotlib.colors import LinearSegmentedColormap
SEQ = LinearSegmentedColormap.from_list('seq', ps.SEQUENTIAL)
letters = iter('ijklmn')
for tname in top:
    j = types.index(tname); vmax = np.percentile(obs[cm, j], 99)
    for v, title, col in [(obs[:, j], f'observed {tname} share (k=20)', None),
                          (pbg[:, j], f'z only D([z,0])', ps.LATENT['z']), (pfull[:, j], f'z + s D([z,s])', ps.LATENT['s'])]:
        L = next(letters)
        t_ = title if col is None else f'{title}: {tname}\nr with observed {pearsonr(obs[cm, j], v[cm])[0]:.2f}'
        fig, ax = tissue(L, t_, col)
        ax.scatter(xy[~cm, 0], xy[~cm, 1], s=0.06, color='#DCD8D3', lw=0, rasterized=True)
        im = ax.scatter(xy[cm, 0], xy[cm, 1], s=0.15, c=v[cm], cmap=SEQ, vmin=0, vmax=vmax, lw=0, rasterized=True)
        cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.04, pad=0.02); cb.set_label('neighbour fraction', fontsize=6); cb.ax.tick_params(labelsize=6, length=2)
        ps.save(fig, OUT / f'pt_{L}_niche_{tname.replace(" ", "_").replace("/", "_")}'); plt.close(fig)

# main results
R = pd.read_csv(D / 'lineage_test_summary.csv')
names = ['Clone across\ntumours', 'Clone 2-1 vs 2-2\n(AUROC)', 'Subclade\nin clone 5', 'Fitness\nin clone 5', 'Lineage distance\n(partial)']
R = R.iloc[[0, 1, 2, 3, 5]].reset_index(drop=True)
fig, axes = plt.subplots(1, 5, figsize=(150 * MM, 40 * MM))
for ax, (_, r), nm in zip(axes, R.iterrows(), names):
    vals = [r.z, r.s, r.scVI]
    ax.bar([0, 1, 2], vals, 0.7, color=[ps.LATENT['z'], ps.LATENT['s'], ps.NEUTRAL['mid']])
    ax.axhline(float(r.chance), color=ps.INK, lw=0.6, ls='--'); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['z', 's', 'scVI'])
    ax.set_title(nm, fontsize=6.5); lo = min(0, min(vals) - 0.02)
    ax.set_ylim(lo if float(r.chance) != 0.5 else 0.4, max(vals) * 1.15 + 0.01)
    for x, v in enumerate(vals): ax.text(x, v + 0.005, f'{v:.2f}', ha='center', va='bottom', fontsize=5.8)
axes[0].set_ylabel('held-out score (spatial blocks)'); ps.panel_label(axes[0], 'o', x=-0.6)
fig.tight_layout(); ps.save(fig, OUT / 'pt_o_results'); plt.close(fig)
print('fonts:', [f.name for f in OUT.glob('pt_*.svg') if not ps.check_fonts(f)[0]])
