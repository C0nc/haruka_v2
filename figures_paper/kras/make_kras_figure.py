#!/usr/bin/env python3
"""KRAS-inhibitor mouse lung (Xenium) figure: the drug spares Itgb6+ low-MAPK persisters; s marks the nodules whose
tumour neighbourhood converts towards them; the baseline (z) does not predict that treatment effect beyond a null.
Haruka fit: results_kras_te_null (control = untreated sections, condition = KRASi-treated; saved model + 3 shuffled-s
nulls).  Response ground truth never used by the model: MAPK pathway activity score (MPAS: Dusp4/6, Spry2/4, Etv4/5,
Ccnd1, Epha2); responder = treated tumour cell with MPAS below the control 25th percentile of its own subtype.
  a tissue: tumour subtypes (treated example)      b tissue: MAPK MPAS on tumour cells
  c per section: tumour burden and Itgb6+ share     d treated: responder rate by tumour subtype (25% = no response)
  e-h UMAPs of z and s (treated cells): cell group and MPAS
  i nodule treatment effect (TE, decoded neighbour shift): section-centred PCA, PC1 loadings
  j tissue: nodules coloured by their Itgb6+-neighbour TE
  k type-matched nodules: Itgb6+-neighbour TE vs responder rate, per section
  l can the baseline predict the TE?  z / p_bg / s vs the shuffled-s null
  m-q one interpretable background axis (z-PC2 of nodule mean z, bench_kras_xenium/nodule_te/name_bad_te_z.py):
      m tumour-cell genes, n predicted untreated neighbours, o held-out-section prediction of the s-specific bad TE and of the
      responder rate (z-PC3 failed, greyed), p nodules coloured by z-PC2, q z-PC2 already varies in untreated control nodules
  r-v cell-type interactions, high vs low z-PC2 tertile nodules (zpc2_interactions.py): r niche composition, s squidpy
      nhood_enrichment (tumour row), t LIANA+ rank_aggregate tumour <-> myeloid / AT1 / ciliated (descriptive, no test of
      the high-low difference), u / v example nodules (highest / lowest z-PC2 in JB_M0; boxes on a and p mark their fields of view)
Sources: figures_paper/kras/data/ (kras_fig_data.py) and bench_kras_xenium/nodule_te/ (nodule_te.py, refit runs).
Caveat (legend): control and treated sections are separate Xenium runs, so arm is confounded with run.
"""
import sys, json, warnings
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from PIL import Image
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/yancui/haruka_v2/figures_paper'); import paperstyle as ps
ps.setup(); MM = ps.MM
OUT = Path('/home/yancui/haruka_v2/figures_paper/kras'); DAT = OUT / 'data'; NT = Path('/home/yancui/haruka_v2/bench_kras_xenium/nodule_te')
CZ, CS = ps.LATENT['z'], ps.LATENT['s']
TUM = {'Tumor_epithelial': ps.DOMAIN[1], 'Tumor_epithelial_Itgb6': ps.REGION['CC/ACO'], 'Tumor_EMT': ps.REGION['CTX_L1/MEN']}
TNAME = {'Tumor_epithelial': 'Tumour epithelial', 'Tumor_epithelial_Itgb6': 'Tumour Itgb6+', 'Tumor_EMT': 'Tumour EMT'}
def group(c):
    if c in TUM: return c
    if c.endswith('_EC'): return 'Endothelial'
    if c in ('AT1', 'AT2', 'Ciliated'): return 'Epithelial (normal)'
    if c.startswith(('Fibroblast', 'Pericyte', 'SmoothMuscle', 'Cartilage', 'StriatedMuscle', 'Schwann')): return 'Stromal'
    if c.startswith(('Macrophage', 'moMacrophage', 'Monocyte', 'Neutrophil', 'cDC', 'moDC', 'mregDC', 'pDC')): return 'Myeloid'
    return 'Lymphoid'
GCOL = {**TUM, 'Epithelial (normal)': ps.MAJOR['Astrocyte'], 'Endothelial': ps.MAJOR['Vascular'], 'Stromal': ps.MAJOR['Oligodendrocyte'],
        'Myeloid': ps.MAJOR['Microglia'], 'Lymphoid': ps.MAJOR['Neuron']}
GNAME = {**TNAME, 'Epithelial (normal)': 'Epithelial (normal)', 'Endothelial': 'Endothelial', 'Stromal': 'Stromal', 'Myeloid': 'Myeloid', 'Lymphoid': 'Lymphoid'}
SEQ = LinearSegmentedColormap.from_list('seq', ps.SEQUENTIAL); DIV = LinearSegmentedColormap.from_list('div', ps.DIVERGING)
ARM = {'control': ps.CONDITION['control'], 'treated': ps.CONDITION['disease']}
made = []
def save(fig, name): ps.save(fig, OUT / name); plt.close(fig); made.append(name)
def leg(ax, items, **kw):
    ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', ms=3, color=c, label=l) for l, c in items], loc='upper left',
              bbox_to_anchor=(1.0, 1.0), fontsize=5.8, handletextpad=0.2, borderaxespad=0, **kw)

E = np.load(DAT / 'example.npz', allow_pickle=True); xy = E['xy']; sec = str(E['section']); ctE = E['cell_type']
Dx = np.load(NT / 'zpc2_interactions' / 'D_examples.npz', allow_pickle=True)
assert str(Dx['section']) == sec, 'u/v examples must come from the example section'
def fov(ax, pad=60):
    # dashed boxes marking the fields of view shown in panels u (high z-PC2) and v (low z-PC2)
    for k, lab in [('hi', 'u'), ('lo', 'v')]:
        X_ = Dx[f'{k}_xy']; lo_, hi_ = X_.min(0) - pad, X_.max(0) + pad; w_ = hi_ - lo_; c_ = ps.DELTA['down'] if k == 'hi' else ps.DELTA['up']
        sz = max(w_.max(), 400); cen_ = (lo_ + hi_) / 2
        ax.add_patch(plt.Rectangle(cen_ - sz / 2, sz, sz, fill=False, ec=c_, lw=1.0, zorder=6))
        ax.text(cen_[0] + sz / 2 + 60, cen_[1] + sz / 2, lab, color=c_, fontsize=7, fontweight='bold', ha='left', va='top', zorder=6)
def tissue(L, title, w=50, col=None):
    fig, ax = plt.subplots(figsize=(w * MM, w * MM)); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, pad=2, color=col or ps.INK, fontweight='bold' if col else 'normal'); ps.panel_label(ax, L, x=-0.1, y=1.1); return fig, ax
def sbar(ax, X, um=1000):
    x0, y0 = X[:, 0].min(), X[:, 1].min() - 120; ax.plot([x0, x0 + um], [y0, y0], color=ps.INK, lw=1); ax.text(x0 + um / 2, y0 - 80, f'{um} µm', ha='center', va='top', fontsize=5.8)
# a
fig, ax = tissue('a', f'{sec} (KRASi): tumour subtypes')
ax.scatter(*xy[~np.isin(ctE, list(TUM))].T, s=0.15, c=ps.NEUTRAL['light'], lw=0, rasterized=True)
for t, c in TUM.items(): ax.scatter(*xy[ctE == t].T, s=0.35, c=c, lw=0, rasterized=True)
leg(ax, [(TNAME[t], c) for t, c in TUM.items()] + [('Other cells', ps.NEUTRAL['light'])]); sbar(ax, xy); fov(ax); save(fig, 'kr_a_tissue_tumour')
# b
fig, ax = tissue('b', 'MAPK activity (MPAS)')
ax.scatter(*xy.T, s=0.15, c=ps.NEUTRAL['light'], lw=0, rasterized=True)
m = np.isfinite(E['mpas']); q = np.nanpercentile(E['mpas'], [5, 95])
im = ax.scatter(*xy[m].T, s=0.4, c=np.clip(E['mpas'][m], *q), cmap=SEQ, lw=0, rasterized=True)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01); cb.ax.tick_params(labelsize=5.8); cb.set_label('MPAS (low = drug acted)', fontsize=5.8); cb.outline.set_linewidth(0.4)
sbar(ax, xy); save(fig, 'kr_b_tissue_mpas')
# c
Sx = pd.read_csv(DAT / 'sections.csv')
fig, axs = plt.subplots(1, 2, figsize=(56 * MM, 38 * MM))
for ax, col, lab in [(axs[0], 'burden', 'tumour burden'), (axs[1], 'itgb6_share', 'Itgb6+ share\nof tumour')]:
    for x, arm in enumerate(['control', 'treated']):
        t = Sx[Sx.arm == arm]; ax.scatter(np.full(len(t), x) + np.linspace(-0.12, 0.12, len(t)), t[col], s=10, c=ARM[arm], edgecolors=ps.INK, lw=0.3, zorder=3)
        ax.plot([x - 0.25, x + 0.25], [t[col].mean()] * 2, color=ps.INK, lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f'control\n(n={(Sx.arm == "control").sum()})', f'KRASi\n(n={(Sx.arm == "treated").sum()})']); ax.set_xlim(-0.5, 1.5)
    ax.set_title(lab, pad=2); ax.set_ylim(0, None)
axs[0].set_ylabel('fraction (per section)'); ps.panel_label(axs[0], 'c', x=-0.55); fig.tight_layout(); save(fig, 'kr_c_composition')
# d
R = pd.read_csv(DAT / 'subtype_resp.csv')
fig, ax = plt.subplots(figsize=(40 * MM, 38 * MM))
for x, t in enumerate(TUM):
    v = R[R.subtype == t].resp_rate; ax.scatter(np.full(len(v), x) + np.linspace(-0.1, 0.1, len(v)), v, s=10, c=TUM[t], edgecolors=ps.INK, lw=0.3, zorder=3)
    ax.plot([x - 0.25, x + 0.25], [v.mean()] * 2, color=ps.INK, lw=0.8)
ax.axhline(0.25, color=ps.NEUTRAL['mid'], lw=0.6, ls=':'); ax.text(2.45, 0.255, 'no response', fontsize=5.8, ha='right', va='bottom', color=ps.NEUTRAL['dark'])
ax.set_xticks(range(3)); ax.set_xticklabels([TNAME[t].replace('Tumour ', '') for t in TUM]); ax.set_ylabel('responder rate (treated sections)')
ax.set_title('Itgb6+ cells do not respond', pad=2); ps.panel_label(ax, 'd', x=-0.4); save(fig, 'kr_d_subtype_response')
# e-h UMAPs
U = np.load(DAT / 'umap.npz', allow_pickle=True); ug = np.array([group(c) for c in U['cell_type']])
for L, k, by in [('e', 'z', 'type'), ('f', 's', 'type'), ('g', 'z', 'mpas'), ('h', 's', 'mpas')]:
    X = U[f'umap_{k}']; col = CZ if k == 'z' else CS
    fig, ax = plt.subplots(figsize=(42 * MM, 42 * MM)); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f'{k} UMAP: ' + ('cell group' if by == 'type' else 'tumour MPAS'), pad=2, color=col, fontweight='bold'); ps.panel_label(ax, L, x=-0.04, y=1.03)
    if by == 'type':
        for g_, c in GCOL.items(): ax.scatter(*X[ug == g_].T, s=0.2, c=c, lw=0, rasterized=True)
        if k == 's': leg(ax, [(GNAME[g_], c) for g_, c in GCOL.items()])
    else:
        mm = np.isfinite(U['mpas']); ax.scatter(*X[~mm].T, s=0.2, c=ps.NEUTRAL['light'], lw=0, rasterized=True)
        q = np.nanpercentile(U['mpas'], [5, 95]); im = ax.scatter(*X[mm].T, s=0.3, c=np.clip(U['mpas'][mm], *q), cmap=SEQ, lw=0, rasterized=True)
        if k == 's':
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01); cb.ax.tick_params(labelsize=5.8); cb.set_label('MPAS', fontsize=5.8); cb.outline.set_linewidth(0.4)
    save(fig, f'kr_{L}_umap_{k}_{by}')
# i nodule TE PCA (all treated nodules, refit)
Nz = np.load(NT / 'nodules_refit.npz', allow_pickle=True); Nm = pd.read_csv(NT / 'nodules_refit.csv'); types = list(Nz['types'])
tr = Nm.treated.to_numpy(); T = Nz['TE'][tr]; st = Nm.section.to_numpy()[tr]
Tc = T.copy()
for u in np.unique(st): Tc[st == u] -= Tc[st == u].mean(0)
pca = PCA(4).fit(Tc); ld = pd.Series(pca.components_[0], index=types)
if ld['Tumor_epithelial_Itgb6'] < 0: ld = -ld
top = pd.concat([ld.nsmallest(4), ld.nlargest(4)]).sort_values()
fig, ax = plt.subplots(figsize=(46 * MM, 40 * MM))
ax.barh(range(len(top)), top, color=[ps.DELTA['up'] if v > 0 else ps.DELTA['down'] for v in top], height=0.7)
ax.set_yticks(range(len(top))); ax.set_yticklabels([TNAME.get(i, i).replace('_', ' ') for i in top.index]); ax.axvline(0, color=ps.INK, lw=0.5)
ax.set_xlabel('PC1 loading (neighbour type)'); ax.set_title(f'nodule TE axis: PC1 {pca.explained_variance_ratio_[0]:.0%} of variance\n({tr.sum()} treated nodules, 4 sections)', pad=2)
ps.panel_label(ax, 'i', x=-0.75); save(fig, 'kr_i_te_axis')
# j nodule map
fig, ax = tissue('j', 'nodule TE: Δ Itgb6+ neighbours')
ax.scatter(*xy.T, s=0.15, c=ps.NEUTRAL['light'], lw=0, rasterized=True)
nd = E['nodule'] >= 0; te = E['nodule_te_itgb6']; lim = np.nanpercentile(np.abs(te[nd]), 95)
ax.scatter(*E['tum_xy'][~nd].T, s=0.25, c=ps.NEUTRAL['mid'], lw=0, rasterized=True)
im = ax.scatter(*E['tum_xy'][nd].T, s=0.5, c=te[nd], cmap=DIV, vmin=-lim, vmax=lim, lw=0, rasterized=True)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01); cb.ax.tick_params(labelsize=5.8); cb.set_label('Δ Itgb6+ neighbour share', fontsize=5.8); cb.outline.set_linewidth(0.4)
sbar(ax, xy); save(fig, 'kr_j_tissue_nodule_te')
# k type-matched nodules: TE vs responder rate
Oz = np.load(NT / 'nodules_refit_ownTumEpi.npz', allow_pickle=True); Om = pd.read_csv(NT / 'nodules_refit_ownTumEpi.csv')
tro = Om.treated.to_numpy(); x = Oz['TE'][tro][:, list(Oz['types']).index('Tumor_epithelial_Itgb6')]; y = Om.resp_rate.to_numpy()[tro]; so = Om.section.to_numpy()[tro]
B = pd.read_csv(NT / 'te_vs_biology_refit_ownTumEpi.csv'); b = B[(B.te == 'TE Tumor_epithelial_Itgb6') & (B.biology == 'responder rate')].iloc[0]
SC = dict(zip(sorted(np.unique(so)), [ps.REGION['CC/ACO'], ps.REGION['CTX_L2/3'], ps.REGION['CTX_L4/5/6'], ps.REGION['CTX_L1/MEN']]))
fig, ax = plt.subplots(figsize=(48 * MM, 40 * MM))
for u in np.unique(so):
    m = so == u; ax.scatter(x[m], y[m], s=12, c=SC[u], edgecolors=ps.INK, lw=0.3, label=f'{u} (n={m.sum()})')
ax.set_xlabel('Δ Itgb6+ neighbour share (TE from s)'); ax.set_ylabel('nodule responder rate')
ax.set_title(f'tumour-epithelial nodules\nwithin-section ρ median {b["median"]:+.2f}, partial {b["partial"]:+.2f}', pad=2)
ax.legend(fontsize=5.6, loc='upper right', handletextpad=0.1, borderaxespad=0, labelspacing=0.2); ps.panel_label(ax, 'k', x=-0.32); save(fig, 'kr_k_te_vs_response')
# l baseline predicts TE? (type-matched, leave-section-out ridge, from run_refit_own.out)
import re
_log = open(NT / 'run_refit_own.out').read()
_l = re.search(r'TE PC1-2\s+z \(baseline\) ([+-][\d.]+).*?p_bg \(baseline\) ([+-][\d.]+).*?s \(ceiling\) ([+-][\d.]+)', _log)
vals = {'z (baseline)': float(_l[1]), 'p_bg (baseline)': float(_l[2]), 's': float(_l[3])}
_n = re.findall(r'NULL shuffled-s \d: z -> null TE PC1-2 ([+-][\d.]+) .*? s -> null TE PC1-2 ([+-][\d.]+)', _log)
nulls = {'z (baseline)': [float(a) for a, _ in _n], 's': [float(b) for _, b in _n]}
assert len(_n) == 3, 'run_refit_own.out: expected 3 shuffled-s null lines'
fig, ax = plt.subplots(figsize=(44 * MM, 40 * MM))
cols = [CZ, ps.NEUTRAL['mid'], CS]
ax.bar(range(3), list(vals.values()), color=cols, width=0.6)
for i, k in enumerate(vals):
    if k in nulls: ax.scatter([i + 0.38] * 3, nulls[k], s=10, marker='x', c=ps.INK, lw=0.6, zorder=3)
ax.axhline(0, color=ps.INK, lw=0.5); ax.set_xticks(range(3)); ax.set_xticklabels(['z\n(baseline)', 'p_bg\n(baseline)', 's\n(ceiling)'])
ax.set_ylabel('held-out r (leave-section-out)\nnodule TE PC1-2'); ax.set_ylim(-0.4, 1.0)
ax.set_title('baseline z does not beat\nthe shuffled-s null (×)', pad=2); ps.panel_label(ax, 'l', x=-0.45); save(fig, 'kr_l_baseline_predicts_te')


# ---------------- m-q background axis z-PC2 (bench_kras_xenium/nodule_te/name_bad_te_z.py)
import anndata as ad
F2 = np.load(NT / 'name_zPC2_fig.npz', allow_pickle=True)
G2 = pd.read_csv(NT / 'name_zPC2_genes.csv').dropna(); nb2 = pd.read_csv(NT / 'name_zPC2_neighbours.csv', index_col=0).rho
Hd = pd.read_csv(NT / 'name_bad_te_heldout.csv')
# m genes
t = pd.concat([G2.nsmallest(7, 'rho'), G2.nlargest(8, 'rho').iloc[::-1]]).sort_values('rho')
fig, ax = plt.subplots(figsize=(40 * MM, 54 * MM))
ax.barh(range(len(t)), t.rho, color=[ps.DELTA['down'] if v > 0 else ps.DELTA['up'] for v in t.rho], height=0.7)
ax.set_yticks(range(len(t))); ax.set_yticklabels(t.gene, fontstyle='italic'); ax.axvline(0, color=ps.INK, lw=0.5)
ax.set_xlabel('ρ with z-PC2\n(tumour cells)'); ax.set_title('background axis z-PC2:\ngenes (+ = worse TE)', pad=2, color=CZ, fontweight='bold')
ps.panel_label(ax, 'm', x=-0.6); save(fig, 'kr_m_zpc2_genes')
# n neighbours
t = pd.concat([nb2.head(4), nb2.tail(6)]).sort_values()
fig, ax = plt.subplots(figsize=(42 * MM, 40 * MM))
ax.barh(range(len(t)), t.values, color=[ps.DELTA['down'] if v > 0 else ps.DELTA['up'] for v in t.values], height=0.7)
ax.set_yticks(range(len(t))); ax.set_yticklabels([i.replace('_', ' ') for i in t.index]); ax.axvline(0, color=ps.INK, lw=0.5)
ax.set_xlabel('ρ with z-PC2'); ax.set_title('z-PC2: predicted untreated\nneighbours D([z,0])', pad=2, color=CZ, fontweight='bold')
ps.panel_label(ax, 'n', x=-0.75); save(fig, 'kr_n_zpc2_neighbours')
# o held-out sections
fig, ax = plt.subplots(figsize=(50 * MM, 40 * MM))
for x, (axn, col) in enumerate([('zPC2', CZ), ('zPC3', ps.NEUTRAL['mid'])]):
    t = Hd[Hd.axis == axn]
    for j, (m_, lab) in enumerate([('rho_bad_te', 'bad TE'), ('rho_resp', 'responder\nrate')]):
        xx = x * 2.6 + j; ax.scatter(np.full(len(t), xx) + np.linspace(-0.12, 0.12, len(t)), t[m_], s=10, c=col, edgecolors=ps.INK, lw=0.3, zorder=3)
        ax.plot([xx - 0.25, xx + 0.25], [t[m_].median()] * 2, color=ps.INK, lw=0.8)
ax.axhline(0, color=ps.INK, lw=0.5); ax.set_xticks([0, 1, 2.6, 3.6]); ax.set_xticklabels(['bad TE', 'responders', 'bad TE', 'responders'], rotation=30, ha='right')
ax.text(0.5, 0.86, 'z-PC2', transform=ax.get_xaxis_transform(), ha='center', color=CZ, fontweight='bold', fontsize=6)
ax.text(3.1, 0.86, 'z-PC3 (fails)', transform=ax.get_xaxis_transform(), ha='center', color=ps.NEUTRAL['dark'], fontsize=6)
ax.set_ylabel('held-out section ρ\n(axis fit on other 3)'); ax.set_ylim(-0.75, 1.0)
ax.set_title('prediction on held-out sections', pad=2); ps.panel_label(ax, 'o', x=-0.42); save(fig, 'kr_o_zpc2_heldout')
# p tissue: nodules by z-PC2
ob = ad.read_h5ad('/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad', backed='r').obs
pxy = np.c_[ob.x_centroid.to_numpy(float), ob.y_centroid.to_numpy(float)][F2['ex_cells']]
fig, ax = tissue('p', f'z-PC2 per nodule ({F2["ex_section"]})', col=CZ)
ax.scatter(*xy.T, s=0.15, c=ps.NEUTRAL['light'], lw=0, rasterized=True)
ax.scatter(*xy[ctE == 'Ciliated'].T, s=0.3, c=ps.MAJOR['Astrocyte'], lw=0, rasterized=True)
lim = np.percentile(np.abs(F2['treated_score']), 95)
im = ax.scatter(*pxy.T, s=0.5, c=F2['ex_score'], cmap=DIV, vmin=-lim, vmax=lim, lw=0, rasterized=True)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01); cb.ax.tick_params(labelsize=5.8); cb.set_label('z-PC2 (+ = worse TE)', fontsize=5.8); cb.outline.set_linewidth(0.4)
ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', ms=3, color=ps.MAJOR['Astrocyte'], label='Ciliated (airway)')], loc='lower right', fontsize=5.8, borderaxespad=0, handletextpad=0.1)
sbar(ax, xy); fov(ax); save(fig, 'kr_p_tissue_zpc2')
# q control vs treated
fig, ax = plt.subplots(figsize=(40 * MM, 40 * MM))
bins_ = np.linspace(-2.5, 2.5, 26)
ax.hist(F2['control_score'], bins=bins_, density=True, color=ARM['control'], alpha=0.9, label=f'control nodules (n={len(F2["control_score"])})')
ax.hist(F2['treated_score'], bins=bins_, density=True, histtype='step', color=ARM['treated'], lw=1.0, label=f'treated nodules (n={len(F2["treated_score"])})')
ax.set_xlabel('z-PC2 (section-centred)'); ax.set_ylabel('density'); ax.set_ylim(0, ax.get_ylim()[1] * 1.45); ax.legend(fontsize=5.6, loc='upper left', borderaxespad=0)
ax.set_title(f'present before treatment\n(SD control/treated {F2["control_score"].std() / F2["treated_score"].std():.2f})', pad=2)
ps.panel_label(ax, 'q', x=-0.35); save(fig, 'kr_q_zpc2_control')


# ---------------- r-v cell-type interactions at the two ends of z-PC2 (bench_kras_xenium/nodule_te/zpc2_interactions.py)
IA = NT / 'zpc2_interactions'
NG = {'Tumour': ps.DOMAIN[1], 'Endothelial': ps.MAJOR['Vascular'], 'AT1': ps.MAJOR['Astrocyte'], 'AT2': ps.REGION['STR_LS/NDB'],
      'Ciliated': ps.REGION['STR_CP/ACB'], 'Pericyte/SMC': ps.REGION['CTX_L2/3'], 'Fibroblast': ps.MAJOR['Oligodendrocyte'],
      'Macrophage alveolar': ps.REGION['VEN'], 'Macrophage interstitial/mo': ps.MAJOR['Microglia'], 'Monocyte': ps.DOMAIN[0],
      'Neutrophil': ps.REGION['CTX_L1/MEN'], 'DC': ps.REGION['CTX_L4/5/6'], 'T/NK': ps.MAJOR['Neuron'], 'B': ps.REGION['CC/ACO'], 'Other': ps.NEUTRAL['light']}
HL = {'high': ps.DELTA['down'], 'low': ps.DELTA['up']}
# r composition
Cn = pd.read_csv(IA / 'A_niche_composition.csv'); As = pd.read_csv(IA / 'A_summary.csv').sort_values('diff_median')
fig, ax = plt.subplots(figsize=(56 * MM, 50 * MM))
for y, g in enumerate(As.group):
    d = [Cn[(Cn.section == u) & (Cn.level == 'high')][g].mean() - Cn[(Cn.section == u) & (Cn.level == 'low')][g].mean() for u in sorted(Cn.section.unique())]
    ax.scatter(d, np.full(len(d), y) + np.linspace(-0.15, 0.15, len(d)), s=6, c=NG.get(g, ps.NEUTRAL['mid']), edgecolors=ps.INK, lw=0.2, zorder=3)
    md = np.median(d); ax.plot([md, md], [y - 0.3, y + 0.3], color=ps.INK, lw=0.8)
ax.axvline(0, color=ps.INK, lw=0.5); ax.set_yticks(range(len(As))); ax.set_yticklabels(As.group)
ax.set_xlabel('niche share, high − low z-PC2\n(dot = section, bar = median)')
ax.set_title('who surrounds the nodule\n(tumour + 50 µm niche, non-tumour cells)', pad=2); ps.panel_label(ax, 'r', x=-0.75); save(fig, 'kr_r_niche_composition')
# s squidpy nhood enrichment, tumour row
Zh = pd.read_csv(IA / 'B_nhood_z_high.csv', index_col=0).loc['Tumour']; Zl = pd.read_csv(IA / 'B_nhood_z_low.csv', index_col=0).loc['Tumour']
keep = [g for g in Zh.index if g not in ('Other', 'Monocyte') and np.isfinite(Zl[g])]
od = (Zh[keep] - Zl[keep]).sort_values().index
fig, ax = plt.subplots(figsize=(52 * MM, 50 * MM)); yy = np.arange(len(od))
ax.barh(yy + 0.2, Zh[od], height=0.38, color=HL['high'], label='high z-PC2 (worse)'); ax.barh(yy - 0.2, Zl[od], height=0.38, color=HL['low'], label='low z-PC2 (better)')
ax.set_yticks(yy); ax.set_yticklabels(od); ax.axvline(0, color=ps.INK, lw=0.5); ax.set_xlabel('neighbourhood enrichment z\n(tumour vs cell group; squidpy)')
ax.legend(fontsize=5.6, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=1, borderaxespad=0); ax.set_title('tumour contacts', pad=2); ps.panel_label(ax, 's', x=-0.7); save(fig, 'kr_s_nhood_enrichment')
# t LIANA tumour <-> partner LR, top high- and low-specific
Mx = pd.read_csv(IA / 'C_tumour_partner_lr.csv'); Mx = Mx[(Mx.magnitude_rank_high < 0.05) | (Mx.magnitude_rank_low < 0.05)].sort_values('log_ratio')
t = pd.concat([Mx.head(7), Mx.tail(9)])
lab_ = [f"{r.source.replace('Macrophage ', 'Mac ').replace('interstitial/mo', 'int/mo')} → {r.target.replace('Macrophage ', 'Mac ').replace('interstitial/mo', 'int/mo')}: {r.ligand_complex}–{r.receptor_complex}" for r in t.itertuples()]
fig, ax = plt.subplots(figsize=(74 * MM, 58 * MM)); yy = np.arange(len(t))
for col, L, mk in [('magnitude_rank_high', 'high', 'o'), ('magnitude_rank_low', 'low', 's')]:
    v = -np.log10(t[col].clip(lower=1e-4)); ax.scatter(v, yy, s=12, c=HL[L], marker=mk, edgecolors=ps.INK, lw=0.2, zorder=3, label=f'{L} z-PC2 niche')
for y_, (a1, b1) in enumerate(zip(-np.log10(t.magnitude_rank_high.clip(lower=1e-4)), -np.log10(t.magnitude_rank_low.clip(lower=1e-4)))): ax.plot([a1, b1], [y_, y_], color=ps.NEUTRAL['mid'], lw=0.5)
ax.set_yticks(yy); ax.set_yticklabels(lab_, fontsize=5.6); ax.axhline(6.5, color=ps.NEUTRAL['mid'], lw=0.4, ls=':')
ax.set_xlabel('LIANA+ magnitude (−log10 rank)'); ax.legend(fontsize=5.6, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, borderaxespad=0)
ax.set_title('tumour ↔ myeloid / AT1 ligand–receptor\n(top: higher in high niche; bottom: in low)', pad=2); ps.panel_label(ax, 't', x=-1.25); save(fig, 'kr_t_liana')
# u, v example nodules
for L, k, ttl in [('u', 'hi', 'high z-PC2 nodule (worse)'), ('v', 'lo', 'low z-PC2 nodule (better)')]:
    X_ = Dx[f'{k}_xy']; g_ = Dx[f'{k}_g']
    fig, ax = plt.subplots(figsize=(40 * MM, 40 * MM)); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f'{ttl}\nz-PC2 {float(Dx[k + "_score"]):+.2f} · box {L} in a, p', pad=2, color=HL['high' if k == 'hi' else 'low'], fontweight='bold'); ps.panel_label(ax, L, x=-0.04, y=1.03)
    for gname, c in NG.items():
        mm = g_ == gname
        if mm.any(): ax.scatter(*X_[mm].T, s=6 if gname != 'Tumour' else 5, c=c, lw=0.15, edgecolors=ps.INK if gname != 'Tumour' else 'none', zorder=2 if gname == 'Tumour' else 3)
    x0, y0 = X_[:, 0].min(), X_[:, 1].min() - 12; ax.plot([x0, x0 + 50], [y0, y0], color=ps.INK, lw=1); ax.text(x0 + 25, y0 - 6, '50 µm', ha='center', va='top', fontsize=5.8)
    if k == 'lo':
        present = [gname for gname in NG if gname in set(Dx['hi_g']) | set(Dx['lo_g'])]
        leg(ax, [(gname, NG[gname]) for gname in present])
    save(fig, f'kr_{L}_example_{k}')

bad = [n for n in made if not ps.check_fonts(OUT / f'{n}.svg')[0]]; print('panels', len(made), '| fonts below minimum:', bad)
rows = [['kr_a_tissue_tumour', 'kr_b_tissue_mpas', 'kr_c_composition', 'kr_d_subtype_response'],
        ['kr_e_umap_z_type', 'kr_f_umap_s_type', 'kr_g_umap_z_mpas', 'kr_h_umap_s_mpas'],
        ['kr_i_te_axis', 'kr_j_tissue_nodule_te', 'kr_k_te_vs_response', 'kr_l_baseline_predicts_te'],
        ['kr_m_zpc2_genes', 'kr_n_zpc2_neighbours', 'kr_o_zpc2_heldout', 'kr_p_tissue_zpc2', 'kr_q_zpc2_control'],
        ['kr_r_niche_composition', 'kr_s_nhood_enrichment', 'kr_t_liana'],
        ['kr_u_example_hi', 'kr_v_example_lo']]
ims = [[Image.open(OUT / f'{n}.png').convert('RGB') for n in r] for r in rows]; sc = 0.35
ims = [[im.resize((int(im.width * sc), int(im.height * sc)), Image.LANCZOS) for im in r] for r in ims]
W = max(sum(im.width for im in r) + 30 * (len(r) - 1) for r in ims); Hh = sum(max(im.height for im in r) for r in ims) + 30 * (len(ims) - 1)
page = Image.new('RGB', (W + 40, Hh + 40), 'white'); yy = 20
for r in ims:
    xx = 20
    for im in r: page.paste(im, (xx, yy)); xx += im.width + 30
    yy += max(im.height for im in r) + 30
page.save(OUT / 'KRAS_inhibitor_full.png'); print('full page', page.size)
