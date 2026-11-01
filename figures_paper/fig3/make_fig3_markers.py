#!/usr/bin/env python3
"""Figure 3 — the biology behind the s plaque axis, human vs mouse.

  q  microglia, 20 protein/ortholog pairs: plaque gradient human vs mouse (left) and loading on
     s's plaque axis human vs mouse (right); Spearman + permutation p
  r  microglia + astrocytes: marker x {gradient, s beyond z} for both species (heatmap)
  s  STARmap programmes (DAM, homeostatic microglia, disease-associated astrocytes): loading on z vs s
  t  Haruka s UMAP of microglia coloured by the conserved and the discordant markers
     (STARmap: Cst7, Apoe, P2ry12; CODEX: ApoE, CD11c, TMEM119)
Statistics: within-group Pearson (features and proximity centred within donor|slice x type [x domain]);
held-out z/s plaque scores as in the radial tests. Source: bench_codex_cns_2026/wholeslice/plaque_states/cross_species.
"""
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import umap
warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import paperstyle as ps
ps.setup()
H = Path(f'{_HR}'); X = H / 'bench_codex_cns_2026/wholeslice/plaque_states/cross_species'
DIV = LinearSegmentedColormap.from_list('div', ps.DIVERGING); SEQ = LinearSegmentedColormap.from_list('seq', ps.SEQUENTIAL)
O = pd.read_csv(X / 'cross_species_orthologs.csv'); rng = np.random.default_rng(0)


def fonts(f):
    ok, sz = ps.check_fonts(HERE / f'{f}.svg')
    if not ok: print(f, 'FONT BELOW MINIMUM', sz)

# ---------- q: cross-species scatter (microglia) ----------
M = O[O.cell_type == 'Microglia']
fig, axs = plt.subplots(1, 2, figsize=(92 * ps.MM, 46 * ps.MM))
for ax, k, lab in zip(axs, ['grad', 'load_s'], ['Plaque gradient', "Loading on s's plaque axis"]):
    x, y = M[f'{k}_codex'].to_numpy(), M[f'{k}_starmap'].to_numpy()
    r = spearmanr(x, y).statistic; p = np.mean([abs(spearmanr(x, rng.permutation(y)).statistic) >= abs(r) for _ in range(5000)])
    agree = np.sign(x) == np.sign(y)
    ax.axhline(0, color=ps.NEUTRAL['mid'], lw=0.5); ax.axvline(0, color=ps.NEUTRAL['mid'], lw=0.5)
    ax.scatter(x, y, s=10, c=np.where(agree, ps.DELTA['up'], ps.DELTA['down']), lw=0, zorder=3)
    OFF = {'grad': {'ApoE': (-24, -9), 'CD11c': (3, 3), 'CD68': (4, -2), 'CD74': (-22, 4), 'GFAP': (-6, 5), 'TMEM119': (-36, 2)},
           'load_s': {'ApoE': (-24, -9), 'CD11c': (-10, 5), 'GFAP': (4, 3), 'CD68': (4, -6), 'CD74': (4, -2), 'Gal3': (-16, 4),
                      'Vimentin': (3, -7), 'PSD95': (4, -2)}}[k]
    for xi, yi, n in zip(x, y, M.protein):
        if n in OFF: ax.annotate(n, (xi, yi), xytext=OFF[n], textcoords='offset points', fontsize=5.6)
    ax.set_xlabel('Human (CODEX protein)'); ax.set_ylabel('Mouse (STARmap RNA)')
    ax.set_title(f'{lab}\nρ = {r:+.2f}, p = {p:.3f}', pad=2)
fig.text(0.5, 1.04, 'Microglia, 20 protein–ortholog pairs', ha='center', fontsize=6.5)
fig.text(0.0, 1.04, 'q', fontsize=ps.PANEL_LABEL_PT, fontweight='bold'); plt.tight_layout(w_pad=1.5)
ps.save(fig, HERE / 'fig3q_cross_species_microglia'); plt.close(fig); fonts('fig3q_cross_species_microglia')

# ---------- r: heatmap ----------
SHOW = ['ApoE', 'CD11c', 'CD68', 'CD74', 'Gal3', 'CD44', 'Vimentin', 'GFAP', 'TMEM119', 'IBA1', 'S100B', 'AQP4', 'PSD95']
cols = [('grad_codex', 'Human\ngradient'), ('s_only_codex', 'Human\ns beyond z'), ('grad_starmap', 'Mouse\ngradient'), ('s_only_starmap', 'Mouse\ns beyond z')]
fig, axs = plt.subplots(1, 2, figsize=(92 * ps.MM, 58 * ps.MM), sharey=True)
for ax, t in zip(axs, ['Microglia', 'Astrocyte']):
    T = O[O.cell_type == t].set_index('protein').loc[SHOW]
    V = T[[c for c, _ in cols]].to_numpy()
    im = ax.imshow(V, cmap=DIV, vmin=-0.35, vmax=0.35, aspect='auto')
    ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in cols], rotation=90)
    ax.set_yticks(range(len(SHOW))); ax.set_yticklabels([f'{p} / {T.loc[p, "gene"]}' for p in SHOW])
    ax.axvline(1.5, color='white', lw=1.5); ax.set_title(t, pad=2); ax.tick_params(length=0)
    for sp_ in ax.spines.values(): sp_.set_visible(False)
cb = fig.colorbar(im, ax=axs, fraction=0.04, pad=0.02); cb.set_label('Correlation with plaque proximity', labelpad=1)
cb.set_ticks([-0.3, 0, 0.3]); cb.outline.set_linewidth(.4)
fig.text(0.0, 1.0, 'r', fontsize=ps.PANEL_LABEL_PT, fontweight='bold')
ps.save(fig, HERE / 'fig3r_marker_heatmap'); plt.close(fig); fonts('fig3r_marker_heatmap')

# ---------- s: STARmap programmes, z vs s ----------
S = pd.read_csv(X / 'marker_axes_starmap.csv')
PROG = {'DAM': ['Trem2', 'Tyrobp', 'Cst7', 'Apoe', 'Lpl', 'Cd9', 'Itgax', 'Spp1', 'Clec7a', 'Cd68', 'Ctsb', 'Ctsd', 'Axl', 'Ccl6'],
        'Homeostatic\nmicroglia': ['P2ry12', 'Tmem119', 'Cx3cr1', 'Sall1', 'Selplg', 'Csf1r'],
        'Disease-assoc.\nastrocyte': ['Gfap', 'Vim', 'Serpina3n', 'C4b', 'Cd44', 'Ctss', 'Osmr', 'Lcn2']}
CT = {'DAM': 'Microglia', 'Homeostatic\nmicroglia': 'Microglia', 'Disease-assoc.\nastrocyte': 'Astrocyte'}
fig, ax = plt.subplots(figsize=(58 * ps.MM, 44 * ps.MM)); rows = []
for i, (p, g) in enumerate(PROG.items()):
    x = S[S.cell_type == CT[p]].set_index('feature'); g = [k for k in g if k in x.index]
    for j, k in enumerate(['load_z', 'load_s']):
        v = x.loc[g, k].to_numpy(); xp = i + (j - 0.5) * 0.36
        ax.bar(xp, v.mean(), width=0.32, color=ps.LATENT[k[-1]], lw=0)
        ax.scatter(xp + rng.uniform(-0.08, 0.08, len(v)), v, s=3, color=ps.INK, lw=0, zorder=3)
        rows.append(dict(programme=p.replace('\n', ' '), cell_type=CT[p], rep=k[-1], n_genes=len(g), mean=v.mean()))
ax.axhline(0, color=ps.NEUTRAL['mid'], lw=0.5); ax.set_xticks(range(3)); ax.set_xticklabels(PROG.keys())
ax.set_ylabel('Gene loading on plaque axis')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=ps.LATENT['z'], label='z'), Patch(color=ps.LATENT['s'], label='s')], loc='upper right', handlelength=1)
ax.set_title('STARmap programmes (dots = genes)', pad=2); ps.panel_label(ax, 's', x=-0.3, y=1.04)
ps.save(fig, HERE / 'fig3s_programmes_z_vs_s'); plt.close(fig); fonts('fig3s_programmes_z_vs_s')
pd.DataFrame(rows).to_csv(HERE / 'fig3s_source_data.csv', index=False)

# ---------- t: Haruka s UMAP of microglia, markers mapped back ----------
def centre(A, g):
    A = A.astype(np.float64).copy()
    for k in np.unique(g):
        m = g == k; A[m] -= A[m].mean(0)
    return A

b = ad.read_h5ad(H / 'bench_real_discovery/data/ad.h5ad'); ob = b.obs
MDs = H / 'results_real_compgated_lambda10_samplebatch_sepheads/ad/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
mb = (ob.condition.astype(str) == 'disease').to_numpy() & (ob.ct_top.astype(str) == 'Micro').to_numpy()
Eb = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(StandardScaler().fit_transform(np.load(MDs / 'salient.npy')[mb]))
Cn = b.layers['counts']; Cn = Cn.tocsr() if sp.issparse(Cn) else sp.csr_matrix(Cn)
Lb = Cn.multiply(1e4 / np.asarray(Cn.sum(1)).ravel()[:, None]).tocsr(); Lb.data = np.log1p(Lb.data)
a = ad.read_h5ad(H / 'bench_codex_cns_2026/wholeslice/codex_wholeslice_all7_no_oligo.h5ad'); o = a.obs
MD = H / 'results_codex_wholeslice/codex_wholeslice_all7_no_oligo/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
ma = (o.condition.astype(str) == 'disease').to_numpy() & (o.region.astype(str) == 'GM').to_numpy() & (o.cell_type.astype(str) == 'Microglia').to_numpy()
Ea = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(StandardScaler().fit_transform(np.load(MD / 'salient.npy')[ma]))
dist = np.load(H / 'bench_codex_cns_2026/wholeslice/plaque_states/cell_dist_any_abeta_um.npy')
Dt = pd.read_csv(H / 'final_results_nb/starmap_ad/biology/plaque_distance/starmap_plaque_distance_cells.csv.gz').set_index('cell_id')
dsb = pd.Series(b.obs_names).map(Dt.distance_to_abeta_associated_cell_spatial_units).to_numpy(float)
gn = list(b.var_names); pn = list(a.var_names)
panels = [('STARmap', Eb, 'Distance to Aβ\n(cell spacings)', np.minimum(dsb[mb] / 31.4, 8), True),
          ('STARmap', Eb, 'Cst7 (DAM)', np.asarray(Lb[:, gn.index('Cst7')].todense()).ravel()[mb], False),
          ('STARmap', Eb, 'Apoe', np.asarray(Lb[:, gn.index('Apoe')].todense()).ravel()[mb], False),
          ('STARmap', Eb, 'P2ry12 (homeostatic)', np.asarray(Lb[:, gn.index('P2ry12')].todense()).ravel()[mb], False),
          ('CODEX', Ea, 'Distance to plaque\n(µm, cap 60)', np.minimum(dist[ma], 60), True),
          ('CODEX', Ea, 'ApoE', np.asarray(a.X[ma][:, pn.index('ApoE')]).ravel(), False),
          ('CODEX', Ea, 'CD11c (Itgax)', np.asarray(a.X[ma][:, pn.index('CD11c')]).ravel(), False),
          ('CODEX', Ea, 'TMEM119', np.asarray(a.X[ma][:, pn.index('TMEM119')]).ravel(), False)]
fig, axs = plt.subplots(2, 4, figsize=(183 * ps.MM, 86 * ps.MM))
for ax, (ds, E, lab, v, is_d) in zip(axs.ravel(), panels):
    if is_d:
        o_ = np.argsort(-v); sc = ax.scatter(E[o_, 0], E[o_, 1], c=v[o_], cmap=LinearSegmentedColormap.from_list('s', ps.SEQUENTIAL[::-1]), s=1.2 if ds == 'STARmap' else 0.4, lw=0, rasterized=True)
    else:
        lo, hi = np.percentile(v, [2, 98]); o_ = np.argsort(v)
        sc = ax.scatter(E[o_, 0], E[o_, 1], c=np.clip(v[o_], lo, hi), cmap=SEQ, s=1.2 if ds == 'STARmap' else 0.4, lw=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values(): sp_.set_visible(False)
    ax.set_title(lab, pad=1)
    cb = fig.colorbar(sc, ax=ax, orientation='horizontal', fraction=0.05, pad=0.03, aspect=20); cb.outline.set_linewidth(.4)
    if not is_d: cb.set_ticks([lo, hi]); cb.set_ticklabels(['low', 'high'])
axs[0, 0].set_ylabel(f'STARmap microglia\ns UMAP (n = {mb.sum():,})'); axs[1, 0].set_ylabel(f'CODEX microglia, GM\ns UMAP (n = {ma.sum():,})')
fig.text(0.0, 1.0, 't', fontsize=ps.PANEL_LABEL_PT, fontweight='bold'); plt.tight_layout(h_pad=1.0, w_pad=0.6)
ps.save(fig, HERE / 'fig3t_s_umap_markers'); plt.close(fig); fonts('fig3t_s_umap_markers')
print('done')
