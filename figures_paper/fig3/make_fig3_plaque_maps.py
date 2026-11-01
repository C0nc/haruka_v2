#!/usr/bin/env python3
"""Figure 3 (plaque) — embedding and spatial views.

  h  CODEX microglia (AD, grey matter): UMAP of z and of s, coloured by distance to the
     nearest Abeta plaque; second row coloured by myeloid state
  i  STARmap microglia (AD slices): UMAP of z and of s, coloured by distance to the nearest
     Abeta-associated cell
  j  CODEX zoom (AD-4, grey matter): held-out plaque-proximity score from z vs from s;
     plaques drawn at their true size
  k  STARmap zoom (slice 1): the same, Abeta-associated cells ringed
UMAP inputs are centred within donor (CODEX) / slice x fine type x domain (STARmap), exactly
as in the statistics, so donor offsets cannot drive the layout. UMAP: 15 neighbours,
min_dist 0.3, random_state 0. Spatial scores are donor/slice-held-out ridge predictions.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
import umap
warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import paperstyle as ps
ps.setup()
H = Path('/home/yancui/haruka_v2')
CP = H / 'bench_codex_cns_2026/wholeslice/plaque_states'; SP = H / 'bench_real_discovery/starmap_plaque_radial'
SEQ = LinearSegmentedColormap.from_list('seq', ps.SEQUENTIAL[::-1])        # near = dark
DIV = LinearSegmentedColormap.from_list('div', ps.DIVERGING)
src = {}


def centre(X, g):
    X = X.astype(np.float64).copy()
    for k in np.unique(g):
        m = g == k; X[m] -= X[m].mean(0)
    return X


def heldout_score(X, dl, types, groups_test, mask):
    out = np.full(len(dl), np.nan)
    for t in np.unique(types[mask]):
        for gt in np.unique(groups_test[mask]):
            tr = mask & (types == t) & (groups_test != gt); te = mask & (types == t) & (groups_test == gt)
            sc = StandardScaler().fit(X[tr]); r = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X[tr]), dl[tr])
            out[te] = -r.predict(sc.transform(X[te]))
        m = mask & (types == t); out[m] = (out[m] - out[m].mean()) / out[m].std()
    return out


def umap_pair(Xz, Xs, colour, cmode, title, fname, letter, cbar_label, vmin=None, vmax=None, cats=None):
    fig, axs = plt.subplots(1, 2, figsize=(84 * ps.MM, 40 * ps.MM))
    for ax, X, key in zip(axs, [Xz, Xs], ['z', 's']):
        E = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(StandardScaler().fit_transform(X))
        if cmode == 'cont':
            o = np.argsort(-colour)                                     # near-plaque cells drawn last (on top)
            sc = ax.scatter(E[o, 0], E[o, 1], c=colour[o], cmap=SEQ, vmin=vmin, vmax=vmax, s=0.8, lw=0, rasterized=True)
        else:
            for c in cats:
                m = colour == c
                ax.scatter(E[m, 0], E[m, 1], color=ps.MYELOID[c], s=0.8, lw=0, rasterized=True, label=c)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_visible(False)
        ax.set_title(f'{key}', color=ps.LATENT[key], fontweight='bold', pad=1)
        ax.set_xlabel('UMAP 1', labelpad=1); ax.set_ylabel('UMAP 2', labelpad=1)
        src[f'{fname}_{key}'] = pd.DataFrame({'umap1': E[:, 0], 'umap2': E[:, 1], 'colour': colour})
    if cmode == 'cont':
        cb = fig.colorbar(sc, ax=axs, fraction=0.035, pad=0.02); cb.outline.set_linewidth(.4); cb.set_label(cbar_label, labelpad=1)
    else:
        h = [plt.Line2D([], [], marker='o', ls='', color=ps.MYELOID[c], ms=3) for c in cats]
        fig.legend(h, [c.replace(' microglia', '') for c in cats], loc='center right', bbox_to_anchor=(1.12, 0.5), handletextpad=0.2)
    fig.suptitle(title, y=1.04, fontsize=6.5)
    fig.text(0.0, 1.04, letter, fontsize=ps.PANEL_LABEL_PT, fontweight='bold', va='center')
    ps.save(fig, HERE / fname); plt.close(fig)


# ================= CODEX =================
a = ad.read_h5ad(H / 'bench_codex_cns_2026/wholeslice/codex_wholeslice_all7_no_oligo.h5ad'); o = a.obs
MD = H / 'results_codex_wholeslice/codex_wholeslice_all7_no_oligo/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
don = o.donor_id.astype(str).to_numpy(); ct = o.cell_type.astype(str).to_numpy(); xy = a.obsm['spatial_um']
dist = np.load(CP / 'cell_dist_any_abeta_um.npy'); dl = np.log1p(np.minimum(dist, 150))
use = (o.condition.astype(str).to_numpy() == 'disease') & (o.region.to_numpy() == 'GM')
g = (o.donor_id.astype(str) + '|' + o.cell_type.astype(str)).to_numpy()
Zc, Sc = centre(np.load(MD / 'background.npy'), g), centre(np.load(MD / 'salient.npy'), g)
mg = use & (ct == 'Microglia')
umap_pair(Zc[mg], Sc[mg], np.minimum(dist[mg], 60), 'cont', f'CODEX microglia, 4 AD donors (n = {mg.sum():,})',
          'fig3h_codex_microglia_umap_distance', 'h', 'Distance to plaque (µm,\ncapped at 60)', 0, 60)
MS = ['Homeostatic microglia', 'MHC-II+ microglia', 'Activated microglia', 'Macrophage', 'Unresolved']
umap_pair(Zc[mg], Sc[mg], o.myeloid_state.astype(str).to_numpy()[mg], 'cat', 'CODEX microglia, myeloid state',
          'fig3h2_codex_microglia_umap_state', 'h', '', cats=MS)

# spatial zoom: AD-4 grey matter, the 500 um window with the most dense plaques
scZ = heldout_score(Zc, dl, ct, don, use); scS = heldout_score(Sc, dl, ct, don, use)
pl = pd.read_csv(H / 'bench_codex_cns_2026/external/CODEX-CNS/codex_notebooks/marker_intensities/12032021-Bahareh_3280_Alz_abeta.csv',
                 usecols=['class', 'x', 'y', 'area_um2'])
pxy = pl[['x', 'y']].to_numpy() * 0.3774; prad = np.sqrt(pl.area_um2.to_numpy() / np.pi)
W = 500; m4 = don == 'AD-4'; best = None
for x0 in np.arange(xy[m4, 0].min(), xy[m4, 0].max() - W, 100):
    for y0 in np.arange(xy[m4, 1].min(), xy[m4, 1].max() - W, 100):
        inw = m4 & (xy[:, 0] >= x0) & (xy[:, 0] < x0 + W) & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + W)
        if inw.sum() < 150 or (inw & use).sum() / inw.sum() < 0.95: continue
        inp = (pxy[:, 0] >= x0) & (pxy[:, 0] < x0 + W) & (pxy[:, 1] >= y0) & (pxy[:, 1] < y0 + W)
        far = (dist[inw] > 60).mean()
        if not 0.35 <= far <= 0.65: continue                       # half the cells plaque-free: contrast is visible
        n = (inp & (pl['class'] == 'dense').to_numpy()).sum()
        if best is None or n > best[0]: best = (n, x0, y0)
_, x0, y0 = best
win = use & m4 & (xy[:, 0] >= x0) & (xy[:, 0] < x0 + W) & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + W)
inp = (pxy[:, 0] >= x0 - 30) & (pxy[:, 0] < x0 + W + 30) & (pxy[:, 1] >= y0 - 30) & (pxy[:, 1] < y0 + W + 30)


def zoom_pair(xyw, scores, plaques, title, fname, letter, bar, bar_label, dot):
    fig, axs = plt.subplots(1, 2, figsize=(84 * ps.MM, 44 * ps.MM))
    for ax, key in zip(axs, ['z', 's']):
        for (px, py, r, dense) in plaques:
            style = {'dense': dict(facecolor=ps.NEUTRAL['light'], lw=0.5, ls='-'), 'diffuse': dict(facecolor='none', lw=0.35, ls=(0, (2, 1.5))),
                     'cell': dict(facecolor='none', lw=0.7, ls='-')}[dense]
            ax.add_patch(Circle((px, py), r, edgecolor=ps.INK, zorder=3 if dense == 'cell' else 1, **style))
        o_ = np.argsort(scores[key])
        sc = ax.scatter(xyw[o_, 0], xyw[o_, 1], c=scores[key][o_], cmap=DIV, vmin=-2, vmax=2, s=dot, lw=0.2,
                        edgecolors='white', zorder=2, rasterized=True)
        ax.set_xlim(xyw[:, 0].min() - 5, xyw[:, 0].max() + 5); ax.set_ylim(xyw[:, 1].max() + 5, xyw[:, 1].min() - 5)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_visible(False)
        ax.set_title(f'{key}', color=ps.LATENT[key], fontweight='bold', pad=1)
    xl, yl = axs[0].get_xlim(), axs[0].get_ylim()
    axs[0].plot([xl[0], xl[0] + bar], [yl[0] + (yl[0] - yl[1]) * 0.04] * 2, color=ps.INK, lw=1, clip_on=False)
    axs[0].text(xl[0] + bar / 2, yl[0] + (yl[0] - yl[1]) * 0.07, bar_label, ha='center', va='top')
    cb = fig.colorbar(sc, ax=axs, fraction=0.035, pad=0.02); cb.set_ticks([-2, 0, 2]); cb.outline.set_linewidth(.4)
    cb.set_label('Held-out plaque-\nproximity score', labelpad=1)
    fig.suptitle(title, y=1.03, fontsize=6.5, x=0.5); fig.text(0.0, 1.03, letter, fontsize=ps.PANEL_LABEL_PT, fontweight='bold', va='center')
    ps.save(fig, HERE / fname); plt.close(fig)

plaques = [(pxy[i, 0], pxy[i, 1], prad[i], pl['class'].iloc[i]) for i in np.where(inp)[0]]
zoom_pair(xy[win], {'z': scZ[win], 's': scS[win]}, plaques, 'CODEX AD-4, grey matter (filled = dense, dashed = diffuse plaque)',
          'fig3j_codex_zoom', 'j', 100, '100 µm', 7)
r = {k: np.corrcoef(v[win], -dl[win])[0, 1] for k, v in {'z': scZ, 's': scS}.items()}
src['j'] = pd.DataFrame({'x_um': xy[win, 0], 'y_um': xy[win, 1], 'cell_type': ct[win], 'dist_um': dist[win], 'z_score': scZ[win], 's_score': scS[win]})

# ================= STARmap =================
b = ad.read_h5ad(H / 'bench_real_discovery/data/ad.h5ad'); ob = b.obs
MDs = H / 'results_real_compgated_lambda10_samplebatch_sepheads/ad/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
CROSS = {'CTX-Ex': 'Neuron', 'CA1': 'Neuron', 'CA2': 'Neuron', 'CA3': 'Neuron', 'DG': 'Neuron', 'Inh': 'Neuron', 'LHb': 'Neuron',
         'Astro': 'Astrocyte', 'Oligo': 'Oligodendrocyte', 'OPC': 'Oligodendrocyte', 'Micro': 'Microglia', 'Endo': 'Vascular', 'SMC': 'Vascular'}
maj = pd.Series(ob.ct_top.astype(str).to_numpy()).map(CROSS).to_numpy()
Dt = pd.read_csv(H / 'final_results_nb/starmap_ad/biology/plaque_distance/starmap_plaque_distance_cells.csv.gz').set_index('cell_id')
dsb = pd.Series(b.obs_names).map(Dt.distance_to_abeta_associated_cell_spatial_units).to_numpy(float)
SPC = 31.4; dlb = np.log1p(np.minimum(dsb, 16 * SPC)); slb = ob.slice_id.astype(str).to_numpy(); xyb = b.obsm['spatial'].astype(float)
gb = (ob.slice_id.astype(str) + '|' + ob.ct_top.astype(str) + '|' + ob.tissue_domain.astype(str)).to_numpy()
useb = np.isin(slb, ['disease_r1', 'disease_r2']) & np.isfinite(dlb)
Zb, Sb = centre(np.load(MDs / 'background.npy'), gb), centre(np.load(MDs / 'salient.npy'), gb)
mgb = useb & (maj == 'Microglia')
umap_pair(Zb[mgb], Sb[mgb], np.minimum(dsb[mgb] / SPC, 8), 'cont', f'STARmap microglia, 2 AD slices (n = {mgb.sum():,})',
          'fig3i_starmap_microglia_umap_distance', 'i', 'Distance to Aβ cell\n(cell spacings)', 0, 8)
sbZ = heldout_score(Zb, dlb, maj, slb, useb); sbS = heldout_score(Sb, dlb, maj, slb, useb)
ab = ob.plaque_abeta.to_numpy() == 1; m1 = slb == 'disease_r1'
W2 = 1500; best = None
for x0 in np.arange(xyb[m1, 0].min(), xyb[m1, 0].max() - W2, 250):
    for y0 in np.arange(xyb[m1, 1].min(), xyb[m1, 1].max() - W2, 250):
        inw = m1 & (xyb[:, 0] >= x0) & (xyb[:, 0] < x0 + W2) & (xyb[:, 1] >= y0) & (xyb[:, 1] < y0 + W2)
        n = (inw & ab).sum()
        if best is None or n > best[0]: best = (n, x0, y0)
_, x0, y0 = best
winb = useb & m1 & (xyb[:, 0] >= x0) & (xyb[:, 0] < x0 + W2) & (xyb[:, 1] >= y0) & (xyb[:, 1] < y0 + W2)
rings = [(xyb[i, 0], xyb[i, 1], 30, 'cell') for i in np.where(winb & ab)[0]]
zoom_pair(xyb[winb], {'z': sbZ[winb], 's': sbS[winb]}, rings, 'STARmap slice 1 (rings = Aβ-associated cells)',
          'fig3k_starmap_zoom', 'k', 10 * SPC, '10 cell spacings', 5)
rb = {k: np.corrcoef(v[winb], -dlb[winb])[0, 1] for k, v in {'z': sbZ, 's': sbS}.items()}
src['k'] = pd.DataFrame({'x': xyb[winb, 0], 'y': xyb[winb, 1], 'major_type': maj[winb], 'dist_units': dsb[winb], 'z_score': sbZ[winb], 's_score': sbS[winb]})

for k, v in src.items(): v.to_csv(HERE / f'fig3{k}_source_data.csv.gz' if len(k) == 1 else HERE / f'{k}_source_data.csv.gz', index=False, compression='gzip')
for f in ['fig3h_codex_microglia_umap_distance', 'fig3h2_codex_microglia_umap_state', 'fig3i_starmap_microglia_umap_distance', 'fig3j_codex_zoom', 'fig3k_starmap_zoom']:
    ok, sz = ps.check_fonts(HERE / f'{f}.svg'); print(f, 'min font', min(sz), 'OK' if ok else 'BELOW MINIMUM')
print('zoom r(score, proximity): CODEX', {k: round(v, 3) for k, v in r.items()}, 'STARmap', {k: round(v, 3) for k, v in rb.items()})
