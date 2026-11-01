#!/usr/bin/env python3
"""Figure 3 — Haruka results first, follow-up evidence mapped back onto them.

One block per dataset (2 rows x 4 panels, each its own SVG):
  row 1  Haruka UMAPs (raw z and raw s of the AD cells, as the model outputs them)
         u1 z UMAP, cell type | u2 s UMAP, cell type | u3 z UMAP, distance to plaque |
         u4 s UMAP, distance to plaque
  row 2  the same evidence in tissue (one AD section)
         m1 cell type + plaques | m2 held-out plaque-proximity score from z |
         m3 the same from s | m4 a Haruka-native output:
            STARmap = counterfactual salient microglia-share shift D([z,s]) - D([z,0])
            CODEX   = s-CC domain (KMeans k=12) coloured by its plaque log2 enrichment
Held-out scores come from the follow-up analysis (per-cell-type ridge on log distance,
donor/slice held out; features centred within donor/slice x type [x domain]).
UMAP: 15 neighbours, min_dist 0.3, random_state 0; CODEX subsampled to 40,000 AD cells.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
import umap
warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import paperstyle as ps
ps.setup()
H = Path('/home/yancui/haruka_v2')
SEQ = LinearSegmentedColormap.from_list('seq', ps.SEQUENTIAL[::-1]); DIV = LinearSegmentedColormap.from_list('div', ps.DIVERGING)
SZ = 46 * ps.MM                                                   # each panel 46 x 46 mm; 4 per row = 184 mm


def centre(X, g):
    X = X.astype(np.float64).copy()
    for k in np.unique(g):
        m = g == k; X[m] -= X[m].mean(0)
    return X


def heldout_score(X, dl, types, gtest, mask):
    out = np.full(len(dl), np.nan)
    for t in np.unique(types[mask]):
        for gt in np.unique(gtest[mask]):
            tr = mask & (types == t) & (gtest != gt); te = mask & (types == t) & (gtest == gt)
            sc = StandardScaler().fit(X[tr]); r = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X[tr]), dl[tr])
            out[te] = -r.predict(sc.transform(X[te]))
        m = mask & (types == t); out[m] = (out[m] - out[m].mean()) / out[m].std()
    return out


def panel(fname, letter, title, title_col=None):
    fig, ax = plt.subplots(figsize=(SZ, SZ))
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.set_title(title, pad=2, color=title_col or ps.INK, fontweight='bold' if title_col else 'normal')
    if letter: ps.panel_label(ax, letter, x=-0.06, y=1.06)
    return fig, ax


def finish(fig, ax, fname, sc=None, cbar=None, ticks=None, legend=None, spatial=False, bar=None):
    if spatial:
        ax.set_aspect('equal'); ax.invert_yaxis()
        if bar:
            L, lab = bar; xl, yl = ax.get_xlim(), ax.get_ylim()
            ax.plot([xl[0], xl[0] + L], [yl[0]] * 2, color=ps.INK, lw=1, clip_on=False)
            ax.text(xl[0] + L / 2, yl[0] + 0.02 * (yl[0] - yl[1]), lab, ha='center', va='top')
    else:
        ax.set_xlabel('UMAP 1', labelpad=1); ax.set_ylabel('UMAP 2', labelpad=1)
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, orientation='horizontal', fraction=0.05, pad=0.04 if spatial else 0.1, aspect=25)
        cb.outline.set_linewidth(.4); cb.set_label(cbar, labelpad=1)
        if ticks is not None: cb.set_ticks(ticks)
    if legend is not None:
        h = [plt.Line2D([], [], marker='o', ls='', color=c, ms=3) for c in legend.values()]
        ax.legend(h, list(legend.keys()), loc='upper center', bbox_to_anchor=(0.5, -0.04), ncol=3, handletextpad=0.1,
                  columnspacing=0.6, frameon=False)
    ps.save(fig, HERE / fname); plt.close(fig)
    ok, sz = ps.check_fonts(HERE / f'{fname}.svg')
    if not ok: print(fname, 'FONT BELOW MINIMUM', sz)


def scatter(ax, xy, c=None, cmap=None, norm=None, vmin=None, vmax=None, colors=None, s=0.6, order=None):
    o = np.arange(len(xy)) if order is None else order
    if colors is not None:
        return ax.scatter(xy[o, 0], xy[o, 1], color=colors[o], s=s, lw=0, rasterized=True)
    return ax.scatter(xy[o, 0], xy[o, 1], c=c[o], cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, s=s, lw=0, rasterized=True)


def block(tag, L, Zr, Sr, major, dist, dist_lab, dist_cap, scoreZ, scoreS, xy, sec, plaques, plaque_ms, native, native_lab,
          native_cmap, native_norm, bar, letters):
    MAJ = {k: v for k, v in ps.MAJOR.items() if k in set(major)}
    col = np.array([ps.MAJOR[t] for t in major])
    emb = {}
    for key, X in [('z', Zr), ('s', Sr)]:
        emb[key] = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(StandardScaler().fit_transform(X[L]))
    # ---- row 1: UMAPs ----
    for i, key in enumerate(['z', 's']):
        fig, ax = panel(None, letters[i], f'{key}: cell type', ps.LATENT[key])
        scatter(ax, emb[key], colors=col[L], order=np.random.default_rng(0).permutation(L.sum()))
        finish(fig, ax, f'fig3_{tag}_u{i + 1}_{key}_umap_celltype', legend=MAJ)
    dcap = np.minimum(dist[L], dist_cap); o = np.argsort(-dcap)
    for i, key in enumerate(['z', 's']):
        fig, ax = panel(None, letters[2 + i], f'{key}: distance to plaque', ps.LATENT[key])
        sc = scatter(ax, emb[key], c=dcap, cmap=SEQ, vmin=0, vmax=dist_cap, order=o)
        finish(fig, ax, f'fig3_{tag}_u{3 + i}_{key}_umap_distance', sc, dist_lab)
    # ---- row 2: tissue ----
    fig, ax = panel(None, letters[4], 'Cell type + plaques')
    scatter(ax, xy[sec], colors=col[sec], s=1.2, order=np.random.default_rng(0).permutation(sec.sum()))
    ax.scatter(plaques[:, 0], plaques[:, 1], s=plaque_ms, facecolors='none', edgecolors=ps.INK, lw=0.4)
    finish(fig, ax, f'fig3_{tag}_m1_tissue_celltype', spatial=True, bar=bar)
    for i, (key, scv) in enumerate([('z', scoreZ), ('s', scoreS)]):
        fig, ax = panel(None, letters[5 + i], f'{key}: plaque-proximity score', ps.LATENT[key])
        m = sec & np.isfinite(scv)
        ax.scatter(xy[sec & ~np.isfinite(scv), 0], xy[sec & ~np.isfinite(scv), 1], color=ps.NEUTRAL['light'], s=1.2, lw=0, rasterized=True)
        sc = scatter(ax, xy[m], c=scv[m], cmap=DIV, vmin=-2, vmax=2, s=1.2, order=np.argsort(scv[m]))
        ax.scatter(plaques[:, 0], plaques[:, 1], s=plaque_ms, facecolors='none', edgecolors=ps.INK, lw=0.3)
        finish(fig, ax, f'fig3_{tag}_m{2 + i}_tissue_{key}_score', sc, 'Held-out score (SD)', [-2, 0, 2], spatial=True, bar=bar)
    fig, ax = panel(None, letters[7], native_lab[0], ps.LATENT['s'])
    m = sec & np.isfinite(native)
    ax.scatter(xy[sec & ~m, 0], xy[sec & ~m, 1], color=ps.NEUTRAL['light'], s=1.2, lw=0, rasterized=True)
    sc = scatter(ax, xy[m], c=native[m], cmap=native_cmap, norm=native_norm, s=1.2, order=np.argsort(np.abs(native[m])))
    ax.scatter(plaques[:, 0], plaques[:, 1], s=plaque_ms, facecolors='none', edgecolors=ps.INK, lw=0.3)
    finish(fig, ax, f'fig3_{tag}_m4_tissue_haruka_native', sc, native_lab[1], spatial=True, bar=bar)
    return emb


# ======================= STARmap =======================
b = ad.read_h5ad(H / 'bench_real_discovery/data/ad.h5ad'); ob = b.obs
MDs = H / 'results_real_compgated_lambda10_samplebatch_sepheads/ad/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
CROSS = {'CTX-Ex': 'Neuron', 'CA1': 'Neuron', 'CA2': 'Neuron', 'CA3': 'Neuron', 'DG': 'Neuron', 'Inh': 'Neuron', 'LHb': 'Neuron',
         'Astro': 'Astrocyte', 'Oligo': 'Oligodendrocyte', 'OPC': 'Oligodendrocyte', 'Micro': 'Microglia', 'Endo': 'Vascular', 'SMC': 'Vascular'}
maj = pd.Series(ob.ct_top.astype(str).to_numpy()).map(CROSS).to_numpy()
Dt = pd.read_csv(H / 'final_results_nb/starmap_ad/biology/plaque_distance/starmap_plaque_distance_cells.csv.gz').set_index('cell_id')
dsb = pd.Series(b.obs_names).map(Dt.distance_to_abeta_associated_cell_spatial_units).to_numpy(float)
SPC = 31.4; dlb = np.log1p(np.minimum(dsb, 16 * SPC)); slb = ob.slice_id.astype(str).to_numpy(); xyb = b.obsm['spatial'].astype(float)
gb = (ob.slice_id.astype(str) + '|' + ob.ct_top.astype(str) + '|' + ob.tissue_domain.astype(str)).to_numpy()
dis = np.isin(slb, ['disease_r1', 'disease_r2']) & np.isfinite(dlb)
Zr, Sr = np.load(MDs / 'background.npy'), np.load(MDs / 'salient.npy')
scZ = heldout_score(centre(Zr, gb), dlb, maj, slb, dis); scS = heldout_score(centre(Sr, gb), dlb, maj, slb, dis)
shift = np.load(MDs / 'salient_composition_shift.npy')[:, 9]
sec = dis & (slb == 'disease_r1'); pl = xyb[sec & (ob.plaque_abeta.to_numpy() == 1)]
lim = np.abs(np.percentile(shift[dis], [1, 99])).max()
block('starmap', dis, Zr, Sr, maj, dsb / SPC, 'Distance to Aβ cell (cell spacings)', 8, scZ, scS, xyb, sec, pl, 1.5,
      shift, ('s-driven microglia shift', 'Haruka Δ predicted microglia share'), DIV, TwoSlopeNorm(0, -lim, lim),
      (1000, '1,000 units'), list('abcdefgh'))
print('STARmap done', flush=True)

# ======================= CODEX =======================
a = ad.read_h5ad(H / 'bench_codex_cns_2026/wholeslice/codex_wholeslice_all7_no_oligo.h5ad'); o = a.obs
MD = H / 'results_codex_wholeslice/codex_wholeslice_all7_no_oligo/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
don = o.donor_id.astype(str).to_numpy(); ct = o.cell_type.astype(str).to_numpy(); xy = a.obsm['spatial_um']
dist = np.load(H / 'bench_codex_cns_2026/wholeslice/plaque_states/cell_dist_any_abeta_um.npy'); dl = np.log1p(np.minimum(dist, 150))
adm = o.condition.astype(str).to_numpy() == 'disease'; gmm = adm & (o.region.to_numpy() == 'GM')
g = (o.donor_id.astype(str) + '|' + o.cell_type.astype(str)).to_numpy()
Zc, Sc = np.load(MD / 'background.npy'), np.load(MD / 'salient.npy')
scZc = heldout_score(centre(Zc, g), dl, ct, don, gmm); scSc = heldout_score(centre(Sc, g), dl, ct, don, gmm)
# s-CC domain enrichment computed WITHIN grey matter: white-matter domains are trivially plaque-free
D = pd.read_csv(HERE / 'haruka_domains_codex_wholeslice.csv.gz'); rows = D.obs_row.to_numpy()
ingm = o.region.to_numpy()[rows] == 'GM'; nearg = dist[rows] <= 30
fr = pd.Series(nearg[ingm]).groupby(D.s_cc_domain.to_numpy()[ingm]).mean(); base = nearg[ingm].mean()
le = np.log2(fr / base + 1e-9).clip(-2, 2)
enr = np.full(len(o), np.nan); enr[rows[ingm]] = le.reindex(D.s_cc_domain.to_numpy()[ingm]).to_numpy()
print('CODEX s-CC domain log2 enrichment within GM:', le.round(2).to_dict())
rng = np.random.default_rng(0); sub = np.zeros(len(o), bool); sub[rng.choice(np.where(adm)[0], 40000, replace=False)] = True
sec = adm & (don == 'AD-4')
pl4 = pd.read_csv(H / 'bench_codex_cns_2026/external/CODEX-CNS/codex_notebooks/marker_intensities/12032021-Bahareh_3280_Alz_abeta.csv',
                  usecols=['class', 'x', 'y']); pl4 = pl4.loc[pl4['class'] == 'dense', ['x', 'y']].to_numpy() * 0.3774
block('codex', sub, Zc, Sc, ct, dist, 'Distance to Aβ plaque (µm)', 60, scZc, scSc, xy, sec, pl4, 0.4,
      enr, ('s-CC domains (grey matter)', 'Domain plaque log2 enrichment'), DIV, TwoSlopeNorm(0, -2, 2),
      (1000, '1 mm'), list('ijklmnop'))
print('CODEX done', flush=True)
