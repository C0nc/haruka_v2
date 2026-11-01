#!/usr/bin/env python3
"""Figure 3 (plaque panels) — the salient space carries the plaque-proximal response.

  a  CODEX: held-out predicted plaque proximity vs distance from the plaque (z, s, protein)
  b  STARmap: the same (z, s, scVI), anatomy controlled
  c  CODEX: distance rho on real plaques (dots = donors) vs shifted plaques (band)
  d  STARmap: the same (anatomy controlled; dots = seeds; band = shifted, seed 0)
  e  seed robustness: s - z per donor x cell type x seed (CODEX) and cell type x seed (STARmap)
  f  arm swap: Ctl-3's plaque gradient in z and s when Ctl-3 is reference vs condition arm
  g  features on the part of s's plaque axis that z does not carry (microglia)
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import paperstyle as ps
ps.setup()
H = Path('/home/yancui/haruka_v2')
CP = H / 'bench_codex_cns_2026/wholeslice/plaque_states'; SP = H / 'bench_real_discovery/starmap_plaque_radial'
Z, S = ps.LATENT['z'], ps.LATENT['s']
REPC = {'z': Z, 's': S, 'protein': ps.NEUTRAL['mid'], 'expr HVG500': ps.NEUTRAL['mid'], 'scVI32': ps.NEUTRAL['dark']}
REPL = {'z': 'z', 's': 's', 'protein': 'Protein', 'expr HVG500': 'Expression', 'scVI32': 'scVI'}
CMAP = LinearSegmentedColormap.from_list('div', ps.DIVERGING)
src = {}


def proximity(X, dist_log, groups, types, mask):
    """Per cell type ridge axis on log distance; returns -prediction z-scored (high = near)."""
    out = np.full(len(dist_log), np.nan)
    for t in np.unique(types[mask]):
        m = mask & (types == t)
        sc = StandardScaler().fit(X[m]); p = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X[m]), dist_log[m]).predict(sc.transform(X[m]))
        out[m] = -(p - p.mean()) / p.std()
    return out


def centre(X, g):
    X = X.astype(np.float64).copy()
    for k in np.unique(g):
        m = g == k; X[m] -= X[m].mean(0)
    return X


def map_pair(fig_name, xy, scores, mask, plaques, title_prefix, um_per_unit, bar_um, plaque_size):
    fig, axs = plt.subplots(1, 2, figsize=(88 * ps.MM, 46 * ps.MM))
    for ax, key, col in zip(axs, ['z', 's'], [Z, S]):
        o = np.argsort(scores[key][mask])
        sc = ax.scatter(xy[mask][o, 0], xy[mask][o, 1], c=scores[key][mask][o], cmap=CMAP, vmin=-2, vmax=2, s=1.6, lw=0, rasterized=True)
        ax.scatter(plaques[:, 0], plaques[:, 1], s=plaque_size, facecolors='none', edgecolors=ps.INK, lw=0.4)
        ax.set_aspect('equal'); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values(): sp_.set_visible(False)
        ax.set_title(f'{key}', color=col, fontweight='bold', pad=2)
    x0, y1 = xy[mask, 0].min(), xy[mask, 1].max()
    L = bar_um / um_per_unit
    axs[0].plot([x0, x0 + L], [y1 + 0.03 * (y1 - xy[mask, 1].min())] * 2, color=ps.INK, lw=1, clip_on=False)
    axs[0].text(x0 + L / 2, y1 + 0.06 * (y1 - xy[mask, 1].min()), f'{bar_um} µm' if um_per_unit != 1 else f'{bar_um} units',
                ha='center', va='top', fontsize=6)
    cb = fig.colorbar(sc, ax=axs, fraction=0.03, pad=0.02); cb.set_ticks([-2, 0, 2]); cb.outline.set_linewidth(.4)
    cb.set_label('Plaque-proximity score', labelpad=1)
    fig.text(0.02, 0.97, title_prefix, fontsize=6.5, va='top')
    return fig, axs


# ---------------- a, b: radial profiles from the plaque outward (held-out predictions) ----------------
def heldout_score(X, dl, types, groups_test, mask):
    """Per cell type ridge on log distance, predicted for each held-out donor/slice; z-scored
    within type and sign-flipped so high = predicted near a plaque."""
    out = np.full(len(dl), np.nan)
    for t in np.unique(types[mask]):
        for gt in np.unique(groups_test[mask]):
            tr = mask & (types == t) & (groups_test != gt); te = mask & (types == t) & (groups_test == gt)
            sc = StandardScaler().fit(X[tr]); r = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X[tr]), dl[tr])
            out[te] = -r.predict(sc.transform(X[te]))
        m = mask & (types == t); out[m] = (out[m] - out[m].mean()) / out[m].std()
    return out


def profile_panel(ax, scores, dist, mask, edges, labels, xlabel, title):
    b = np.digitize(dist, edges[1:-1])
    xs = np.arange(len(labels))
    for k, v in scores.items():
        mu = np.array([np.nanmean(v[mask & (b == i)]) for i in xs])
        se = np.array([np.nanstd(v[mask & (b == i)]) / np.sqrt((mask & (b == i)).sum()) for i in xs])
        ax.fill_between(xs, mu - 1.96 * se, mu + 1.96 * se, color=REPC[k], alpha=0.25, lw=0)
        ax.plot(xs, mu, color=REPC[k], lw=1.2, marker='o', ms=2.5, label=REPL[k])
    ax.axhline(0, color=ps.NEUTRAL['mid'], lw=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(labels); ax.set_xlabel(xlabel)
    ax.set_ylabel('Predicted plaque proximity\n(held-out, SD units)'); ax.set_title(title, pad=2)
    ax.legend(loc='upper right', handlelength=1.2)

# CODEX: grey matter, AD donors, donor-held-out
a = ad.read_h5ad(H / 'bench_codex_cns_2026/wholeslice/codex_wholeslice_all7_no_oligo.h5ad'); o = a.obs
MD = H / 'results_codex_wholeslice/codex_wholeslice_all7_no_oligo/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
don = o.donor_id.astype(str).to_numpy(); ct = o.cell_type.astype(str).to_numpy()
dist = np.load(CP / 'cell_dist_any_abeta_um.npy'); dl = np.log1p(np.minimum(dist, 150))
use = (o.condition.astype(str).to_numpy() == 'disease') & (o.region.to_numpy() == 'GM')
g = (o.donor_id.astype(str) + '|' + o.cell_type.astype(str)).to_numpy()
XA = {'z': np.load(MD / 'background.npy'), 's': np.load(MD / 'salient.npy'), 'protein': np.asarray(a.X)}
scA = {k: heldout_score(centre(v, g), dl, ct, don, use) for k, v in XA.items()}
EA = [0, 10, 20, 40, 60, 100, np.inf]; LA = ['0–10', '10–20', '20–40', '40–60', '60–100', '>100']
fig, ax = plt.subplots(figsize=(60 * ps.MM, 44 * ps.MM))
profile_panel(ax, scA, dist, use, EA, LA, 'Distance to nearest Aβ plaque (µm)', 'CODEX, 4 AD donors (grey matter)')
ps.panel_label(ax, 'a', x=-0.3, y=1.04); ps.save(fig, HERE / 'fig3a_codex_radial_profile'); plt.close(fig)
src['a'] = pd.DataFrame({'donor': don[use], 'cell_type': ct[use], 'dist_um': dist[use], **{f'{k}_score': v[use] for k, v in scA.items()}})

# STARmap: AD slices, anatomy controlled (slice x fine type x domain), cross-slice held-out
b = ad.read_h5ad(H / 'bench_real_discovery/data/ad.h5ad'); ob = b.obs
MDs = H / 'results_real_compgated_lambda10_samplebatch_sepheads/ad/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
CROSS = {'CTX-Ex': 'Neuron', 'CA1': 'Neuron', 'CA2': 'Neuron', 'CA3': 'Neuron', 'DG': 'Neuron', 'Inh': 'Neuron', 'LHb': 'Neuron',
         'Astro': 'Astrocyte', 'Oligo': 'Oligodendrocyte', 'OPC': 'Oligodendrocyte', 'Micro': 'Microglia', 'Endo': 'Vascular', 'SMC': 'Vascular'}
maj = pd.Series(ob.ct_top.astype(str).to_numpy()).map(CROSS).to_numpy()
Dt = pd.read_csv(H / 'final_results_nb/starmap_ad/biology/plaque_distance/starmap_plaque_distance_cells.csv.gz').set_index('cell_id')
dsb = pd.Series(b.obs_names).map(Dt.distance_to_abeta_associated_cell_spatial_units).to_numpy(float)
SPC = 31.4; dlb = np.log1p(np.minimum(dsb, 16 * SPC)); slb = ob.slice_id.astype(str).to_numpy()
gb = (ob.slice_id.astype(str) + '|' + ob.ct_top.astype(str) + '|' + ob.tissue_domain.astype(str)).to_numpy()
useb = np.isin(slb, ['disease_r1', 'disease_r2']) & np.isfinite(dlb)
XB = {'z': np.load(MDs / 'background.npy'), 's': np.load(MDs / 'salient.npy'), 'scVI32': np.load(SP / 'scvi32.npy')}
scB = {k: heldout_score(centre(v, gb), dlb, maj, slb, useb) for k, v in XB.items()}
EB = np.array([-0.5, 1e-9, 1, 2, 4, 8, np.inf]) * SPC; LB = ['Aβ cell', '≤1', '1–2', '2–4', '4–8', '>8']
fig, ax = plt.subplots(figsize=(60 * ps.MM, 44 * ps.MM))
profile_panel(ax, scB, dsb, useb, EB, LB, 'Distance to nearest Aβ cell (cell spacings)', 'STARmap, 2 AD slices')
ps.panel_label(ax, 'b', x=-0.3, y=1.04); ps.save(fig, HERE / 'fig3b_starmap_radial_profile'); plt.close(fig)
src['b'] = pd.DataFrame({'slice': slb[useb], 'major_type': maj[useb], 'dist_units': dsb[useb], **{f'{k}_score': v[useb] for k, v in scB.items()}})


# ---------------- c, d: real vs shifted ----------------
def real_vs_null(ax, real, null, types, reps, xlabels):
    w = 0.8 / len(reps)
    for i, t in enumerate(types):
        for j, rp in enumerate(reps):
            x = i + (j - (len(reps) - 1) / 2) * w
            nm = null[(null.cell_type == t) & (null.rep == rp)].rho
            ax.add_patch(plt.Rectangle((x - w * 0.42, nm.min()), w * 0.84, max(nm.max() - nm.min(), 0.006), color=ps.NEUTRAL['light'], lw=0, zorder=0))
            v = real[(real.cell_type == t) & (real.rep == rp)].rho.to_numpy()
            ax.scatter(np.full(len(v), x) + np.linspace(-w * 0.2, w * 0.2, len(v)), v, s=5, color=REPC[rp], lw=0, zorder=3)
            ax.plot([x - w * 0.35, x + w * 0.35], [v.mean()] * 2, color=REPC[rp], lw=1.2, zorder=4)
    ax.set_xticks(range(len(types))); ax.set_xticklabels(xlabels); ax.axhline(0, color=ps.NEUTRAL['mid'], lw=0.5)
    ax.set_ylabel('Distance-to-plaque ρ\n(held-out)')
    for rp in reps: ax.plot([], [], color=REPC[rp], lw=1.2, marker='o', ms=2.5, label=REPL[rp])
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, color=ps.NEUTRAL['light'], label='Shifted plaques'))
    ax.legend(loc='upper left', ncol=len(reps) + 1, handlelength=1.2, columnspacing=0.8, bbox_to_anchor=(0, 1.16))

N = pd.read_csv(CP / 'radial_null_all7_no_oligo.csv')
fig, ax = plt.subplots(figsize=(60 * ps.MM, 42 * ps.MM))
TY = ['Neuron', 'Astrocyte', 'Microglia']
real_vs_null(ax, N[N.kind == 'real'], N[N.kind == 'shifted'].groupby(['cell_type', 'rep', 'it']).rho.mean().reset_index(), TY, ['z', 's', 'protein'], TY)
ax.set_title('CODEX (4 AD donors)', pad=12); ps.panel_label(ax, 'c', x=-0.3, y=1.12)
ps.save(fig, HERE / 'fig3c_codex_real_vs_shifted'); plt.close(fig); src['c'] = N

NS = pd.read_csv(SP / 'starmap_null_scvi.csv')
seeds = pd.concat([pd.read_csv(SP / f).assign(seed=s) for s, f in [(0, 'starmap_null_scvi.csv'), (1, 'starmap_null_scvi_seed1.csv'), (2, 'starmap_null_scvi_seed2.csv')]])
TYS = ['Neuron', 'Astrocyte', 'Oligodendrocyte', 'Microglia', 'Vascular']
fig, ax = plt.subplots(figsize=(88 * ps.MM, 42 * ps.MM))
real_vs_null(ax, seeds[seeds.kind == 'real'], NS[NS.kind == 'shifted'], TYS, ['z', 's', 'scVI32', 'expr HVG500'], ['Neuron', 'Astro.', 'Oligo.', 'Microglia', 'Vascular'])
ax.set_title('STARmap (2 AD slices, anatomy controlled)', pad=12); ps.panel_label(ax, 'd', x=-0.2, y=1.12)
ps.save(fig, HERE / 'fig3d_starmap_real_vs_shifted'); plt.close(fig); src['d'] = seeds

# ---------------- e: seed robustness s - z ----------------
C = pd.concat([pd.read_csv(CP / f).assign(seed=s) for s, f in [(0, 'radial_generic_all7_no_oligo.csv'), (1, 'radial_generic_all7_no_oligo_seed1.csv'), (2, 'radial_generic_all7_no_oligo_seed2.csv')]])
Cd = C.pivot_table(index=['cell_type', 'donor', 'seed'], columns='rep', values='rho_half').reset_index(); Cd['d'] = Cd.s - Cd.z
Sd = seeds[seeds.kind == 'real'].pivot_table(index=['cell_type', 'seed'], columns='rep', values='rho').reset_index(); Sd['d'] = Sd.s - Sd.z
fig, ax = plt.subplots(figsize=(52 * ps.MM, 42 * ps.MM))
rng = np.random.default_rng(0)
for i, (lab, D) in enumerate([('CODEX', Cd), ('STARmap', Sd)]):
    v = D.d.to_numpy(); x = i + rng.uniform(-0.18, 0.18, len(v))
    ax.scatter(x, v, s=6, lw=0, color=np.where(v > 0, ps.DELTA['up'], ps.DELTA['down']))
    ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=ps.INK, lw=1)
    ax.text(i, -0.17, f's > z: {(v > 0).sum()}/{len(v)}', ha='center', va='top', fontsize=6)
ax.axhline(0, color=ps.NEUTRAL['mid'], lw=0.5); ax.set_xticks([0, 1]); ax.set_xticklabels(['CODEX', 'STARmap'])
ax.set_xlim(-0.6, 1.6); ax.set_ylim(-0.2, 0.36); ax.set_ylabel('ρ(s) − ρ(z)'); ax.set_title('3 training seeds', pad=2)
ps.panel_label(ax, 'e', x=-0.38, y=1.04); ps.save(fig, HERE / 'fig3e_seed_robustness'); plt.close(fig)
src['e'] = pd.concat([Cd.assign(dataset='CODEX'), Sd.assign(dataset='STARmap')])

# ---------------- f: arm swap ----------------
F0 = pd.read_csv(CP / 'radial_generic_all7_no_oligo_allarms.csv'); F0 = F0[F0.donor == 'Ctl-3'].assign(arm='Reference\n(AD fit)')
F2 = pd.read_csv(CP / 'radial_generic_ctl3VsCtl14.csv').assign(arm='Condition\n(Ctl-3 vs Ctl-1,4)')
FF = pd.concat([F0, F2]); FF = FF[FF.rep.isin(['z', 's'])]
fig, ax = plt.subplots(figsize=(48 * ps.MM, 42 * ps.MM))
for i, arm in enumerate(FF.arm.unique()):
    for j, rp in enumerate(['z', 's']):
        v = FF[(FF.arm == arm) & (FF.rep == rp)].rho_half.to_numpy()
        x = i + (j - 0.5) * 0.36
        ax.bar(x, v.mean(), width=0.32, color=REPC[rp], lw=0)
        ax.scatter(np.full(len(v), x), v, s=4, color=ps.INK, lw=0, zorder=3)
ax.set_xticks([0, 1]); ax.set_xticklabels(FF.arm.unique()); ax.set_ylabel('Ctl-3 plaque ρ (half-slice)')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=Z, label='z'), Patch(color=S, label='s')], loc='upper left', ncol=2, bbox_to_anchor=(0, 1.15), handlelength=1)
ax.set_title('Amyloid-positive control', pad=12); ps.panel_label(ax, 'f', x=-0.38, y=1.12)
ps.save(fig, HERE / 'fig3f_arm_swap'); plt.close(fig); src['f'] = FF

# ---------------- g: s beyond z (microglia) ----------------
fig, axs = plt.subplots(1, 2, figsize=(80 * ps.MM, 42 * ps.MM))
for ax, (lab, f) in zip(axs, [('CODEX microglia', 's_beyond_z_codex.csv'), ('STARmap microglia', 's_beyond_z_starmap.csv')]):
    B = pd.read_csv(SP / f); B = B[B.cell_type == 'Microglia'].sort_values('rho_s_beyond_z').head(6)[::-1]
    y = np.arange(len(B))
    ax.barh(y + 0.18, -B.rho_s_beyond_z, height=0.34, color=S, lw=0, label='s beyond z')
    ax.barh(y - 0.18, -B.rho_z_axis, height=0.34, color=Z, lw=0, label='z axis')
    ax.set_yticks(y); ax.set_yticklabels(B.feature); ax.axvline(0, color=ps.NEUTRAL['mid'], lw=0.5)
    ax.set_xlabel('Correlation with\nplaque proximity'); ax.set_title(lab, pad=2)
plt.tight_layout(w_pad=1.5); axs[1].legend(*axs[0].get_legend_handles_labels(), loc='lower right', handlelength=1)
ps.panel_label(axs[0], 'g', x=-0.45, y=1.04); ps.save(fig, HERE / 'fig3g_s_beyond_z'); plt.close(fig)

for k, v in src.items(): v.to_csv(HERE / f'fig3{k}_plaque_source_data.csv.gz', index=False, compression='gzip')
bad = []
for f in sorted(HERE.glob('fig3[a-g]_*.svg')):
    if f.name.startswith('fig3a_myeloid'): continue
    ok, sz = ps.check_fonts(f); print(f.name, 'min font', min(sz) if sz else None, 'OK' if ok else 'BELOW MINIMUM')
print('profile slope (near bin minus far bin):', {k: round(np.nanmean(v[use & (dist <= 10)]) - np.nanmean(v[use & (dist > 100)]), 3) for k, v in scA.items()}, {k: round(np.nanmean(v[useb & (dsb == 0)]) - np.nanmean(v[useb & (dsb > 8 * SPC)]), 3) for k, v in scB.items()})
