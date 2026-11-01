#!/usr/bin/env python3
"""Lung fibrosis (GSE250346 Xenium) figure: s carries lesion context and the gradient towards lesions; z does not.
Haruka fit: Unaffected vs all fibrosis, batch = core, one-way z <- s penalty 10, seed 0.  Every z / s readout uses
fibrosis-core (condition) cells only.  Ground truth = pathologist H&E region annotations (GSE250346_HE_annotations).
  a tissue: H&E regions (example core)        b tissue: cell lineage
  c tissue: alveolar-FB s axis vs focus       d tissue: alveolar-FB z axis
  e H&E region pairs, same core + cell type: AUROC z vs s       f decoded niche vs observed niche difference
  g distance-to-lesion, supervised (ridge, leave-one-donor-out)  h distance-to-lesion, unsupervised (best of PC1/PC2/DPT)
  i distance profiles of the s / z axis (3 robust cases)         j robustness: |r| vs spatial-shift null
  k alveolar FB -> focus: genes along the s axis vs genes along real proximity
  l alveolar FB -> focus: decoded salient neighbour shift along the s axis
  m-p global UMAPs of z and s (fibrosis cells, <= 5000 per H&E region): cell lineage and H&E region
  (a, b carry a dashed box marking the window shown in c, d)
Sources: bench_lung_fibrosis_gse250346/zs_stage/{he_region,lesion_gradient}/ and figures_paper/fibrosis/data/ (fib_fig_data.py).
"""
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
import sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
warnings.filterwarnings('ignore')
sys.path.insert(0, f'{_HR}/figures_paper'); import paperstyle as ps
ps.setup(); MM = ps.MM
OUT = Path(f'{_HR}/figures_paper/fibrosis'); DAT = OUT / 'data'
Z = Path(f'{_HR}/bench_lung_fibrosis_gse250346/zs_stage')
CZ, CS = ps.LATENT['z'], ps.LATENT['s']
# lesion colours drawn from the shared palette roles (distinct hues; fibrosis background kept neutral)
LES = {'fibroblastic_focus': ps.DOMAIN[1], 'mixed_inflammation': ps.DOMAIN[2], 'TLS': ps.REGION['VEN'],
       'remodeled_epithelium': ps.REGION['STR_CP/ACB'], 'microscopic_honeycombing': ps.REGION['CTX_L1/MEN'],
       'severe_fibrosis': ps.NEUTRAL['dark'], 'hyperplastic_aec': ps.REGION['CTX_L4/5/6'],
       'minimally_remodeled_alveoli': ps.REGION['STR_LS/NDB'], 'advanced_remodeling': ps.DOMAIN[0]}
NICE = {'fibroblastic_focus': 'Fibroblastic focus', 'mixed_inflammation': 'Mixed inflammation', 'TLS': 'TLS',
        'remodeled_epithelium': 'Remodelled epithelium', 'microscopic_honeycombing': 'Honeycombing', 'severe_fibrosis': 'Severe fibrosis',
        'hyperplastic_aec': 'Hyperplastic AEC', 'minimally_remodeled_alveoli': 'Min. remodelled alveoli', 'advanced_remodeling': 'Advanced remodelling',
        'fibrosis': 'Fibrosis', 'remnant_alveoli': 'Remnant alveoli', 'none': 'Not annotated'}
LIN = {'Epithelial': ps.MAJOR['Astrocyte'], 'Endothelial': ps.MAJOR['Vascular'], 'Mesenchymal': ps.MAJOR['Oligodendrocyte'], 'Immune': ps.MAJOR['Neuron']}
SEQ = LinearSegmentedColormap.from_list('seq', ps.SEQUENTIAL)
made = []
def save(fig, name):
    ps.save(fig, OUT / name); plt.close(fig); made.append(name)

# ---------------- a-d tissue maps
E = np.load(DAT / 'example_core.npz', allow_pickle=True); xy = E['xy']; ex = str(E['core'])
# zoom on the largest focus (connected component of focus cells at 40 um) so the gradient is visible
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree as _T
fx = xy[E['focus']]; G_ = _T(fx).sparse_distance_matrix(_T(fx), 40); ncomp, lab_ = connected_components(G_ != 0, directed=False)
big = np.bincount(lab_).argmax(); cen = fx[lab_ == big].mean(0); R = 480
win = (np.abs(xy - cen) <= R).all(1); wa = (np.abs(E['afb_xy'] - cen) <= R).all(1)
def tissue_ax(L, title, w=52):
    fig, ax = plt.subplots(figsize=(w * MM, w * MM)); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, pad=2); ps.panel_label(ax, L, x=-0.04, y=1.03); return fig, ax
def sbar(ax, um=500):
    x0, y0 = xy[:, 0].min(), xy[:, 1].min() - 60
    ax.plot([x0, x0 + um], [y0, y0], color=ps.INK, lw=1); ax.text(x0 + um / 2, y0 - 40, f'{um} µm', ha='center', va='top', fontsize=5.8)
def zoombox(ax):
    ax.add_patch(plt.Rectangle(cen - R, 2 * R, 2 * R, fill=False, ec=ps.INK, lw=0.7, ls='--', zorder=5))
    ax.text(cen[0] + R, cen[1] + R + 30, 'c, d', ha='right', va='bottom', fontsize=6, fontweight='bold')
fig, ax = tissue_ax('a', f'H&E regions ({ex})')
reg = E['region']; base = reg == 'none'
ax.scatter(*xy[base].T, s=0.6, c=ps.NEUTRAL['light'], lw=0, rasterized=True)
ax.scatter(*xy[reg == 'fibrosis'].T, s=0.6, c=ps.CONDITION['control'], lw=0, rasterized=True)
shown = [r for r in LES if (reg == r).any()]
for r in shown: ax.scatter(*xy[reg == r].T, s=0.9, c=LES[r], lw=0, rasterized=True)
hd = [plt.Line2D([], [], marker='o', ls='', ms=3, color=LES[r], label=NICE[r]) for r in shown] + \
     [plt.Line2D([], [], marker='o', ls='', ms=3, color=ps.CONDITION['control'], label='Fibrosis')]
ax.legend(handles=hd, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=5.8, handletextpad=0.2, borderaxespad=0); sbar(ax); zoombox(ax)
save(fig, 'fib_a_tissue_he')
fig, ax = tissue_ax('b', 'Cell lineage')
lin = E['lineage']
for l, c in LIN.items(): ax.scatter(*xy[lin == l].T, s=0.6, c=c, lw=0, rasterized=True, label=l)
ax.scatter(*xy[E['focus']].T, s=1.2, facecolors='none', edgecolors=ps.INK, lw=0.25, rasterized=True)
hd = [plt.Line2D([], [], marker='o', ls='', ms=3, color=c, label=l) for l, c in LIN.items()] + [plt.Line2D([], [], marker='o', ls='', ms=3, mfc='none', mec=ps.INK, label='Focus cell')]
ax.legend(handles=hd, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=5.8, handletextpad=0.2, borderaxespad=0); sbar(ax); zoombox(ax)
save(fig, 'fib_b_tissue_lineage')
for L, k, col in [('c', 's', CS), ('d', 'z', CZ)]:
    fig, ax = plt.subplots(figsize=(48 * MM, 48 * MM)); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f'{k} PC1, alveolar FBs', pad=10, color=col, fontweight='bold'); ps.panel_label(ax, L, x=-0.08, y=1.06)
    ax.scatter(*xy[win].T, s=2.0, c=ps.NEUTRAL['light'], lw=0, rasterized=True)
    v = E[f'afb_{k}']; ok = np.isfinite(v) & wa; q = np.nanpercentile(v[np.isfinite(v)], [5, 95])
    im = ax.scatter(*E['afb_xy'][ok].T, s=9, c=np.clip(v[ok], *q), cmap=SEQ, lw=0, rasterized=True)
    ax.scatter(*xy[win & E['focus']].T, s=7, c=LES['fibroblastic_focus'], edgecolors=ps.INK, lw=0.2, zorder=4, rasterized=True)
    okc = np.isfinite(v); r = np.corrcoef(v[okc], E['afb_dist'][okc])[0, 1]
    ax.text(0.0, 1.0, f'core r(axis, distance) = {r:+.2f}', transform=ax.transAxes, fontsize=5.8, va='top')
    x0, y0 = cen[0] - R, cen[1] - R - 25; ax.plot([x0, x0 + 200], [y0, y0], color=ps.INK, lw=1); ax.text(x0 + 100, y0 - 20, '200 µm', ha='center', va='top', fontsize=5.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.01); cb.ax.tick_params(labelsize=5.8); cb.set_label('towards focus →', fontsize=5.8); cb.outline.set_linewidth(0.4)
    ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', ms=3, color=LES['fibroblastic_focus'], mec=ps.INK, mew=0.2, label='focus cell')], loc='lower right', fontsize=5.8, handletextpad=0.1, borderaxespad=0)
    save(fig, f'fib_{L}_tissue_afb_{k}')

# ---------------- e region pairs AUROC
A = pd.read_csv(Z / 'he_region/region_pairs.csv'); A['first'] = A.pair.str.split(' vs ').str[0]
order = ['fibroblastic_focus', 'remodeled_epithelium', 'mixed_inflammation', 'hyperplastic_aec', 'microscopic_honeycombing', 'severe_fibrosis', 'minimally_remodeled_alveoli']
fig, ax = plt.subplots(figsize=(84 * MM, 42 * MM)); rng = np.random.default_rng(0)
for x, r in enumerate(order):
    t = A[A['first'] == r]
    for _, q in t.iterrows(): ax.plot([x - 0.17, x + 0.17], [q.auc_z, q.auc_s], color=ps.NEUTRAL['mid'], lw=0.35, alpha=0.6, zorder=1)
    ax.scatter(np.full(len(t), x - 0.17), t.auc_z, s=5, c=CZ, lw=0, zorder=2); ax.scatter(np.full(len(t), x + 0.17), t.auc_s, s=5, c=CS, lw=0, zorder=2)
    ax.text(x, 0.99, f'{(t.auc_s > t.auc_z).sum()}/{len(t)}', ha='center', fontsize=5.8)
ax.axhline(0.5, color=ps.NEUTRAL['mid'], lw=0.5, ls=':')
ax.set_xticks(range(len(order))); ax.set_xticklabels([NICE[r] for r in order], rotation=30, ha='right'); ax.set_ylim(0.35, 1.02)
ax.set_ylabel('AUROC vs fibrosis\n(same core, same cell type)')
ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', ms=3, color=CZ, label='z'), plt.Line2D([], [], marker='o', ls='', ms=3, color=CS, label='s')], loc='lower left', ncol=2, fontsize=5.8)
ax.set_title(f'H&E lesion identity: s > z in {(A.auc_s > A.auc_z).sum()}/{len(A)} (median {A.auc_z.median():.2f} → {A.auc_s.median():.2f})', pad=6)
ps.panel_label(ax, 'e', x=-0.13); save(fig, 'fib_e_region_pairs')

# ---------------- f decoded niche alignment
S = pd.read_csv(Z / 'he_region/region_pairs_spatial.csv'); S['first'] = S.pair.str.split(' vs ').str[0]
fig, ax = plt.subplots(figsize=(46 * MM, 42 * MM))
for r in order:
    t = S[S['first'] == r]; ax.scatter(t.align_bg, t.align_cond, s=6, c=LES.get(r, ps.NEUTRAL['dark']), lw=0, label=NICE[r])
ax.plot([0, 1], [0, 1], color=ps.NEUTRAL['mid'], lw=0.5, ls=':'); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
ax.set_xlabel('decoder with z only, D([z,0])'); ax.set_ylabel('decoder with s, D([z,s])')
ax.legend(loc='lower right', fontsize=5.6, handletextpad=0.1, borderaxespad=0, markerscale=0.8, labelspacing=0.2)
ax.set_title(f'niche difference reproduced\n(r with observed; above line {(S.align_cond > S.align_bg).sum()}/{len(S)})', pad=2)
ps.panel_label(ax, 'f', x=-0.3); save(fig, 'fib_f_niche_alignment')

# ---------------- g supervised distance
G = pd.read_csv(Z / 'lesion_gradient/A_supervised.csv'); lo = ['mixed_inflammation', 'TLS', 'fibroblastic_focus', 'remodeled_epithelium', 'microscopic_honeycombing', 'severe_fibrosis']
fig, ax = plt.subplots(figsize=(66 * MM, 42 * MM))
for x, r in enumerate(lo):
    t = G[G.lesion == r]
    for _, q in t.iterrows(): ax.plot([x - 0.17, x + 0.17], [q.z, q.s], color=ps.NEUTRAL['mid'], lw=0.35, alpha=0.6)
    ax.scatter(np.full(len(t), x - 0.17), t.z, s=5, c=CZ, lw=0, zorder=2); ax.scatter(np.full(len(t), x + 0.17), t.s, s=5, c=CS, lw=0, zorder=2)
    ax.text(x, 0.33, f'{(t.s > t.z).sum()}/{len(t)}', ha='center', fontsize=5.8)
ax.axhline(0, color=ps.NEUTRAL['mid'], lw=0.5, ls=':'); ax.set_ylim(-0.17, 0.36)
ax.set_xticks(range(len(lo))); ax.set_xticklabels([NICE[r] for r in lo], rotation=30, ha='right')
ax.set_ylabel('within-core ρ (predicted vs\nreal distance, donor held out)')
ax.set_title(f'distance to lesion, supervised: s > z {(G.s > G.z).sum()}/{len(G)}', pad=6); ps.panel_label(ax, 'g', x=-0.2); save(fig, 'fib_g_distance_supervised')

# ---------------- h unsupervised best axis
U = pd.read_csv(Z / 'lesion_gradient/unsup_axes_vs_distance.csv')
U['bz'] = U[['PC1_z', 'PC2_z', 'DPT_z']].max(1); U['bs'] = U[['PC1_s', 'PC2_s', 'DPT_s']].max(1)
fig, ax = plt.subplots(figsize=(46 * MM, 42 * MM))
for r in lo:
    t = U[U.lesion == r]; ax.scatter(t.bz, t.bs, s=6, c=LES[r], lw=0, label=NICE[r])
m = max(U.bz.max(), U.bs.max()) * 1.05; ax.plot([0, m], [0, m], color=ps.NEUTRAL['mid'], lw=0.5, ls=':'); ax.set_xlim(0, m); ax.set_ylim(0, m)
ax.set_xlabel('|r| best z axis'); ax.set_ylabel('|r| best s axis')
ax.set_title(f'unsupervised PC1/PC2/DPT vs distance\n(s > z {(U.bs > U.bz).sum()}/{len(U)})', pad=2)
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=5.8, handletextpad=0.1, borderaxespad=0)
ps.panel_label(ax, 'h', x=-0.3); save(fig, 'fib_h_distance_unsupervised')

# ---------------- i profiles
bins = ['0-25', '25-50', '50-100', '100-200', '200-400']
CASE3 = [('Alveolar_fibroblastic_focus', 'Alveolar FBs → focus'), ('Capillary_fibroblastic_focus', 'Capillary → focus (PC2)'), ('CD4_TLS', 'CD4 T → TLS')]
fig, axs = plt.subplots(1, 3, figsize=(110 * MM, 36 * MM), sharey=False)
for ax, (tag, title) in zip(axs, CASE3):
    Pf = pd.read_csv(DAT / 'cases' / f'{tag}_profile.csv', index_col=[0, 1])
    for k, col in [('z', CZ), ('s', CS)]:
        t = Pf.loc[k]; mu = t.mean(); se = t.std() / np.sqrt(t.notna().sum())
        ax.errorbar(range(5), mu.values, yerr=se.values, color=col, lw=0.9, marker='o', ms=2.2, capsize=1.2, elinewidth=0.5, label=k)
    ax.axhline(0, color=ps.NEUTRAL['mid'], lw=0.4, ls=':'); ax.set_xticks(range(5)); ax.set_xticklabels(bins, rotation=40, ha='right')
    ax.set_title(title, pad=2); ax.set_xlabel('distance to lesion (µm)')
axs[0].set_ylabel('axis value (towards lesion = +)\nmean ± s.e. over cores'); axs[0].legend(fontsize=5.8, loc='upper right')
ps.panel_label(axs[0], 'i', x=-0.45); fig.tight_layout(); save(fig, 'fib_i_profiles')

# ---------------- j robustness
C = pd.read_csv(Z / 'lesion_gradient/cases/cases_summary.csv')
lab = {('TLS', 'CD4+ T-cells'): 'CD4 T → TLS', ('fibroblastic_focus', 'Alveolar FBs'): 'Alveolar FB → focus', ('microscopic_honeycombing', 'Alveolar FBs'): 'Alveolar FB → honeycomb',
       ('microscopic_honeycombing', 'Interstitial Macrophages'): 'Int. macrophage → honeycomb', ('fibroblastic_focus', 'Capillary'): 'Capillary → focus'}
fig, ax = plt.subplots(figsize=(62 * MM, 38 * MM))
for y, (_, r) in enumerate(C.iterrows()):
    for k, col, dy in [('z', CZ, -0.13), ('s', CS, 0.13)]:
        ax.plot([r[f'ci_lo_{k}'], r[f'ci_hi_{k}']], [y + dy] * 2, color=col, lw=0.8)
        ax.scatter(r[f'r_{k}'], y + dy, s=9, c=col, lw=0, zorder=3)
        ax.scatter(r[f'null_median_{k}'], y + dy, s=9, marker='|', c=ps.INK, lw=0.7, zorder=3)
    ax.text(0.30, y, f"p {r['null_p_s']:.3f}", va='center', fontsize=5.8, color=CS)
ax.set_yticks(range(len(C))); ax.set_yticklabels([lab[(r.lesion, r.cell_type)] for _, r in C.iterrows()]); ax.invert_yaxis()
ax.set_xlim(0, 0.36); ax.set_xlabel('|r| axis vs distance (95% CI over cores)')
ax.set_title('robustness (| = spatial-shift null median;\np = s vs null)', pad=2); ps.panel_label(ax, 'j', x=-0.75); save(fig, 'fib_j_robustness')

# ---------------- k genes along the s axis vs proximity
g = pd.read_csv(Z / 'lesion_gradient/cases/genes_Alveolar_FBs_fibroblastic_focus.csv')
fig, ax = plt.subplots(figsize=(52 * MM, 44 * MM))
ax.scatter(g.rho_proximity, g.rho_s_axis, s=3, c=ps.NEUTRAL['mid'], lw=0, rasterized=True)
hi = pd.concat([g.nlargest(6, 'rho_s_axis'), g.nsmallest(5, 'rho_s_axis')])
ax.scatter(hi.rho_proximity, hi.rho_s_axis, s=6, c=CS, lw=0)
ys = []
for _, q in hi.sort_values('rho_s_axis').iterrows():   # simple vertical de-overlap of labels
    y = q.rho_s_axis
    while any(abs(y - y0) < 0.045 and abs(q.rho_proximity - x0) < 0.035 for x0, y0 in ys): y += 0.045
    ys.append((q.rho_proximity, y)); ax.text(q.rho_proximity + 0.003, y, q.gene, fontsize=5.6, va='center')
ax.axhline(0, color=ps.NEUTRAL['mid'], lw=0.4, ls=':'); ax.axvline(0, color=ps.NEUTRAL['mid'], lw=0.4, ls=':')
ax.set_xlabel('ρ gene vs proximity to focus'); ax.set_ylabel('ρ gene vs s axis')
ax.set_title(f'Alveolar FBs → focus: genes\n(r over genes = {np.corrcoef(g.rho_s_axis, g.rho_proximity)[0, 1]:.2f})', pad=2)
ps.panel_label(ax, 'k', x=-0.3); save(fig, 'fib_k_genes')

# ---------------- l decoded shift along the s axis
sh = pd.read_csv(DAT / 'cases' / 'Alveolar_fibroblastic_focus_shift.csv').sort_values('rho')
t = pd.concat([sh.head(5), sh.tail(5)])
fig, ax = plt.subplots(figsize=(50 * MM, 44 * MM))
ax.barh(range(len(t)), t.rho, color=[ps.DELTA['up'] if v > 0 else ps.DELTA['down'] for v in t.rho], height=0.7)
ax.set_yticks(range(len(t))); ax.set_yticklabels(t.neighbour); ax.axvline(0, color=ps.INK, lw=0.5)
ax.set_xlabel('ρ decoded neighbour shift vs s axis')
ax.set_title('Alveolar FBs → focus:\nneighbours predicted by s', pad=2); ps.panel_label(ax, 'l', x=-0.95); save(fig, 'fib_l_shift')


# ---------------- m-p global UMAPs (fibrosis cells with one H&E label, <= 5000 per region; fib_umap_data.py)
Um = np.load(DAT / 'umap.npz', allow_pickle=True); ureg = Um['region']; ulin = Um['lineage']
UL = ['fibroblastic_focus', 'mixed_inflammation', 'TLS', 'remodeled_epithelium', 'microscopic_honeycombing', 'minimally_remodeled_alveoli']
for L, k, by in [('m', 'z', 'lineage'), ('n', 's', 'lineage'), ('o', 'z', 'region'), ('p', 's', 'region')]:
    E2 = Um[f'umap_{k}']; col = CZ if k == 'z' else CS
    fig, ax = plt.subplots(figsize=(44 * MM, 44 * MM)); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f'{k} UMAP: ' + ('cell lineage' if by == 'lineage' else 'H&E region'), pad=2, color=col, fontweight='bold'); ps.panel_label(ax, L, x=-0.04, y=1.03)
    if by == 'lineage':
        for l, c in LIN.items(): ax.scatter(*E2[ulin == l].T, s=0.25, c=c, lw=0, rasterized=True)
        hd = [plt.Line2D([], [], marker='o', ls='', ms=3, color=c, label=l) for l, c in LIN.items()]
    else:
        oth = ~np.isin(ureg, UL + ['fibrosis'])
        ax.scatter(*E2[oth].T, s=0.25, c=ps.NEUTRAL['light'], lw=0, rasterized=True)
        ax.scatter(*E2[ureg == 'fibrosis'].T, s=0.25, c=ps.NEUTRAL['mid'], lw=0, rasterized=True)
        for r in UL: ax.scatter(*E2[ureg == r].T, s=0.6 if r == 'fibroblastic_focus' else 0.3, c=LES[r], lw=0, rasterized=True, zorder=3 if r == 'fibroblastic_focus' else 2)
        hd = [plt.Line2D([], [], marker='o', ls='', ms=3, color=LES[r], label=NICE[r]) for r in UL] + \
             [plt.Line2D([], [], marker='o', ls='', ms=3, color=ps.NEUTRAL['mid'], label='Fibrosis'), plt.Line2D([], [], marker='o', ls='', ms=3, color=ps.NEUTRAL['light'], label='Other H&E region')]
    if k == 's': ax.legend(handles=hd, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=5.8, handletextpad=0.2, borderaxespad=0)
    save(fig, f'fib_{L}_umap_{k}_{by}')

bad = [n for n in made if not ps.check_fonts(OUT / f'{n}.svg')[0]]
print('panels', len(made), '| fonts below minimum:', bad)

# ---------------- full page preview
rows = [['fib_a_tissue_he', 'fib_b_tissue_lineage', 'fib_c_tissue_afb_s', 'fib_d_tissue_afb_z'],
        ['fib_e_region_pairs', 'fib_f_niche_alignment', 'fib_g_distance_supervised'],
        ['fib_h_distance_unsupervised', 'fib_i_profiles', 'fib_j_robustness'],
        ['fib_k_genes', 'fib_l_shift'],
        ['fib_m_umap_z_lineage', 'fib_n_umap_s_lineage', 'fib_o_umap_z_region', 'fib_p_umap_s_region']]
ims = [[Image.open(OUT / f'{n}.png').convert('RGB') for n in r] for r in rows]
sc = 0.35; ims = [[im.resize((int(im.width * sc), int(im.height * sc)), Image.LANCZOS) for im in r] for r in ims]
W = max(sum(im.width for im in r) + 30 * (len(r) - 1) for r in ims); Hh = sum(max(im.height for im in r) for r in ims) + 30 * (len(ims) - 1)
page = Image.new('RGB', (W + 40, Hh + 40), 'white'); y = 20
for r in ims:
    x = 20
    for im in r: page.paste(im, (x, y)); x += im.width + 30
    y += max(im.height for im in r) + 30
page.save(OUT / 'Fibrosis_lesion_gradient_full.png'); print('full page', page.size)
