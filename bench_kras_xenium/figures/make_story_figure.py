#!/usr/bin/env python3
"""Figure: what the salient spatial embedding actually encodes, and the real KRAS response."""
from __future__ import annotations
import numpy as np, pandas as pd, anndata as ad, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

SEED = 0
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8a86', '#e3e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white'})

KCC = ('/home/yancui/haruka_v2/results_real_compgated_nb_final/mouselung_kras_xenium_clean_noC3/'
       'haruka_z_s_cellcharter_cc3_condition_only/')
CCC = ('/home/yancui/haruka_v2/results_codex_channelnorm_ab/codex_cns_myeloid_primary_no_aml_channelnorm/'
       'haruka_z_s_cellcharter_cc3_condition_only/')
BIO = '/home/yancui/haruka_v2/bench_kras_xenium/biological_responder/'
TUM = ['Tumor_EMT', 'Tumor_epithelial', 'Tumor_epithelial_Itgb6']

def pc1(x):
    return PCA(n_components=1, random_state=SEED).fit_transform(StandardScaler().fit_transform(x))[:, 0]

CACHE = '/home/yancui/haruka_v2/bench_kras_xenium/figures/story_cache.npz'
import os
if os.path.exists(CACHE):
    _c = np.load(CACHE, allow_pickle=True)
    k_auc, k_rho = list(_c['k_auc']), list(_c['k_rho'])
    c_auc, c_mor = list(_c['c_auc']), list(_c['c_mor'])
    pred = dict(_c['pred'].item()); p_scc = _c['p_scc']; lf_sub = _c['lf_sub']
    meds = list(_c['meds']); rate_vals = list(_c['rate_vals'])
    print('loaded cache', flush=True)
else:
  print('loading KRAS...', flush=True)
  h = ad.read_h5ad('/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad', backed='r')
  rows = np.load(KCC + 'source_rows.npy')
  ko = h.obs.iloc[rows][['slice_id', 'celltype', 'x_centroid', 'y_centroid']].copy()
  ko['cid'] = h.obs['cell_id'].iloc[rows].astype(str).to_numpy()
  h.file.close()
  ksl = ko.slice_id.astype(str).to_numpy(); kct = ko.celltype.astype(str).to_numpy()
  kxy = ko[['x_centroid', 'y_centroid']].to_numpy(float); is_t = np.isin(kct, TUM)
  lf = np.zeros(len(ksl))
  for s in np.unique(ksl):
      i = np.where(ksl == s)[0]
      nb = NearestNeighbors(n_neighbors=21).fit(kxy[i]).kneighbors(kxy[i], return_distance=False)[:, 1:]
      lf[i] = is_t[i][nb].mean(1)
  rng = np.random.default_rng(SEED); sub = np.sort(rng.choice(len(ksl), 120000, replace=False))
  ks = np.load(KCC + 'salient.npy'); kscc = np.load(KCC + 'salient_spatial.npy')

  print('KRAS ladder...', flush=True)
  k_auc, k_rho = [], []
  for blk in [ks[sub]] + [kscc[sub][:, i*32:(i+1)*32] for i in range(3)]:
      p = pc1(blk); a = roc_auc_score(is_t[sub].astype(int), p)
      k_auc.append(max(a, 1-a)); k_rho.append(abs(spearmanr(p, lf[sub]).statistic))

  print('CODEX ladder...', flush=True)
  ca = ad.read_h5ad('/home/yancui/haruka_v2/bench_codex_cns_2026/data/processed/'
                    'codex_cns_myeloid_primary_no_aml_channelnorm.h5ad')
  csrc = np.load(CCC + 'source_rows.npy'); co = ca.obs.iloc[csrc]
  cdon = co['donor_id'].astype(str).to_numpy(); creg = co['region'].astype(str).to_numpy()
  cxy = ca.obsm['spatial'][csrc].astype(float)
  cnb = {d: (np.where(cdon == d)[0],) for d in np.unique(cdon)}
  for d in cnb:
      i = cnb[d][0]
      cnb[d] = (i, NearestNeighbors(n_neighbors=21).fit(cxy[i]).kneighbors(cxy[i], return_distance=False)[:, 1:])
  cs = np.load(CCC + 'salient.npy'); cscc = np.load(CCC + 'salient_spatial.npy')
  def moran(x):
      o = []
      for d, (i, nb) in cnb.items():
          v = x[i] - x[i].mean(0)
          o.append((v * v[nb].mean(1)).sum(0) / np.maximum((v * v).sum(0), 1e-12))
      return float(np.mean(o))
  c_auc, c_mor = [], []
  for blk in [cs] + [cscc[:, i*32:(i+1)*32] for i in range(3)]:
      p = pc1(blk)
      c_auc.append(np.mean([max(v, 1-v) for v in
          [roc_auc_score((creg[cdon == d] == 'WM').astype(int), p[cdon == d]) for d in np.unique(cdon)]]))
      c_mor.append(moran(blk))

  print('biology...', flush=True)
  bio = pd.read_csv(BIO + 'treated_tumor_biological_response.csv.gz')
  allsig = pd.read_csv(BIO + 'all_tumor_signature_scores.csv.gz')
  pos = pd.Series(np.arange(len(ko)), index=ko.cid.to_numpy())
  bl = pos.reindex(bio.cell_id.astype(str)).to_numpy(); ok = np.isfinite(bl); bl = bl[ok].astype(int)
  y = bio.bio_responder.to_numpy()[ok].astype(int); bsl = bio.sl.to_numpy()[ok]
  pred = {}
  for tag, f in [('s intrinsic', 'salient.npy'), ('z intrinsic', 'background.npy'),
                 ('z-CC', 'spatial.npy'), ('s-CC', 'salient_spatial.npy')]:
      X = np.load(KCC + f)[bl].astype(float); sc = []
      for s in np.unique(bsl):
          tr, te = bsl != s, bsl == s
          ss = StandardScaler().fit(X[tr])
          clf = LogisticRegression(max_iter=3000, random_state=SEED,
                                   class_weight='balanced').fit(ss.transform(X[tr]), y[tr])
          sc.append(roc_auc_score(y[te], clf.predict_proba(ss.transform(X[te]))[:, 1]))
      pred[tag] = np.mean(sc)
  p_scc = pc1(kscc[sub]); lf_sub = lf[sub]
  meds = [allsig.loc[allsig.sl == s_, 'MAPK_MPAS'].median() for s_ in
          ['JB_C0','JB_C1','JB_C2','JB_M0','JB_M1','JB_M2','JB_M3']]
  _r = bio.groupby('ct')['bio_responder'].mean()*100
  rate_vals = [_r['Tumor_EMT'], _r['Tumor_epithelial'], _r['Tumor_epithelial_Itgb6']]
  np.savez(CACHE, k_auc=k_auc, k_rho=k_rho, c_auc=c_auc, c_mor=c_mor,
           pred=np.array(pred, dtype=object), p_scc=p_scc, lf_sub=lf_sub,
           meds=meds, rate_vals=rate_vals)
  print('cached', flush=True)

# ---------------- figure ----------------
fig = plt.figure(figsize=(12.6, 8.9))
gs = gridspec.GridSpec(3, 3, height_ratios=[0.42, 1, 1], hspace=0.68, wspace=0.34,
                       left=0.06, right=0.985, top=0.93, bottom=0.07)

axs = fig.add_subplot(gs[0, :]); axs.axis('off')
steps = ['Drug acts on\ntumour cells', 'Arms differ in the\ntumour compartment',
         'Salient latent learns\nTUMOUR IDENTITY', 'Tumour cells are\nspatially clustered',
         '3-hop smoothing →\nTUMOUR DENSITY']
for i, t in enumerate(steps):
    x = 0.022 + i * 0.198
    axs.add_patch(plt.Rectangle((x, 0.18), 0.166, 0.62, transform=axs.transAxes,
                                facecolor='#f4f6fa' if i < 4 else '#fdeee7',
                                edgecolor=BLUE if i < 4 else ORANGE, lw=1.1, zorder=1))
    axs.text(x + 0.083, 0.49, t, transform=axs.transAxes, ha='center', va='center',
             fontsize=8.6, color=INK, zorder=2)
    if i < 4:
        axs.annotate('', xy=(x + 0.194, 0.49), xytext=(x + 0.170, 0.49), xycoords=axs.transAxes,
                     textcoords=axs.transAxes, arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.2))
axs.text(0.0, 0.95, 'The mechanism: aggregation converts a per-cell identity into tissue architecture',
         transform=axs.transAxes, fontsize=10.5, color=INK, fontweight='bold')

hops = [0, 1, 2, 3]
axA = fig.add_subplot(gs[1, 0])
axA.plot(hops, k_auc, '-o', color=BLUE, lw=2, ms=6, label='tumour identity (AUROC)')
axA.plot(hops, k_rho, '-o', color=ORANGE, lw=2, ms=6, label='local tumour density (|ρ|)')
axA.text(3.04, k_auc[-1], f'{k_auc[-1]:.2f}', color=INK, va='center', fontsize=8)
axA.text(3.04, k_rho[-1], f'{k_rho[-1]:.2f}', color=INK, va='center', fontsize=8)
axA.set_title('A  KRAS: each hop buys architecture', loc='left', fontsize=9.5, color=INK)
axA.set_xlabel('aggregation hop'); axA.set_ylabel('value (0–1)')
axA.set_xticks(hops); axA.set_xticklabels(['0\n(s intrinsic)', '1', '2', '3'])
axA.set_ylim(0.2, 1.02); axA.set_xlim(-0.25, 3.55)
axA.grid(axis='y', color=GRID, lw=0.7); axA.set_axisbelow(True)
axA.legend(frameon=False, fontsize=7.6, loc='lower right')

axB = fig.add_subplot(gs[1, 1])
axB.plot(hops, c_auc, '-o', color=BLUE, lw=2, ms=6, label='grey/white matter (AUROC)')
axB.plot(hops, c_mor, '-o', color=ORANGE, lw=2, ms=6, label="spatial smoothness (Moran's I)")
axB.text(3.04, c_auc[-1], f'{c_auc[-1]:.2f}', color=INK, va='center', fontsize=8)
axB.text(3.04, c_mor[-1], f'{c_mor[-1]:.2f}', color=INK, va='center', fontsize=8)
axB.set_title('B  CODEX AD: the same ladder, different organ', loc='left', fontsize=9.5, color=INK)
axB.set_xlabel('aggregation hop'); axB.set_ylabel('value (0–1)')
axB.set_xticks(hops); axB.set_xticklabels(['0\n(s intrinsic)', '1', '2', '3'])
axB.set_ylim(0.2, 1.02); axB.set_xlim(-0.25, 3.55)
axB.grid(axis='y', color=GRID, lw=0.7); axB.set_axisbelow(True)
axB.legend(frameon=False, fontsize=7.6, loc='lower right')

axC = fig.add_subplot(gs[1, 2])
hb = axC.hexbin(p_scc, lf_sub, gridsize=42, bins='log', cmap='Blues', mincnt=1, linewidths=0)
axC.set_title('C  What s-CC PC1 is: a tumour-density axis', loc='left', fontsize=9.5, color=INK)
axC.set_xlabel('s-CC PC1'); axC.set_ylabel('local tumour fraction (20-NN)')
axC.text(0.03, 0.94, f'|ρ| = {k_rho[-1]:.2f}\n|ρ| with drug response = 0.09',
         transform=axC.transAxes, va='top', fontsize=8.2, color=INK)
cb = fig.colorbar(hb, ax=axC, pad=0.02); cb.set_label('cells (log)', fontsize=7.5)
cb.ax.tick_params(labelsize=7)

axD = fig.add_subplot(gs[2, 0])
order = ['JB_C0', 'JB_C1', 'JB_C2', 'JB_M0', 'JB_M1', 'JB_M2', 'JB_M3']
cols = [BLUE]*3 + [ORANGE]*4
axD.bar(range(7), meds, color=cols, width=0.68)
axD.axhline(0, color=MUTED, lw=0.9)
for i, v in enumerate(meds):
    axD.text(i, v + (0.006 if v >= 0 else -0.012), f'{v:+.2f}', ha='center',
             va='bottom' if v >= 0 else 'top', fontsize=7.4, color=INK)
axD.set_xticks(range(7)); axD.set_xticklabels([s.replace('JB_', '') for s in order], fontsize=7.6)
axD.set_title('D  Real response readout: MAPK output', loc='left', fontsize=9.5, color=INK)
axD.set_ylabel('median MAPK MPAS')
from matplotlib.patches import Patch
axD.legend(handles=[Patch(facecolor=BLUE, label='control'),
                    Patch(facecolor=ORANGE, label='KRASi-treated')],
           frameon=False, fontsize=7.8, loc='lower left', bbox_to_anchor=(0.0, -0.40), ncol=2)
axD.grid(axis='y', color=GRID, lw=0.7); axD.set_axisbelow(True)

axE = fig.add_subplot(gs[2, 1])
vals = rate_vals
bars = axE.barh(range(3), vals, color=[BLUE, BLUE, ORANGE], height=0.6)
axE.axvline(25, color=INK2, ls='--', lw=1.2)
axE.text(25.6, 2.42, 'null (25%)', color=INK2, fontsize=7.6)
for i, v in enumerate(vals):
    axE.text(v + 0.7, i, f'{v:.1f}%', va='center', fontsize=8.2, color=INK)
axE.set_yticks(range(3)); axE.set_yticklabels(['EMT', 'epithelial', 'epithelial\nItgb6+'], fontsize=8)
axE.set_xlim(0, 50); axE.set_xlabel('responding cells (%)')
axE.set_title('E  Itgb6+ persisters do not respond', loc='left', fontsize=9.5, color=INK)
axE.grid(axis='x', color=GRID, lw=0.7); axE.set_axisbelow(True)

axF = fig.add_subplot(gs[2, 2])
keys = ['s intrinsic', 'z intrinsic', 'z-CC', 's-CC']
vals = [pred[k] for k in keys]
axF.barh(range(4), vals, color=[BLUE, BLUE, ORANGE, ORANGE], height=0.6)
axF.axvline(0.5, color=INK2, ls='--', lw=1.2)
axF.text(0.505, 3.42, 'chance', color=INK2, fontsize=7.6)
for i, v in enumerate(vals):
    axF.text(v + 0.006, i, f'{v:.3f}', va='center', fontsize=8.2, color=INK)
axF.set_yticks(range(4)); axF.set_yticklabels(keys, fontsize=8)
axF.set_xlim(0.45, 0.8); axF.set_xlabel('AUROC, leave-section-out')
axF.set_title('F  Spatial aggregation destroys the signal', loc='left', fontsize=9.5, color=INK)
axF.legend(handles=[Patch(facecolor=BLUE, label='per-cell'),
                    Patch(facecolor=ORANGE, label='spatially aggregated')],
           frameon=False, fontsize=7.8, loc='lower left', bbox_to_anchor=(0.0, -0.40), ncol=2)
axF.grid(axis='x', color=GRID, lw=0.7); axF.set_axisbelow(True)

out = '/home/yancui/haruka_v2/bench_kras_xenium/figures/salient_story.png'
fig.savefig(out, dpi=220, bbox_inches='tight'); fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
print('wrote', out)
print('KRAS ladder AUROC', np.round(k_auc, 3), 'rho', np.round(k_rho, 3))
print('CODEX ladder AUROC', np.round(c_auc, 3), 'moran', np.round(c_mor, 3))
print('pred', {k: round(v, 3) for k, v in pred.items()})
