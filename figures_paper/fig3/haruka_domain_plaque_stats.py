#!/usr/bin/env python3
"""Haruka outputs vs plaques (no post-hoc decoders).
  domains  MiniBatchKMeans(k=12, seed 0) on the condition-only CellCharter embeddings z-CC and s-CC
           (STARmap: the official final_results_nb assignments; CODEX: same protocol on the whole-slice fit)
  plaque   CODEX: cell <= 30 um of an Abeta object; STARmap: <= 2 cell spacings of an Abeta-associated cell
  per domain: fraction plaque-adjacent, log2 enrichment vs all condition cells
  summary: NMI(domain, plaque-adjacent) and the share of plaque-adjacent cells captured by the
           top-enriched domains covering 20% of cells (z vs s)
  shift    Haruka's counterfactual salient composition shift (microglia column) by distance bin
"""
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
import numpy as np, pandas as pd, anndata as ad, warnings
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
warnings.filterwarnings('ignore')
from pathlib import Path
H = Path(f'{_HR}'); OUT = Path(__file__).resolve().parent


def summarise(tag, dom, near):
    rows = []
    for d in np.unique(dom):
        m = dom == d
        rows.append(dict(domain=int(d), n=int(m.sum()), frac_near=near[m].mean(), log2_enr=np.log2(near[m].mean() / near.mean() + 1e-9)))
    R = pd.DataFrame(rows).sort_values('log2_enr', ascending=False)
    cum = R.n.cumsum() / R.n.sum(); top = R[cum.shift(fill_value=0) < 0.2]
    capt = (R.set_index('domain').loc[top.domain, 'frac_near'] * top.set_index('domain').n).sum() / near.sum()
    print(f'  {tag}: NMI {nmi(near, dom):.4f} | log2 enrichment range {R.log2_enr.min():+.2f} .. {R.log2_enr.max():+.2f} | '
          f'top domains ({top.n.sum() / R.n.sum():.0%} of cells) hold {capt:.0%} of plaque-adjacent cells')
    return R

# ---------------- STARmap ----------------
b = ad.read_h5ad(H / 'bench_real_discovery/data/ad.h5ad', backed='r'); ob = b.obs
A = pd.read_csv(H / 'final_results_nb/starmap_ad/biology/domain_annotation/domain_assignments.csv.gz').set_index('cell_id')
Dt = pd.read_csv(H / 'final_results_nb/starmap_ad/biology/plaque_distance/starmap_plaque_distance_cells.csv.gz').set_index('cell_id')
A = A.join(Dt.distance_to_abeta_associated_cell_spatial_units.rename('d'))
near = (A.d <= 2 * 31.4).to_numpy()
print(f'STARmap: {len(A)} condition cells, {near.mean():.1%} plaque-adjacent')
Rz = summarise('z-CC', A.z_cc_domain.to_numpy(), near); Rs = summarise('s-CC', A.s_cc_domain.to_numpy(), near)
Rz.assign(rep='z', dataset='STARmap').to_csv(OUT / 'haruka_domain_plaque_starmap_z.csv', index=False)
Rs.assign(rep='s', dataset='STARmap').to_csv(OUT / 'haruka_domain_plaque_starmap_s.csv', index=False)
MD = H / 'results_real_compgated_lambda10_samplebatch_sepheads/ad/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
sh = np.load(MD / 'salient_composition_shift.npy')[:, 9]                 # 'Micro'
dd = pd.Series(b.obs_names).map(Dt.distance_to_abeta_associated_cell_spatial_units).to_numpy(float)
dis = ob.condition.astype(str).to_numpy() == 'disease'
e = np.array([-0.5, 1e-9, 1, 2, 4, 8, np.inf]) * 31.4
print('  Haruka salient microglia-share shift by Abeta distance bin:',
      [round(float(sh[dis & (dd > e[i]) & (dd <= e[i + 1])].mean()), 4) for i in range(6)], ' (control mean', round(float(sh[~dis].mean()), 4), ')')

# ---------------- CODEX ----------------
CC = H / 'results_codex_wholeslice/codex_wholeslice_all7_no_oligo/haruka_z_s_cellcharter_cc3_condition_only'
a = ad.read_h5ad(H / 'bench_codex_cns_2026/wholeslice/codex_wholeslice_all7_no_oligo.h5ad', backed='r'); o = a.obs
src = np.load(CC / 'source_rows.npy')
dist = np.load(H / 'bench_codex_cns_2026/wholeslice/plaque_states/cell_dist_any_abeta_um.npy')
nearC = dist[src] <= 30
dz = MiniBatchKMeans(12, random_state=0, n_init=10, batch_size=4096).fit_predict(np.load(CC / 'spatial.npy'))
ds = MiniBatchKMeans(12, random_state=0, n_init=10, batch_size=4096).fit_predict(np.load(CC / 'salient_spatial.npy'))
print(f'\nCODEX: {len(src)} condition cells, {nearC.mean():.1%} plaque-adjacent')
Rz = summarise('z-CC', dz, nearC); Rs = summarise('s-CC', ds, nearC)
gm = o.region.to_numpy()[src] == 'GM'
print('  within grey matter only:'); summarise('z-CC GM', dz[gm], nearC[gm]); summarise('s-CC GM', ds[gm], nearC[gm])
pd.DataFrame({'obs_row': src, 'z_cc_domain': dz, 's_cc_domain': ds}).to_csv(OUT / 'haruka_domains_codex_wholeslice.csv.gz', index=False, compression='gzip')
Rz.assign(rep='z', dataset='CODEX').to_csv(OUT / 'haruka_domain_plaque_codex_z.csv', index=False)
Rs.assign(rep='s', dataset='CODEX').to_csv(OUT / 'haruka_domain_plaque_codex_s.csv', index=False)
MDc = H / 'results_codex_wholeslice/codex_wholeslice_all7_no_oligo/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
shc = np.load(MDc / 'salient_composition_shift.npy')[:, 1]             # 'Microglia'
disc = (o.condition.astype(str) == 'disease').to_numpy() & (o.region.to_numpy() == 'GM')
ec = [0, 10, 20, 40, 60, 100, np.inf]
print('  Haruka salient microglia-share shift by distance bin (GM):',
      [round(float(shc[disc & (dist > ec[i]) & (dist <= ec[i + 1])].mean()), 4) for i in range(6)])
