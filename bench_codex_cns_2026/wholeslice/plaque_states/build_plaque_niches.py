#!/usr/bin/env python3
"""Per-plaque niche table for the whole-slice CODEX fit (all7_no_oligo).

For every official Abeta object (4 AD donors + Ctl-3, which also carries plaques):
  niche     = the 10 nearest segmented cells to the plaque centroid, all within 80 um
              (segmented cells are sparse, ~0.47 per 1000 um2, so a fixed radius leaves the
              median plaque with 2 cells; niches of neighbouring plaques overlap)
  features  = niche mean z, s (Haruka intrinsic), 34 input proteins, cell-type fractions,
              myeloid-state fractions (of IBA1 cells)
  delta     = niche mean minus the SAME donor's plaque-distant GREY-MATTER cells (> 100 um
              from any object); a whole-slice baseline is white-matter heavy and turns delta
              into a GM-vs-WM contrast
Plaque attributes kept: class (dense/diffuse), area, in-object A-beta / ApoE intensity, region.
"""
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
from sklearn.neighbors import NearestNeighbors

W = Path('/home/yancui/haruka_v2/bench_codex_cns_2026/wholeslice')
SRC = Path('/home/yancui/haruka_v2/bench_codex_cns_2026/external/CODEX-CNS/codex_notebooks/marker_intensities')
M = Path('/home/yancui/haruka_v2/results_codex_wholeslice/codex_wholeslice_all7_no_oligo/'
         'haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0')
PX = 0.3774
K, MAXD = 10, 80.0
FILES = {'AD-1': '11162021_Bahareh-3155_EDTA_abeta.csv', 'AD-2': '10202021_BaharehNP_2997-ADCortex_EDTA_abeta.csv',
         'AD-3': '10122021_BaharehNP_ADCortex_EDTA_abeta.csv', 'AD-4': '12032021-Bahareh_3280_Alz_abeta.csv',
         'Ctl-3': '10122021_BaharehNP_1-CtlCortex1796_EDTA_MKT6_abeta.csv'}

a = ad.read_h5ad(W / 'codex_wholeslice_all7_no_oligo.h5ad')
assert (np.load(M / 'obs_names.npy', allow_pickle=True) == a.obs_names.to_numpy()).all()
Z, S = np.load(M / 'background.npy'), np.load(M / 'salient.npy')
P = np.asarray(a.X); prot = list(a.var_names); o = a.obs
CT = ['Neuron', 'Astrocyte', 'Microglia']; MS = ['Homeostatic microglia', 'MHC-II+ microglia',
                                                  'Activated microglia', 'Macrophage', 'Unresolved']
ctoh = np.stack([(o.cell_type == c).to_numpy(float) for c in CT], 1)
msoh = np.stack([(o.myeloid_state.astype(str) == c).to_numpy(float) for c in MS], 1)
xy = a.obsm['spatial'].astype(float)
cells = {'z': Z, 's': S, 'p': P}
rows, feats = [], {k: [] for k in ['z', 's', 'p', 'ct', 'ms']}
dist_any = np.full(len(o), np.inf)
for don, f in FILES.items():
    pl = pd.read_csv(SRC / f)
    m = np.where((o.donor_id == don).to_numpy())[0]
    nn = NearestNeighbors(n_neighbors=K).fit(xy[m])
    dist_any[m] = NearestNeighbors(n_neighbors=1).fit(pl[['x', 'y']].to_numpy(float)).kneighbors(xy[m])[0][:, 0] * PX
    gm = (o.region.to_numpy()[m] == 'GM')
    far = m[(dist_any[m] > 100) & gm]                           # GM-matched baseline (plaques sit in GM)
    print(don, 'GM distal baseline cells', len(far), flush=True)
    base = {k: v[far].mean(0) for k, v in cells.items()}
    base['ct'] = ctoh[far].mean(0)
    dd, nb = nn.kneighbors(pl[['x', 'y']].to_numpy(float))
    abcol = next((c for c in ['a-Beta', 'A-Beta'] if c in pl), None)    # AD-4 header merges it with PSD-95
    for i, idx in enumerate(nb):
        if dd[i, -1] * PX > MAXD: continue
        g = m[idx]; my = g[o.cell_type.to_numpy()[g] == 'Microglia']
        rows.append(dict(donor=don, condition='AD' if don.startswith('AD') else 'control', plaque=int(pl.cell[i]),
                         cls=pl['class'][i], area_um2=pl.area_um2[i], solidity=pl.cell_solidity[i],
                         abeta=pl[abcol][i] if abcol else np.nan, apoe_in_plaque=pl.ApoE[i],
                         region=o.region.to_numpy()[g].astype(str).tolist().count('WM') / len(g),
                         n_cells=len(g), n_microglia=len(my)))
        for k, v in cells.items(): feats[k].append(v[g].mean(0) - base[k])
        feats['ct'].append(ctoh[g].mean(0) - base['ct'])
        feats['ms'].append(msoh[my].mean(0) if len(my) else np.full(len(MS), np.nan))
T = pd.DataFrame(rows).rename(columns={'region': 'wm_fraction'})
np.savez_compressed('plaque_niches.npz', dz=np.array(feats['z']), ds=np.array(feats['s']), dp=np.array(feats['p']),
                    dct=np.array(feats['ct']), ms=np.array(feats['ms']), prot=np.array(prot), ct=np.array(CT), mstates=np.array(MS))
T.to_csv('plaque_niches.csv', index=False)
np.save('cell_dist_any_abeta_um.npy', dist_any)
print(T.groupby(['donor', 'cls']).size().unstack().to_string())
print('abeta missing (AD-4 header merge):', T.abeta.isna().groupby(T.donor).mean().round(2).to_dict())
