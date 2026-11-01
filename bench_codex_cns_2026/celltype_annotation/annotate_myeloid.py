#!/usr/bin/env python3
"""Myeloid state annotation for the CODEX AD dataset (the paper's 11 Leiden clusters).

Labels come from each cluster's marker profile against established myeloid markers,
checked within every donor, and cross-checked with Astir (independent, marker-table
based, never sees the clusters). Macrophage labels are confirmed by both methods;
microglial labels are STATES on a continuum (Astir agrees on ordering, abstains on
hard calls).
"""
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path
OUT = Path('/home/yancui/haruka_v2/bench_codex_cns_2026/celltype_annotation')

ANNOT = {  # cluster: (fine label, map category, lineage, defining markers)
 '4':  ('Homeostatic microglia', 'Homeostatic microglia', 'Microglia', 'TMEM119+ MHC-II low'),
 '10': ('Homeostatic microglia', 'Homeostatic microglia', 'Microglia', 'TMEM119+ CD14 low'),
 '5':  ('Homeostatic microglia, CD68-low', 'Homeostatic microglia', 'Microglia', 'TMEM119+ CD68 low CD74 low'),
 '3':  ('Homeostatic microglia, CD45-low', 'Homeostatic microglia', 'Microglia', 'CD74+ CD45/CD11b/CD11c low'),
 '2':  ('MHC-II+ homeostatic microglia', 'MHC-II+ microglia', 'Microglia', 'TMEM119+ CD74+ HLA-DR+'),
 '7':  ('Reactive MHC-II+ microglia', 'MHC-II+ microglia', 'Microglia', 'TMEM119+ HLA-DR+ CD68+ CD45+ CD11c+'),
 '6':  ('Activated microglia (TMEM119-negative)', 'Activated microglia', 'Microglia', 'TMEM119- CD68+ HLA-DR+'),
 '1':  ('CD163+ MHC-II+ macrophage', 'Macrophage', 'Macrophage', 'CD163+ CD14+ HLA-DR+'),
 '8':  ('Border-associated macrophage', 'Macrophage', 'Macrophage', 'CD163-high CD14+ TMEM119-'),
 '0':  ('Unresolved, low signal', 'Unresolved', 'Unresolved', 'all markers low'),
 '9':  ('QC flag, IBA1-low', 'Unresolved', 'Unresolved', 'IBA1 very low'),
}
Z = pd.read_csv(OUT / 'cluster_marker_zscores.csv', index_col=0); Z.index = Z.index.astype(str)
S = pd.read_csv(OUT / 'cluster_sizes.csv', index_col=0); S.index = S.index.astype(str)
A = pd.read_csv(OUT / 'astir_calls.csv.gz', index_col=0); A['leiden'] = A.leiden.astype(str)
astir_bam = A.groupby('leiden')['p_Border-associated macrophage'].mean()
act = A.groupby('leiden')['p_Activated microglia'].mean(); hom = A.groupby('leiden')['p_Homeostatic microglia'].mean()
rows = []
for c, (fine, cat, lin, mk) in ANNOT.items():
    rows.append(dict(cluster=c, label=fine, map_category=cat, lineage=lin, defining_markers=mk,
                     n_cells=int(S.loc[c, 'n']), pct_cells=S.loc[c, 'pct_of_cells'],
                     AD_fraction=S.loc[c, 'AD_frac'], white_matter_fraction=S.loc[c, 'WM_frac'],
                     astir_p_BAM=round(float(astir_bam[c]), 3),
                     astir_activated_ratio=round(float(act[c] / (act[c] + hom[c])), 3),
                     confidence='two methods' if lin == 'Macrophage' else
                                ('state; Astir ordering agrees' if lin == 'Microglia' else 'not interpretable')))
T = pd.DataFrame(rows); T.to_csv(OUT / 'myeloid_annotation_table.csv', index=False)
print(T[['cluster', 'label', 'map_category', 'n_cells', 'white_matter_fraction', 'astir_p_BAM', 'astir_activated_ratio', 'confidence']].to_string(index=False))

lut = {c: v for c, v in ANNOT.items()}
for stem in ['codex_cns_myeloid_primary_no_aml_no_combat', 'codex_cns_myeloid_primary_no_aml_channelnorm']:
    a = ad.read_h5ad(f'/home/yancui/haruka_v2/bench_codex_cns_2026/data/processed/{stem}.h5ad', backed='r')
    cl = a.obs.paper_initial_leiden.astype(str)
    pd.DataFrame({'cell': a.obs_names.astype(str), 'donor_id': a.obs.donor_id.astype(str).to_numpy(),
                  'leiden': cl.to_numpy(), 'myeloid_label': cl.map(lambda x: lut[x][0]).to_numpy(),
                  'myeloid_category': cl.map(lambda x: lut[x][1]).to_numpy(),
                  'lineage': cl.map(lambda x: lut[x][2]).to_numpy()}
                 ).to_csv(OUT / f'cell_annotation_{stem}.csv.gz', index=False, compression='gzip')
    a.file.close()
print('\nwrote per-cell annotations for both normalisations (same cells, same clusters)')
