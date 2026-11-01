#!/usr/bin/env python3
"""Cross-species comparison of the plaque marker axes (20 protein <-> ortholog pairs) and
programme-level view in STARmap (DAM / homeostatic microglia / disease-associated astrocytes)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ORTH = {'AQP4': 'Aqp4', 'ApoE': 'Apoe', 'CD11b': 'Itgam', 'CD11c': 'Itgax', 'CD14': 'Cd14', 'CD163': 'Cd163', 'CD44': 'Cd44', 'CD68': 'Cd68',
        'CD74': 'Cd74', 'GFAP': 'Gfap', 'IBA1': 'Aif1', 'PCNA': 'Pcna', 'S100B': 'S100b', 'SMA': 'Acta2', 'Synaptophysin': 'Syp',
        'TMEM119': 'Tmem119', 'Vimentin': 'Vim', 'Claudin5': 'Cldn5', 'Gal3': 'Lgals3', 'PSD95': 'Dlg4'}
C = pd.read_csv('marker_axes_codex.csv'); S = pd.read_csv('marker_axes_starmap.csv')
rng = np.random.default_rng(0)
out = []
for t in ['Microglia', 'Astrocyte', 'Neuron']:
    c = C[C.cell_type == t].set_index('feature'); s = S[S.cell_type == t].set_index('feature')
    P = pd.DataFrame({'protein': list(ORTH), 'gene': list(ORTH.values())})
    for k in ['grad', 'load_s', 'load_z', 's_only']:
        P[f'{k}_codex'] = c.loc[P.protein, k].to_numpy(); P[f'{k}_starmap'] = s.loc[P.gene, k].to_numpy()
    P['cell_type'] = t; out.append(P)
    print(f'\n===== {t}: 20 ortholog pairs =====')
    for k in ['grad', 'load_s', 'load_z', 's_only']:
        x, y = P[f'{k}_codex'], P[f'{k}_starmap']; r = spearmanr(x, y).statistic
        null = [spearmanr(x, rng.permutation(y.to_numpy())).statistic for _ in range(5000)]
        sign = (np.sign(x) == np.sign(y)).mean()
        print(f'  {k:7s} cross-species Spearman {r:+.2f} (perm p={np.mean(np.abs(null) >= abs(r)):.3f}), same sign {sign:.0%}')
    P['agree'] = np.sign(P.grad_codex) == np.sign(P.grad_starmap)
    show = P.assign(mag=P.grad_codex.abs() + P.grad_starmap.abs()).sort_values('mag', ascending=False).head(10)
    print('  strongest plaque gradients (grad: human / mouse | s_only: human / mouse):')
    for r in show.itertuples():
        print(f'    {r.protein:13s}/{r.gene:8s} grad {r.grad_codex:+.2f} / {r.grad_starmap:+.2f} {"AGREE" if r.agree else "DISAGREE":8s} '
              f'| s_only {r.s_only_codex:+.2f} / {r.s_only_starmap:+.2f}')
pd.concat(out).to_csv('cross_species_orthologs.csv', index=False)

# within-dataset: the top markers on the s plaque axis and beyond z
for name, T in [('CODEX', C), ('STARmap', S)]:
    print(f'\n===== {name}: top features (all features) =====')
    for t in ['Microglia', 'Astrocyte', 'Neuron']:
        x = T[T.cell_type == t]
        f = lambda col, asc: ', '.join(f'{r.feature} {getattr(r, col):+.2f}' for r in x.sort_values(col, ascending=asc).head(6).itertuples())
        print(f'  {t:9s} grad up:    {f("grad", False)}\n            grad down:  {f("grad", True)}\n'
              f'            s-only up:  {f("s_only", False)}\n            s-only down:{f("s_only", True)}')

# STARmap programmes
PROG = {'DAM': ['Trem2', 'Tyrobp', 'Cst7', 'Apoe', 'Lpl', 'Cd9', 'Itgax', 'Spp1', 'Clec7a', 'Cd68', 'Ctsb', 'Ctsd', 'Axl', 'Ccl6'],
        'Homeostatic MG': ['P2ry12', 'Tmem119', 'Cx3cr1', 'Sall1', 'Selplg', 'Csf1r'],
        'DAA (astro)': ['Gfap', 'Vim', 'Serpina3n', 'C4b', 'Cd44', 'Ctss', 'Osmr', 'Lcn2']}
print('\n===== STARmap programme genes present, mean over genes (grad | load_s | load_z | s_only) =====')
for t in ['Microglia', 'Astrocyte', 'Neuron']:
    x = S[S.cell_type == t].set_index('feature')
    for p, g in PROG.items():
        g = [i for i in g if i in x.index]
        v = x.loc[g, ['grad', 'load_s', 'load_z', 's_only']].mean()
        print(f'  {t:9s} {p:15s} ({len(g):2d} genes) grad {v.grad:+.3f} | s {v.load_s:+.3f} | z {v.load_z:+.3f} | s-only {v.s_only:+.3f}')
