#!/usr/bin/env python3
"""Optimise the 7-region Morandi palette for regions that actually TOUCH in tissue."""
import sys, numpy as np, anndata as ad
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, '/home/yancui/haruka_v2/figures_paper/palette')
from search_morandi import oklch_to_srgb, lin_to_srgb, lin_to_oklab, CVD, to_hex

a = ad.read_h5ad('/home/yancui/haruka_v2/bench_mouse_brain_scviva/data_composition_pairs_all/mouse_brain_B5_paired.h5ad', backed='r')
dis = a.obs.condition.astype(str).to_numpy() == 'disease'
y = a.obs.gt_background_subregion.astype(str).to_numpy()[dis]; xy = np.asarray(a.obsm['spatial'])[dis]; a.file.close()
cls = sorted(np.unique(y)); idx = {c: i for i, c in enumerate(cls)}; yi = np.array([idx[c] for c in y])
nb = NearestNeighbors(n_neighbors=7).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
A = np.zeros((7, 7))
for j in range(nb.shape[1]): np.add.at(A, (yi, yi[nb[:, j]]), 1)
A = A + A.T; np.fill_diagonal(A, 0); frac = A / A.sum()
adj = [(i, j) for i in range(7) for j in range(i + 1, 7) if frac[i, j] > .005]
print('regions:', cls)
print('touching pairs (share of all cross-region contacts):')
for i, j in sorted(adj, key=lambda p: -frac[p]): print(f'   {cls[i]:11s} - {cls[j]:11s} {frac[i, j]*100:5.1f}%')

def dE(lins, pairs):
    res = {}
    for k, M in [('normal', None)] + list(CVD.items()):
        X = np.array(lins) if M is None else np.clip(np.array(lins) @ M.T, 0, 1)
        L = lin_to_oklab(X); res[k] = min(np.sqrt(((L[i] - L[j]) ** 2).sum()) * 100 for i, j in pairs)
    return res

allp = [(i, j) for i in range(7) for j in range(i + 1, 7)]
rng = np.random.default_rng(3); best = (-1,)
anch = {'CC/ACO': 245, 'CTX_L1/MEN': 80, 'CTX_L2/3': 150, 'CTX_L4/5/6': 350, 'STR_CP/ACB': 220, 'STR_LS/NDB': 35, 'VEN': 305}
hue0 = np.array([anch[c] for c in cls])
for _ in range(60000):
    h = hue0 + rng.normal(0, 12, 7); L = rng.uniform(.50, .84, 7); C = rng.uniform(.05, .095, 7)
    lins = []
    for Li, Ci, hi in zip(L, C, h):
        lin, _s = oklch_to_srgb(Li, Ci, hi)
        if (lin < -1e-4).any() or (lin > 1 + 1e-4).any(): break
        lins.append(np.clip(lin, 0, 1))
    if len(lins) < 7: continue
    t = dE(lins, adj)
    obj = min(t['normal'] / 15, min(t['protan'], t['deutan'], t['tritan']) / 8)
    if obj > best[0]: best = (obj, [to_hex(lin_to_srgb(l)) for l in lins], t, dE(lins, allp), C, L)
obj, hexes, t, ta, C, L = best
print('\nREGION (Morandi, adjacency-optimised):')
for c, hx, ci, li in zip(cls, hexes, C, L): print(f'   {c:11s} {hx}   chroma {ci:.3f}  lightness {li:.2f}')
print(f'touching pairs  worst dE x100: normal {t["normal"]:.1f} protan {t["protan"]:.1f} deutan {t["deutan"]:.1f} tritan {t["tritan"]:.1f}')
print(f'all pairs       worst dE x100: normal {ta["normal"]:.1f} protan {ta["protan"]:.1f} deutan {ta["deutan"]:.1f} tritan {ta["tritan"]:.1f}')
