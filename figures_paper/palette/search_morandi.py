#!/usr/bin/env python3
"""Search for low-saturation (Morandi-style) categorical palettes that stay separable.

Low chroma is what makes categories hard to tell apart, so separation is computed,
not judged by eye: maximise the worst pairwise OKLab distance under normal vision and
simulated protan / deutan / tritan vision, with chroma held in a muted band and
lightness kept dark enough to read against a white page.
"""
import numpy as np

def srgb_to_lin(c): c = np.asarray(c, float); return np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)
def lin_to_srgb(c): c = np.clip(c, 0, 1); return np.where(c <= .0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - .055)
M1 = np.array([[.4122214708, .5363325363, .0514459929], [.2119034982, .6806995451, .1073969566], [.0883024619, .2817188376, .6299787005]])
M2 = np.array([[.2104542553, .7936177850, -.0040720468], [1.9779984951, -2.4285922050, .4505937099], [.0259040371, .7827717662, -.8086757660]])
def lin_to_oklab(rgb): return (np.cbrt(rgb @ M1.T)) @ M2.T
def oklab_to_lin(lab):
    lms = (lab @ np.linalg.inv(M2).T) ** 3; return lms @ np.linalg.inv(M1).T
def oklch_to_srgb(L, C, h):
    lab = np.array([L, C * np.cos(np.radians(h)), C * np.sin(np.radians(h))])
    lin = oklab_to_lin(lab[None])[0]; return lin, lin_to_srgb(lin)
# Machado et al. 2009, severity 1.0
CVD = {'protan': np.array([[.152286, 1.052583, -.204868], [.114503, .786281, .099216], [-.003882, -.048116, 1.051998]]),
       'deutan': np.array([[.367322, .860646, -.227968], [.280085, .672501, .047413], [-.011820, .042940, .968881]]),
       'tritan': np.array([[1.255528, -.076749, -.178779], [-.078411, .930809, .147602], [.004733, .691367, .303900]])}
def to_hex(rgb): return '#' + ''.join(f'{int(round(v * 255)):02X}' for v in np.clip(rgb, 0, 1))
def hex_lin(h): return srgb_to_lin([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])

def score(lins):
    labs = {'normal': lin_to_oklab(np.array(lins))}
    for k, M in CVD.items(): labs[k] = lin_to_oklab(np.clip(np.array(lins) @ M.T, 0, 1))
    out = {}
    for k, L in labs.items():
        d = np.sqrt(((L[:, None] - L[None]) ** 2).sum(-1)) * 100; np.fill_diagonal(d, 1e9); out[k] = d.min()
    return out

def search(n, hue_anchors, Lr=(.56, .80), Cr=(.055, .095), iters=40000, seed=0):
    rng = np.random.default_rng(seed); best = (-1, None)
    for _ in range(iters):
        h = np.array(hue_anchors) + rng.normal(0, 9, n)
        L = rng.uniform(*Lr, n); C = rng.uniform(*Cr, n)
        lins, ok = [], True
        for Li, Ci, hi in zip(L, C, h):
            lin, s = oklch_to_srgb(Li, Ci, hi)
            if (lin < -1e-4).any() or (lin > 1 + 1e-4).any(): ok = False; break
            lins.append(np.clip(lin, 0, 1))
        if not ok: continue
        sc = score(lins)
        obj = min(sc['normal'] / 15, min(sc['protan'], sc['deutan']) / 8, sc['tritan'] / 8)
        if obj > best[0]: best = (obj, [to_hex(lin_to_srgb(l)) for l in lins], sc, (L, C, h))
    return best

if __name__ == '__main__':
    # Morandi hue anchors: dusty blue, sand/ochre, sage, plum, powder blue, terracotta, mauve
    region = search(7, [250, 75, 150, 345, 225, 40, 310], seed=1)
    domain = search(3, [290, 10, 130], Lr=(.58, .78), Cr=(.06, .10), seed=2)
    for name, (obj, hexes, sc, (L, C, h)) in [('REGION', region), ('DOMAIN', domain)]:
        print(f'{name}: {hexes}')
        print(f'   worst dE x100 -> normal {sc["normal"]:.1f}  protan {sc["protan"]:.1f}  deutan {sc["deutan"]:.1f}  tritan {sc["tritan"]:.1f}')
        print(f'   OKLCH chroma {np.round(C, 3)}  lightness {np.round(L, 2)}')
