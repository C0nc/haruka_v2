#!/usr/bin/env python3
"""Assemble Figure 3 and Extended Data Figs 3-4 (183 mm wide page previews) from the subpanel
600-dpi renders, re-lettered by fig3_layout.py.  Each subpanel's old letter (top-left corner) is
masked and the final letter is placed by this script.  Subpanel SVGs are copied to final/ under
their final names."""
import sys, shutil
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import paperstyle as ps
from fig3_layout import PAGES, LET
ps.setup()
FIN = HERE / 'final'; FIN.mkdir(exist_ok=True)


def load(stem):
    if stem.startswith('fig3t_'):                       # t was drawn as one 2-row figure: STARmap top, CODEX bottom
        im = plt.imread(HERE / 'fig3t_s_umap_markers.png'); blank = (im[..., :3].min(2) > 0.99).all(1)
        mid = im.shape[0] // 2; cand = np.where(blank[int(mid * 0.8):int(mid * 1.2)])[0]
        h = int(mid * 0.8) + (cand[len(cand) // 2] if len(cand) else mid - int(mid * 0.8))
        im = im[:h] if 'starmap' in stem else im[h:]
    else:
        im = plt.imread(HERE / f'{stem}.png')
    im = im[..., :3].copy()
    im[:int(0.075 * min(im.shape[:2]) + 60), :int(0.06 * im.shape[1] + 60)] = 1.0   # mask the old letter
    return im


for page, rows in PAGES.items():
    ims = [[load(s) for s in r] for r in rows]
    ar = [sum(i.shape[1] / i.shape[0] for i in r) for r in ims]      # row width at unit height
    hs = np.array([1 / a for a in ar]); W = 183
    H = W * hs.sum() * 1.06
    fig = plt.figure(figsize=(W * ps.MM, H * ps.MM)); y = 1.0
    for r, stems, h, a_ in zip(ims, rows, hs / hs.sum(), ar):
        x = 0.0
        for im, s in zip(r, stems):
            w = (im.shape[1] / im.shape[0]) / a_             # row fills the page width
            ax = fig.add_axes([0.012 + x * 0.985, y - h + 0.004, w * 0.96, h * 0.96]); ax.imshow(im, interpolation='antialiased'); ax.axis('off')
            fig.text(0.004 + x * 0.985, y - 0.004, LET(s), fontsize=ps.PANEL_LABEL_PT, fontweight='bold', va='top')
            x += w
            src = HERE / ('fig3t_s_umap_markers.svg' if s.startswith('fig3t_') else f'{s}.svg')
            shutil.copy(src, FIN / f'{page}{LET(s)}_{s.replace("fig3_", "").replace("fig3", "")}.svg')
        y -= h
    fig.savefig(FIN / f'{page}_full.png', dpi=300); fig.savefig(FIN / f'{page}_full.pdf')
    plt.close(fig); print(page, f'{H:.0f} mm tall')
