#!/usr/bin/env python3
"""Page preview of the Figure 3 plaque panels (183 mm wide). Each panel stays its own SVG;
this only tiles the 600-dpi PNG renders for review."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import paperstyle as ps
ps.setup()
rows = [['fig3a_codex_radial_profile', 'fig3b_starmap_radial_profile', 'fig3e_seed_robustness'],
        ['fig3c_codex_real_vs_shifted', 'fig3d_starmap_real_vs_shifted'],
        ['fig3f_arm_swap', 'fig3g_s_beyond_z']]
fig = plt.figure(figsize=(183 * ps.MM, 170 * ps.MM))
y = 1.0; H = [0.34, 0.33, 0.33]
for r, h in zip(rows, H):
    ims = [plt.imread(HERE / f'{n}.png') for n in r]
    ws = [im.shape[1] / im.shape[0] for im in ims]; tot = sum(ws); x = 0
    for im, w in zip(ims, ws):
        ax = fig.add_axes([x, y - h, w / tot * 0.98, h * 0.97]); ax.imshow(im); ax.axis('off'); x += w / tot
    y -= h
fig.savefig(HERE / 'fig3_plaque_composite.png', dpi=300); fig.savefig(HERE / 'fig3_plaque_composite.pdf')
print('ok')
