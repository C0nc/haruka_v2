#!/usr/bin/env python3
"""One-page card of the shared Morandi palette, by colour role."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import paperstyle as ps
ps.setup()

ROLES = [
    ('Spatial background subregions', list(ps.REGION.items())),
    ('Salient domains', [(f'D{i}', c) for i, c in enumerate(ps.DOMAIN)] + [('no domain', ps.NO_DOMAIN)]),
    ('Method family', [('Haruka', ps.METHOD['haruka']), ('Haruka ablation', ps.METHOD['haruka_ablation']),
                       ('baseline', ps.METHOD['baseline'])]),
    ('Representation', [('real space', ps.LATENT['space']), ('background z', ps.LATENT['z']),
                        ('salient s', ps.LATENT['s'])]),
    ('Direction (paired difference)', [('Haruka better', ps.DELTA['up']), ('Haruka worse', ps.DELTA['down'])]),
    ('Condition', [('control', ps.CONDITION['control']), ('disease / treated', ps.CONDITION['disease'])]),
]
H = 128
fig = plt.figure(figsize=(183 * ps.MM, H * ps.MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_xlim(0, 183); ax.set_ylim(0, H)
ax.text(6, H - 6, 'Haruka paper palette — Morandi, low saturation', fontsize=8, fontweight='bold', va='top')
ax.text(6, H - 10.5, 'Every set checked by computed separation under normal and colour-blind vision.',
        fontsize=6, va='top', color=ps.NEUTRAL['dark'])
y = H - 20
for title, items in ROLES:
    ax.text(6, y + 3.2, title, fontsize=6.5, fontweight='bold', va='center')
    x = 6
    for name, hexc in items:
        ax.add_patch(plt.Rectangle((x, y - 6.5), 22, 7.5, color=hexc, ec=ps.NEUTRAL['light'], lw=.4))
        ax.text(x, y - 8.0, name, fontsize=5.8, va='top')
        ax.text(x, y - 11.0, hexc.upper(), fontsize=5.6, va='top', color=ps.NEUTRAL['dark'])
        x += 24.5
    y -= 18
fig.savefig(HERE / 'palette_card.svg'); fig.savefig(HERE / 'palette_card.png', dpi=300)
ok, sizes = ps.check_fonts(HERE / 'palette_card.svg'); print('palette card fonts', sizes, 'OK' if ok else 'BELOW MIN')
