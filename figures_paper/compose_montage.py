#!/usr/bin/env python3
"""Assemble the full-page PNG montages from the per-panel PNGs (exercise, PEtracer).

    python compose_montage.py exercise_blocks | exercise_clock | petracer
Each row is scaled to the full page width by the summed aspect ratio of its panels.
"""
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
import sys, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.image as mi

FP = Path(f'{_HR}/figures_paper')


def rows_montage(rows, out, width, dpi, shrink=(0.98, 0.97)):
    ims = [[mi.imread(f) for f in r] for r in rows]
    ar = [sum(i.shape[1] / i.shape[0] for i in r) for r in ims]
    hs = np.array([1 / a for a in ar]); Ht = hs.sum()
    fig = plt.figure(figsize=(width, width * Ht)); y = 1
    for r, h, a in zip(ims, hs / Ht, ar):
        x = 0
        for im in r:
            w = (im.shape[1] / im.shape[0]) / a
            ax = fig.add_axes([x, y - h * shrink[1], w * shrink[0], h * shrink[1]]); ax.imshow(im); ax.axis('off'); x += w
        y -= h
    fig.savefig(out, dpi=dpi); print('wrote', out)


def exercise_blocks():
    d = FP / 'exercise'
    rows = [['exb_a_z_umap_celltype', 'exb_b_s_umap_celltype', 'exb_c_z_umap_region', 'exb_d_s_umap_region'],
            ['exb_e_z_umap_vascular_clock', 'exb_f_s_umap_vascular_clock', 'exb_g_z_umap_vascular_condition', 'exb_h_s_umap_vascular_condition'],
            ['exb_i_tissue_celltype', 'exb_j_tissue_observed'], ['exb_k_tissue_z', 'exb_l_tissue_s'], ['exb_m_intra_vascular_partition']]
    rows_montage([[d / f'{f}.png' for f in r] for r in rows], d / 'Exercise_blocks_full.png', 14, 150)


def exercise_clock():
    d = FP / 'exercise'
    ims = [mi.imread(d / f) for f in ['ex_a_observed.png', 'ex_b_from_z.png', 'ex_c_from_s.png', 'ex_d_pred_vs_obs.png']]
    fig = plt.figure(figsize=(13, 9))
    for i, im in enumerate(ims[:3]):
        ax = fig.add_axes([i / 3, 0.42, 1 / 3, 0.58]); ax.imshow(im); ax.axis('off')
    ax = fig.add_axes([0.15, 0.0, 0.7, 0.4]); ax.imshow(ims[3]); ax.axis('off')
    fig.savefig(d / 'Exercise_clock_shift_full.png', dpi=200); print('wrote', d / 'Exercise_clock_shift_full.png')


def petracer():
    d = FP / 'petracer'
    g = lambda p: sorted(glob.glob(str(d / f'{p}*.png')))[0]
    rows = [['pt_a_', 'pt_b_', 'pt_c_', 'pt_d_'], ['pt_e_', 'pt_f_', 'pt_g_', 'pt_h_'], ['pt_i_', 'pt_j_', 'pt_k_'], ['pt_l_', 'pt_m_', 'pt_n_'], ['pt_o_']]
    rows_montage([[g(p) for p in r] for r in rows], d / 'PEtracer_full.png', 13, 150)


if __name__ == '__main__':
    {'exercise_blocks': exercise_blocks, 'exercise_clock': exercise_clock, 'petracer': petracer}[sys.argv[1]]()
