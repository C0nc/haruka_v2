#!/usr/bin/env python3
"""Assemble whole-page vector SVGs from the per-panel SVGs (same row layouts as the PNG page previews).

Each panel SVG is nested as an <svg> element, so text stays editable and vectors stay vectors; ids are
prefixed per panel so clip paths and markers from different matplotlib files do not collide.
    python compose_svg_pages.py            # all pages
    python compose_svg_pages.py kras fig3  # some pages
Pages: fig3 (Fig3, EDFig3, EDFig4), exercise_blocks, exercise_clock, petracer, fibrosis, kras.
Figure 2 already has a vector page: fig2/make_fig2.py writes fig2/fig2_composite.svg.
"""
import re, sys, glob
from pathlib import Path
import numpy as np

FP = Path(__file__).resolve().parent
GAP = 6.0          # pt between panels and rows
MARGIN = 8.0       # pt around the page
_HEAD = re.compile(r'<svg\b[^>]*>', re.S)
_LETTER = re.compile(r'<text style="font-weight: 700; font-size: 8px;[^>]*>[a-z]{1,2}</text>')   # ps.panel_label
LETTER_STYLE = "font-weight: 700; font-size: 8px; font-family: 'Arial', 'Helvetica', 'DejaVu Sans', sans-serif; fill: #2e2b28"


def read_panel(path, k, strip_letter=False):
    """Return (width, height, inner markup with ids prefixed) of a matplotlib SVG, in its viewBox units (pt)."""
    s = Path(path).read_text()
    head = _HEAD.search(s); vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', head.group(0)).group(1).split()]
    body = s[head.end():s.rindex('</svg>')]
    body = re.sub(r'<metadata>.*?</metadata>', '', body, flags=re.S)
    if strip_letter: body = _LETTER.sub('', body)
    p = f'p{k}_'
    body = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{p}{m.group(1)}"', body)
    body = re.sub(r'url\(#([^)]+)\)', lambda m: f'url(#{p}{m.group(1)})', body)
    body = re.sub(r'(xlink:href|href)="#([^"]+)"', lambda m: f'{m.group(1)}="#{p}{m.group(2)}"', body)
    return vb[2], vb[3], body


class Page:
    def __init__(self): self.items, self.k = [], 0

    def panel(self, path, crop=None, letter=None):
        """crop = (y0, y1) as fractions of the panel height, for panels drawn as one multi-row file;
        letter = replace the panel's own letter with this one, placed at the slot's top-left (as make_fig3_full.py)."""
        w, h, body = read_panel(path, self.k, strip_letter=letter is not None); self.k += 1
        y0, y1 = crop or (0.0, 1.0)
        return dict(w=w, h=h * (y1 - y0), vb=(0, h * y0, w, h * (y1 - y0)), body=body, letter=letter)

    def place(self, p, x, y, scale):
        self.items.append((p, x, y, scale))

    def write(self, out, W, H):
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" '
                 f'width="{W:.2f}pt" height="{H:.2f}pt" viewBox="0 0 {W:.2f} {H:.2f}">',
                 f'<rect width="{W:.2f}" height="{H:.2f}" fill="#ffffff"/>']
        for p, x, y, s in self.items:
            vb = ' '.join(f'{v:.3f}' for v in p['vb'])
            parts.append(f'<svg x="{x:.3f}" y="{y:.3f}" width="{p["w"] * s:.3f}" height="{p["h"] * s:.3f}" '
                         f'viewBox="{vb}" overflow="hidden">{p["body"]}</svg>')
            if p['letter']:
                parts.append(f'<text style="{LETTER_STYLE}" x="{x - 4:.3f}" y="{y + 6:.3f}">{p["letter"]}</text>')
        parts.append('</svg>')
        Path(out).write_text('\n'.join(parts)); print('wrote', out, f'{W / 72 * 25.4:.0f} x {H / 72 * 25.4:.0f} mm')


def stretch_rows(rows, out, W=None):
    """Every row scaled to the same width (as compose_montage.rows_montage / make_fig3_full)."""
    pg = Page(); R = [[pg.panel(**a) if isinstance(a, dict) else pg.panel(a) for a in r] for r in rows]
    W = W or max(sum(p['w'] for p in r) + GAP * (len(r) - 1) for r in R)
    y = MARGIN
    for r in R:
        inner = W - GAP * (len(r) - 1)
        hrow = inner / sum(p['w'] / p['h'] for p in r)     # common row height filling the width
        x = MARGIN
        for p in r:
            s = hrow / p['h']; pg.place(p, x, y, s); x += p['w'] * s + GAP
        y += hrow + GAP
    pg.write(out, W + 2 * MARGIN, y - GAP + MARGIN)


def native_rows(rows, out):
    """Panels at their drawn size, left-aligned rows (as the fibrosis/KRAS PNG previews)."""
    pg = Page(); R = [[pg.panel(a) for a in r] for r in rows]
    W = max(sum(p['w'] for p in r) + GAP * (len(r) - 1) for r in R); y = MARGIN
    for r in R:
        x = MARGIN
        for p in r: pg.place(p, x, y, 1.0); x += p['w'] + GAP
        y += max(p['h'] for p in r) + GAP
    pg.write(out, W + 2 * MARGIN, y - GAP + MARGIN)


def fig3():
    sys.path.insert(0, str(FP / 'fig3')); from fig3_layout import PAGES, LET
    import matplotlib.pyplot as plt
    d = FP / 'fig3'
    # panel t was drawn as one 2-row figure (STARmap top, CODEX bottom): split at the blank row, as make_fig3_full.py does
    im = plt.imread(d / 'fig3t_s_umap_markers.png'); blank = (im[..., :3].min(2) > 0.99).all(1)
    mid = im.shape[0] // 2; cand = np.where(blank[int(mid * 0.8):int(mid * 1.2)])[0]
    f = (int(mid * 0.8) + (cand[len(cand) // 2] if len(cand) else mid - int(mid * 0.8))) / im.shape[0]
    T = {'fig3t_starmap_s_umap_markers': (0.0, f), 'fig3t_codex_s_umap_markers': (f, 1.0)}
    for page, rows in PAGES.items():
        stretch_rows([[dict(path=d / ('fig3t_s_umap_markers.svg' if s in T else f'{s}.svg'), crop=T.get(s), letter=LET(s)) for s in r] for r in rows],
                     d / 'final' / f'{page}_full.svg', W=183 / 25.4 * 72 - 2 * MARGIN)


def exercise_blocks():
    d = FP / 'exercise'
    rows = [['exb_a_z_umap_celltype', 'exb_b_s_umap_celltype', 'exb_c_z_umap_region', 'exb_d_s_umap_region'],
            ['exb_e_z_umap_vascular_clock', 'exb_f_s_umap_vascular_clock', 'exb_g_z_umap_vascular_condition', 'exb_h_s_umap_vascular_condition'],
            ['exb_i_tissue_celltype', 'exb_j_tissue_observed'], ['exb_k_tissue_z', 'exb_l_tissue_s'], ['exb_m_intra_vascular_partition']]
    stretch_rows([[d / f'{f}.svg' for f in r] for r in rows], d / 'Exercise_blocks_full.svg')


def exercise_clock():
    d = FP / 'exercise'; pg = Page()
    top = [pg.panel(d / f) for f in ['ex_a_observed.svg', 'ex_b_from_z.svg', 'ex_c_from_s.svg']]
    bot = pg.panel(d / 'ex_d_pred_vs_obs.svg')
    W = sum(p['w'] for p in top) + 2 * GAP
    hrow = (W - 2 * GAP) / sum(p['w'] / p['h'] for p in top); x = MARGIN
    for p in top: s = hrow / p['h']; pg.place(p, x, MARGIN, s); x += p['w'] * s + GAP
    s = min(0.7 * W / bot['w'], 1e9); y = MARGIN + hrow + GAP
    pg.place(bot, MARGIN + (W - bot['w'] * s) / 2, y, s)
    pg.write(d / 'Exercise_clock_shift_full.svg', W + 2 * MARGIN, y + bot['h'] * s + MARGIN)


def petracer():
    d = FP / 'petracer'; g = lambda p: sorted(glob.glob(str(d / f'{p}*.svg')))[0]
    rows = [['pt_a_', 'pt_b_', 'pt_c_', 'pt_d_'], ['pt_e_', 'pt_f_', 'pt_g_', 'pt_h_'], ['pt_i_', 'pt_j_', 'pt_k_'], ['pt_l_', 'pt_m_', 'pt_n_'], ['pt_o_']]
    stretch_rows([[g(p) for p in r] for r in rows], d / 'PEtracer_full.svg')


def fibrosis():
    d = FP / 'fibrosis'
    rows = [['fib_a_tissue_he', 'fib_b_tissue_lineage', 'fib_c_tissue_afb_s', 'fib_d_tissue_afb_z'],
            ['fib_e_region_pairs', 'fib_f_niche_alignment', 'fib_g_distance_supervised'],
            ['fib_h_distance_unsupervised', 'fib_i_profiles', 'fib_j_robustness'],
            ['fib_k_genes', 'fib_l_shift'],
            ['fib_m_umap_z_lineage', 'fib_n_umap_s_lineage', 'fib_o_umap_z_region', 'fib_p_umap_s_region']]
    native_rows([[d / f'{n}.svg' for n in r] for r in rows], d / 'Fibrosis_lesion_gradient_full.svg')


def kras():
    d = FP / 'kras'
    rows = [['kr_a_tissue_tumour', 'kr_b_tissue_mpas', 'kr_c_composition', 'kr_d_subtype_response'],
            ['kr_e_umap_z_type', 'kr_f_umap_s_type', 'kr_g_umap_z_mpas', 'kr_h_umap_s_mpas'],
            ['kr_i_te_axis', 'kr_j_tissue_nodule_te', 'kr_k_te_vs_response', 'kr_l_baseline_predicts_te'],
            ['kr_m_zpc2_genes', 'kr_n_zpc2_neighbours', 'kr_o_zpc2_heldout', 'kr_p_tissue_zpc2', 'kr_q_zpc2_control'],
            ['kr_r_niche_composition', 'kr_s_nhood_enrichment', 'kr_t_liana'],
            ['kr_u_example_hi', 'kr_v_example_lo']]
    native_rows([[d / f'{n}.svg' for n in r] for r in rows], d / 'KRAS_inhibitor_full.svg')


ALL = dict(fig3=fig3, exercise_blocks=exercise_blocks, exercise_clock=exercise_clock, petracer=petracer, fibrosis=fibrosis, kras=kras)
if __name__ == '__main__':
    for k in sys.argv[1:] or ALL: ALL[k]()
