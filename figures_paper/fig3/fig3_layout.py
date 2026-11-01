"""Single source of truth for Figure 3 / Extended Data Fig. 3-4 panel letters and page layout.

Every panel script asks LET(stem) for its letter, so a subpanel SVG always carries the letter it
has on the assembled page.  Rows are tiled left to right by make_fig3_full.py.
"""
MAIN = [  # Figure 3: s captures the plaque-associated microglial niche (STARmap-led)
    ['fig3_starmap_u3_z_umap_distance', 'fig3_starmap_u4_s_umap_distance', 'fig3b_starmap_radial_profile', 'fig3e_seed_robustness'],
    ['fig3_starmap_m4_tissue_haruka_native', 'fig3_cf_bins_starmap', 'fig3_cf_decomposition_starmap'],
    ['fig3t_starmap_s_umap_markers'],
    ['fig3s_programmes_z_vs_s', 'fig3q_cross_species_microglia'],
]
ED3 = [   # Extended Data Fig. 3: robustness, CODEX quantification, marker table
    ['fig3a_codex_radial_profile', 'fig3c_codex_real_vs_shifted', 'fig3f_arm_swap'],
    ['fig3d_starmap_real_vs_shifted', 'fig3g_s_beyond_z'],
    ['fig3r_marker_heatmap', 'fig3_cf_bins_codex', 'fig3_cf_decomposition_codex'],
]
ED4 = [   # Extended Data Fig. 4: Haruka embeddings and tissue maps
    ['fig3_starmap_u1_z_umap_celltype', 'fig3_starmap_u2_s_umap_celltype', 'fig3_starmap_m1_tissue_celltype',
     'fig3_starmap_m2_tissue_z_score', 'fig3_starmap_m3_tissue_s_score'],
    ['fig3_codex_u1_z_umap_celltype', 'fig3_codex_u2_s_umap_celltype', 'fig3_codex_u3_z_umap_distance', 'fig3_codex_u4_s_umap_distance'],
    ['fig3_codex_m1_tissue_celltype', 'fig3_codex_m2_tissue_z_score', 'fig3_codex_m3_tissue_s_score', 'fig3_codex_m4_tissue_haruka_native'],
    ['fig3t_codex_s_umap_markers'],
]
PAGES = {'Fig3': MAIN, 'EDFig3': ED3, 'EDFig4': ED4}
_LET = {stem: chr(ord('a') + i) for page in PAGES.values() for i, stem in enumerate(s for row in page for s in row)}
_PAGE = {stem: name for name, page in PAGES.items() for row in page for stem in row}


def LET(stem: str) -> str:
    """Final letter of a panel ('' if the panel is not on any page)."""
    return _LET.get(stem, '')


def PAGE(stem: str) -> str:
    return _PAGE.get(stem, '')
