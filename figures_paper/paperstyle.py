"""Shared style for every Haruka paper figure.

Import this in every figure script so typography and colour mean the same thing on
every page.  Sizes are final printed sizes: build each axes at its real millimetre
size and save without rescaling, so a 6 pt label is 6 pt in print.

Colour roles (one meaning per colour across the whole paper):
  METHOD    method family in benchmarks (identity is carried by axis labels, colour = family)
  LATENT    which representation: real coordinates / background z / salient s
  DELTA     direction of a paired difference only (better / worse)
  CONDITION experimental arm
  REGION    planted background subregions (fixed per anatomical class)
  DOMAIN    planted salient domains D0-D2
"""
from __future__ import annotations
import re
from pathlib import Path
import matplotlib as mpl

MM = 1 / 25.4
INK = '#2E2B28'          # warm near-black, matches the Morandi ground
MIN_FONT_PT = 5.5          # user requirement: every font strictly larger than 5 pt
PANEL_LABEL_PT = 8

# ---- Morandi palette: low-saturation, grey-tinged tones (fashion / interior design
# reference).  Low chroma makes categories harder to separate, so every set was chosen by
# computed separation (OKLab dE x100, normal + protan/deutan/tritan simulation), not by eye;
# see palette/search_morandi.py and palette/search_region_adjacency.py.
NEUTRAL = {'light': '#E8E4DF', 'mid': '#A8A29C', 'dark': '#5C5652'}
METHOD = {'haruka': '#3F5F86', 'haruka_ablation': '#93A9C6', 'baseline': '#DAD4CC'}
LATENT = {'space': '#5C5652', 'z': '#5D82A8', 's': '#C48A62'}
DELTA = {'up': '#46755A', 'down': '#E0A39B', 'zero': '#A8A29C'}   # separated by lightness, CVD dE >= 20
CONDITION = {'control': '#D3CEC8', 'disease': '#7A716C'}
# optimised for regions that TOUCH in tissue: every touching pair dE >= 17.5 normal, >= 10.5 CVD.
# Non-touching pairs may be similar; position and the region key separate them.
REGION = {'CC/ACO': '#39698E', 'CTX_L1/MEN': '#AE874D', 'CTX_L2/3': '#3C7564',
          'CTX_L4/5/6': '#B1868E', 'STR_CP/ACB': '#61ABC6', 'STR_LS/NDB': '#EBBEB8',
          'VEN': '#7F5387'}
DOMAIN = ['#C8A9DB', '#A6647D', '#7F9D5E']     # lilac / dusty rose / sage; all pairs dE >= 18.5 normal, >= 10 CVD
NO_DOMAIN = '#ECE9E5'
# CODEX myeloid states (map categories). Cells are intermixed, so ALL pairs must separate:
# worst dE 15.6 normal, >= 9.9 CVD, and >= 12.8 from the z / s colours.
MYELOID = {'Homeostatic microglia': '#60A079', 'MHC-II+ microglia': '#A796C7',
           'Activated microglia': '#FBA9A5', 'Macrophage': '#8A5076', 'Unresolved': '#D9D5D0'}
# major brain cell types, shared by CODEX (protein) and STARmap (RNA) via the crosswalk.
# all pairs dE >= 15.6 normal, >= 9.1 CVD, and >= 10.2 from the z / s colours.
MAJOR = {'Neuron': '#A7C6FE', 'Astrocyte': '#E8C488', 'Oligodendrocyte': '#82A86F',
         'Microglia': '#A67D94', 'Vascular': '#7B524F'}
# sequential ramp for distance-like quantities: built only from existing roles
# (light ground -> salient tone -> ink), so 'near the plaque' reads dark and warm
SEQUENTIAL = ['#ECE9E5', '#C48A62', '#2E2B28']
# muted diverging ramp for z-scored heatmaps (low -> mid -> high)
DIVERGING = ['#4E6E91', '#F4F1ED', '#A95C6E']


def setup() -> None:
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 6.5, 'axes.titlesize': 6.5, 'axes.labelsize': 6.5,
        'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.major.pad': 2, 'ytick.major.pad': 2, 'axes.labelpad': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
        'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
        'ytick.color': INK, 'text.color': INK,
        'savefig.dpi': 600, 'figure.dpi': 150,
    })


def panel_label(ax, letter: str, x: float = -0.2, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=PANEL_LABEL_PT,
            fontweight='bold', ha='left', va='bottom', color=INK)


def save(fig, path_no_ext: str | Path, png: bool = True) -> list[str]:
    """SVG (primary, editable text) + PDF; PNG preview optional."""
    p = Path(path_no_ext); p.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ('svg', 'pdf') + (('png',) if png else ()):
        f = f'{p}.{ext}'; fig.savefig(f, bbox_inches='tight', pad_inches=0.01); out.append(f)
    return out


_FS = [re.compile(r'font-size:\s*([0-9.]+)px'), re.compile(r'font:\s*(?:[a-z]+\s+)*([0-9.]+)px')]


def font_sizes(svg_path: str | Path) -> list[float]:
    """All font sizes in an SVG written by matplotlib (1 px == 1 pt at 72 dpi)."""
    txt = Path(svg_path).read_text()
    return sorted({float(v) for rx in _FS for v in rx.findall(txt)})


def check_fonts(svg_path: str | Path, minimum: float = MIN_FONT_PT) -> tuple[bool, list[float]]:
    sizes = font_sizes(svg_path)
    return (bool(sizes) and min(sizes) >= minimum), sizes
