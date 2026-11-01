#!/usr/bin/env python3
"""Generate the reproduction notebooks (one per paper figure) from the step lists below.

Each notebook: what the figure claims, the default Haruka fit, then every step as a `run(cmd, stage)` cell
(stages: download, data, fit, analysis, figure), and finally the figure itself. Inputs come from the Hugging Face
dataset (section 0), so a notebook runs end to end on a fresh clone. Regenerate with
    python notebooks/build_notebooks.py
"""
import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).resolve().parent
KERNEL = {'name': 'haruka2', 'display_name': 'Python (haruka2)', 'language': 'python'}

DEFAULT_ARGS = """--seed 0 --max_epochs 150 \\
  --n_latent 32 --n_hidden 256 --n_layers 1 --decoder_layers 2 --k_niche 20 --neighbor_embedding_dim 16 \\
  --composition_loss_weight 1 --neighbor_mean_loss_weight 1 --neighbor_expression_mode composition_gated --spatial_weight 100 \\
  --background_wasserstein_penalty 0 --zs_independence_penalty 10 --decoder_batch_dim 8 --batch_injection_mode decoder_only \\
  --augmentation_mode sparse --num_workers 4 --direct_only"""

HEADER = """import sys, pathlib; sys.path.insert(0, str(next(p for p in [pathlib.Path.cwd(), pathlib.Path.cwd() / 'notebooks'] if (p / 'repro.py').exists())))
from repro import run, require, show, fetch, fetch_fitted, STAGES, H, WK, PY, PY_VIO, GPU
# Default: download inputs + the paper's fitted Haruka outputs from Hugging Face, then analyses -> figure.
# The Haruka fit step reuses the unpacked fit; HARUKA_REFIT=1 (env) refits from scratch. Auxiliary fits (baselines, seeds) still run.
# Redraw only from saved outputs: STAGES.update(download=False, fit=False, analysis=False)  (or env HARUKA_FIGURE_ONLY=1)
# Rebuild the inputs from the raw public releases instead of Hugging Face: STAGES.update(data=True)
print(H, STAGES)"""


def md(s): return nbf.v4.new_markdown_cell(s.strip())
def code(s): return nbf.v4.new_code_cell(s.strip())


def notebook(name, title, intro, steps, figures):
    nb = nbf.v4.new_notebook(); nb.metadata['kernelspec'] = KERNEL
    cells = [md(f'# {title}\n\n{intro}'), code(HEADER)]
    if name in HF_FILES:
        cells.append(md('## 0 Preprocessed data (Hugging Face)\n\nThe ready-to-fit inputs are read directly from '
                        f'[{HF_REPO}](https://huggingface.co/datasets/{HF_REPO}) (cached in `hf_data/`) and linked to the '
                        'paths the scripts read. The data stage below rebuilds the same files from the raw releases and is off by default.'))
        cells.append(code('\n'.join(f"fetch('{f}', {dst})" for f, dst in HF_FILES[name])))
        cells.append(md('The paper\'s default Haruka fit, published as fitted AnnData (input + z, s, predicted neighbourhoods; '
                        'see the dataset card). It is unpacked into the fit folder, so the fit step below reuses it. '
                        'Set `HARUKA_REFIT=1` before launching to refit instead.'))
        cells.append(code('\n'.join(f"fetch_fitted('{f}', {dst})" for f, dst in HF_FITTED[name])))
    for kind, text in steps:
        cells.append(md(text) if kind == 'md' else code(text))
    cells.append(md('## Figure'))
    cells += [code(f"show(H / '{f}')") for f in figures]
    nb.cells = cells
    nbf.write(nb, OUT / f'{name}.ipynb')
    print('wrote', name)


HF_REPO = 'C1vy/haruka_v2_data'
_FIG2 = [(f'02_fig2_benchmark/mouse_brain_{s}_paired.h5ad', f"H / 'bench_mouse_brain_scviva/data_composition_pairs_all/mouse_brain_{s}_paired.h5ad'") for s in ['A8', 'A9', 'B17', 'B3', 'B5']]
HF_FILES = {
    '02_fig2_benchmark': _FIG2,
    '03_fig3_ad_plaque': [('03_fig3_ad_plaque/ad_starmap.h5ad', "H / 'bench_real_discovery/data/ad.h5ad'"),
                          ('03_fig3_ad_plaque/codex_wholeslice_all7_no_oligo.h5ad', "H / 'bench_codex_cns_2026/wholeslice/codex_wholeslice_all7_no_oligo.h5ad'"),
                          ('03_fig3_ad_plaque/codex_wholeslice_ctl3VsCtl14.h5ad', "H / 'bench_codex_cns_2026/wholeslice/codex_wholeslice_ctl3VsCtl14.h5ad'")],
    '04_exercise': [('04_exercise/exercise_old_full.h5ad', "WK / 'haruka_exercise/data/exercise_old_full.h5ad'")],
    '05_petracer': [('05_petracer/petracer_M2-1_lineage.h5ad', "H / 'bench_petracer_lineage/m2_lineage/petracer_M2-1_lineage.h5ad'")],
    '06_fibrosis': [('06_fibrosis/gse250346_unaffected_vs_all_fibrosis.h5ad', "H / 'bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad'"),
                    ('06_fibrosis/he_annot_matched.csv.gz', "WK / 'haruka_fibrosis_he/he_annot_matched.csv.gz'")],
    '07_kras': [('07_kras/mouselung_kras_xenium_clean_noC3.h5ad', "H / 'data/mouselung_kras_xenium_clean_noC3.h5ad'")],
}

RUN = 'haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
HF_FITTED = {
    '02_fig2_benchmark': [(f'fitted/02_fig2_benchmark/mouse_brain_{s}_paired_haruka_fitted.h5ad',
                           f"H / 'bench_mouse_brain_scviva/results_shared_full_niche_compgated_lambda10/mouse_brain_{s}_paired/haruka_shared_full_niche_compgated_seed0'")
                          for s in ['A8', 'A9', 'B17', 'B3', 'B5']],
    '03_fig3_ad_plaque': [('fitted/03_fig3_ad_plaque/ad_haruka_fitted.h5ad', f"H / 'results_real_compgated_lambda10_samplebatch_sepheads/ad/{RUN}'"),
                          ('fitted/03_fig3_ad_plaque/codex_wholeslice_all7_no_oligo_haruka_fitted.h5ad', f"H / 'results_codex_wholeslice/codex_wholeslice_all7_no_oligo/{RUN}'"),
                          ('fitted/03_fig3_ad_plaque/codex_wholeslice_ctl3VsCtl14_haruka_fitted.h5ad', f"H / 'results_codex_wholeslice/codex_wholeslice_ctl3VsCtl14/{RUN}'")],
    '04_exercise': [('fitted/04_exercise/exercise_old_full_haruka_fitted.h5ad', f"WK / 'haruka_exercise/results_w10_s100/exercise_old_full/{RUN}'")],
    '05_petracer': [('fitted/05_petracer/petracer_M2-1_lineage_haruka_fitted.h5ad', f"H / 'bench_petracer_lineage/m2_lineage/fits/petracer_M2-1_lineage/{RUN}'")],
    '06_fibrosis': [('fitted/06_fibrosis/gse250346_unaffected_vs_all_fibrosis_haruka_fitted.h5ad', f"H / 'results_real_compgated_nb_final/gse250346_unaffected_vs_all_fibrosis/{RUN}'")],
    '07_kras': [('fitted/07_kras/mouselung_kras_xenium_clean_noC3_haruka_fitted.h5ad', f"H / 'results_kras_te_null/mouselung_kras_xenium_clean_noC3/{RUN}'")],
}

NOTEBOOKS = {}

# ---------------------------------------------------------------- 00 default model
NOTEBOOKS['00_default_model'] = dict(
    title='Haruka v2: the default model and how every figure was fitted',
    intro="""Every figure in the paper uses one model configuration: the **composition-gated shared-decoder Haruka**
(`bench_kras_xenium/run_haruka_separate_batch_heads.py`, which wraps `bench_mouse_brain_scviva/run_haruka_shared_decoder.py`).

* **z (background)**: per-cell latent shared by control and condition cells.
* **s (salient)**: condition-specific latent. The decoder sets s = 0 for control cells, so control s is never used
  and every z-vs-s readout is computed on condition-arm cells only.
* **Decoders**: a negative-binomial expression decoder plus two niche heads, the cell-type composition of the k = 20
  spatial neighbours and the composition-gated neighbour mean expression (`spatial_weight 100`).
* **Salient composition shift**: D([z, s]) − D([z, 0]), what s adds to the predicted neighbourhood.
* **z/s independence**: one-way ridge penalty (z ← s, weight 10).
* **Batch**: injected into the decoders only (`decoder_batch_dim 8`); separate batch projectors for the
  composition and neighbour-mean heads. z and s encoders are batch-blind.
* **Training**: 150 epochs, 32-d latents, sparse augmentation.""",
    steps=[
        ('md', """## Setup (once per machine)

```bash
git clone -b v2 https://github.com/C0nc/haruka_v2 && cd haruka_v2
conda create -n haruka2 python=3.11 -y && conda activate haruka2
pip install -r requirements.txt
bash install_contrastivevi_patch.sh            # the Haruka model: patched scvi ContrastiveVI module
python -m ipykernel install --user --name haruka2
```
Then open any notebook: section 0 downloads its inputs from Hugging Face and the notebook runs end to end.
The check below confirms the installed scvi carries the patch."""),
        ('code', """import os, filecmp, scvi
dst = os.path.join(os.path.dirname(scvi.__file__), 'external', 'contrastivevi')
bad = [f for f in os.listdir(H / 'contrastivevi_patch') if f.endswith('.py') and not filecmp.cmp(H / 'contrastivevi_patch' / f, os.path.join(dst, f), shallow=False)]
print('scvi', scvi.__version__, '| patch installed' if not bad else f'| NOT patched: {bad} -> run install_contrastivevi_patch.sh')"""),
        ('md', '## Default arguments (shared by all fits)'),
        ('code', f'DEFAULT = """{DEFAULT_ARGS}"""\nprint(DEFAULT)'),
        ('md', """## Per-dataset arguments

| figure | data | batch key | condition (target) | cell type key | likelihood / HVG | extra |
|---|---|---|---|---|---|---|
| Fig. 2 benchmark | paired mouse-brain simulation (5 section pairs) | slice | disease | cell type | NB | λ sweep, λ = 10 shown |
| Fig. 3 AD plaque | STARmap AD, CODEX CNS | slice / donor | disease | cell type | Normal, pretransformed | |
| Exercise | old control vs old exercise MERFISH | run | exercise | cell type | NB | |
| PEtracer | M2-1 lineage section | graph | malignant cells in tumours | cell type | NB | clone never seen |
| Lung fibrosis | GSE250346 Xenium, 1.63 M cells, 343 genes | core (`sample`) | fibrosis | cell_type | NB, all genes | |
| KRAS inhibitor | Xenium 5K, 920 k cells | slice | KRASi_treated | celltype | NB, 2000 HVG | 3 shuffled-s nulls |

The exact command for each is in that figure's notebook."""),
        ('md', '## Batch runner used for the real-data fits'),
        ('code', "print(open(H / 'bench_real_discovery/run_all_compgated_samplebatch_sepheads.sh').read())"),
        ('md', """## Outputs of a fit (`<out_root>/<data stem>/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0/`)

`background.npy` (z), `salient.npy` (s), `obs_names.npy`, `background_composition.npy` D([z,0]),
`condition_composition.npy` D([z,s]), `salient_composition_shift.npy`, `meta.json` (all arguments + timings),
optionally `model/` and `null_shuffled_s_composition_shift_{k}.npy`."""),
        ('code', """import json
m = json.load(open(H / 'notebooks/kras_default_fit_meta.json'))   # copy of the KRAS fit's meta.json
{k: m[k] for k in list(m)[:40] if not isinstance(m[k], (list, dict))}"""),
    ],
    figures=[],
)

# ---------------------------------------------------------------- fibrosis
FIB = 'bench_lung_fibrosis_gse250346'
FIT_FIB = 'results_real_compgated_nb_final/gse250346_unaffected_vs_all_fibrosis/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
NOTEBOOKS['06_fibrosis'] = dict(
    title='Lung fibrosis (GSE250346): s carries lesion context and the gradient toward fibrotic lesions',
    intro="""Unaffected vs fibrotic lung cores (Xenium, 343 genes). Ground truth for intra-fibrosis heterogeneity is the
pathologist H&E region annotation released with GSE250346. All z/s comparisons use fibrosis-arm cells only.

Main results: within fibrotic cores s separates H&E regions better than z in 75/109 region pairs; along the distance
to the nearest lesion, s predicts proximity better than z in 94/139 (supervised) cell type × core strata. Strongest
case: alveolar fibroblasts approaching fibroblastic foci (spatial-shift null p = 0.005; collagen/POSTN/CTHRC1 up).
Disease stage (Less vs More Affected) is carried by z, not s.""",
    steps=[
        ('md', '## 1 Data: download, export from the Seurat RDS (R), build the h5ad'),
        ('code', f"""run(f'wget -c -O {FIB}/data/GSE250346_corrected.rds https://ftp.ncbi.nlm.nih.gov/geo/series/GSE250nnn/GSE250346/suppl/GSE250346_Seurat_GSE250346_CORRECTED_SEE_RDS_README_082024.rds', 'data')
run(f'Rscript {FIB}/export_gse250346_unaffected_vs_fibrosis.R {FIB}/data/GSE250346_corrected.rds {FIB}/data/export_unaffected_vs_fibrosis', 'data')
run(f'{{PY}} {FIB}/build_h5ad.py --export-dir {FIB}/data/export_unaffected_vs_fibrosis --out {FIB}/data/gse250346_unaffected_vs_all_fibrosis.h5ad --contrast unaffected-vs-fibrosis', 'data')"""),
        ('md', '## 2 H&E annotations (GEO supplementary) matched to cells by barcode + centroid'),
        ('code', """HE = WK / 'haruka_fibrosis_he'
run(f'mkdir -p {HE}/raw {HE}/x && cd {HE}/raw && curl -sSLO https://ftp.ncbi.nlm.nih.gov/geo/series/GSE250nnn/GSE250346/suppl/GSE250346_HE_annotations.tar.gz && tar -xzf GSE250346_HE_annotations.tar.gz -C ../x', 'data')
run(f'{PY} bench_lung_fibrosis_gse250346/match_he_annotations.py', 'data')"""),
        ('md', '## 3 Fit (default model; batch = core; ~6.8 h on one GPU)'),
        ('code', "run('bash bench_real_discovery/run_all_compgated_samplebatch_sepheads.sh large_lung', 'fit')"),
        ('code', f"require(H / '{FIT_FIB}/background.npy', H / '{FIT_FIB}/salient.npy', WK / 'haruka_fibrosis_he/he_annot_matched.csv.gz')"),
        ('md', """## 4 Analyses (`zs_stage/`)
* `he_region_zs.py`: per H&E region pair, separability by z vs s (panel e)
* `he_region_spatial.py`: the same against a spatial-shift null (panel f)
* `lesion_gradient.py`: supervised proximity-to-lesion gradient, z vs s per cell type × core (panel g)
* `lesion_gradient_unsup.py`: unsupervised PC1/PC2/DPT axes vs lesion distance (panel h)
* `lesion_cases.py`: deep dive on the strongest cases, genes and spatial-shift nulls (panels j, k)"""),
        ('code', """Z = 'bench_lung_fibrosis_gse250346/zs_stage'
run(f'{PY} he_region_zs.py', 'analysis', cwd=H / Z, log='he_region/run.out')
run(f'{PY} he_region_spatial.py', 'analysis', cwd=H / Z, log='he_region/spatial.out')
run(f'{PY} lesion_gradient.py', 'analysis', cwd=H / Z, log='lesion_gradient/run.out')
run(f'{PY} lesion_gradient_unsup.py', 'analysis', cwd=H / Z, log='lesion_gradient/unsup.out')
run(f'mkdir -p lesion_gradient/cases && {PY} lesion_cases.py', 'analysis', cwd=H / Z, log='lesion_gradient/cases/run.out')"""),
        ('md', '## 5 Figure data (example core TILD117MA1, global UMAPs) and the figure'),
        ('code', """F = H / 'figures_paper/fibrosis'
run(f'{PY} fib_fig_data.py', 'analysis', cwd=F, env={'EXAMPLE_CORE': 'TILD117MA1'})
run(f'{PY} fib_umap_data.py', 'analysis', cwd=F)
run(f'{PY} make_fibrosis_figure.py', 'figure', cwd=F)"""),
    ],
    figures=['figures_paper/fibrosis/Fibrosis_lesion_gradient_full.png'],
)

# ---------------------------------------------------------------- KRAS
KR = 'bench_kras_xenium'
FIT_KR = 'results_kras_te_null/mouselung_kras_xenium_clean_noC3/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
NOTEBOOKS['07_kras'] = dict(
    title='KRAS inhibitor (MRTX1133) mouse lung Xenium: treatment effect from s, outcome predictor from z',
    intro="""3 control and 4 treated sections, 920 k cells. The per-nodule treatment effect (TE) is the salient composition
shift D([z,s]) − D([z,0]) averaged over a nodule's tumour cells; its main axis (94 %) is conversion toward
Itgb6+ persister tumour epithelium, and nodules with more conversion respond less (independent MAPK MPAS responder
rate, never seen by Haruka).

To find the background that goes with a bad TE, the TE is made s-specific (real − mean of 3 shuffled-s nulls).
Background axis **z-PC2** (alveolar-parenchymal niche: Nqo1/Ptgs2/Ager/Sdc1; endothelium, AT1/AT2, monocytes and
alveolar macrophages vs airway-adjacent ciliated/SMC) predicts the bad TE on held-out sections (3/4; +0.53 vs
responders −0.39) and exists before treatment (control SD ratio 0.94). Interactions: squidpy neighbourhood
enrichment and LIANA+ (lipid exchange Lpl–Sdc1, Apob–Lrp1 at the bad end; complement C3 at the good end).""",
    steps=[
        ('md', '## 1 Data: the KRAS Xenium h5ad is distributed as-is (X = normalize+log1p, raw counts in `layers[\"counts\"]`), fetched in section 0'),
        ('code', "require(H / 'data/mouselung_kras_xenium_clean_noC3.h5ad')"),
        ('md', '## 2 Fit (default model; batch = slice, 2000 HVG, saves the model + 3 shuffled-s nulls; ~2.8 h on one GPU)'),
        ('code', f"""run(f'mkdir -p results_kras_te_null && bash {KR}/run_kras_te_null_fit.sh {{GPU}}', 'fit', log='results_kras_te_null/fit.log')
require(H / '{FIT_KR}/salient_composition_shift.npy', H / '{FIT_KR}/null_shuffled_s_composition_shift_2.npy')"""),
        ('md', '## 3 Independent biological responder (MAPK MPAS, lower quartile of control per tumour subtype)'),
        ('code', f"run(f'{{PY}} {KR}/define_biological_responder.py', 'analysis', log='{KR}/bio_responder.log')"),
        ('md', '## 4 Nodule treatment effect (panels i–l)'),
        ('code', f"""M = H / '{FIT_KR}'; N = ' '.join(str(M / f'null_shuffled_s_composition_shift_{{k}}.npy') for k in range(3))
NT = H / '{KR}/nodule_te'
run(f'{{PY}} nodule_te.py {{M}} refit {{N}}', 'analysis', cwd=NT, log='run_refit.out')
run(f'{{PY}} nodule_te.py {{M}} refit_ownTumEpi {{N}}', 'analysis', cwd=NT, env={{'OWN_TYPE': 'Tumor_epithelial'}}, log='run_refit_own.out')"""),
        ('md', '## 5 Which background goes with a bad TE; naming z-PC2; held-out prediction; interactions (panels m–v)'),
        ('code', """run(f'{PY} bad_te_background.py', 'analysis', cwd=NT)
run(f'{PY} name_bad_te_z.py', 'analysis', cwd=NT)
run(f'timeout 5400 {PY} zpc2_interactions.py', 'analysis', cwd=NT)"""),
        ('code', """import pandas as pd
display(pd.read_csv(NT / 'name_bad_te_heldout.csv').round(2))
display(pd.read_csv(NT / 'zpc2_interactions/A_summary.csv').round(3))"""),
        ('md', '## 6 Figure data and the figure'),
        ('code', """K = H / 'figures_paper/kras'
run(f'{PY} kras_fig_data.py', 'analysis', cwd=K)
run(f'{PY} make_kras_figure.py', 'figure', cwd=K)"""),
    ],
    figures=['figures_paper/kras/KRAS_inhibitor_full.png'],
)

# Figures 2, 3, exercise and PEtracer are appended by build_notebooks_more.py
try:
    import build_notebooks_more  # noqa: F401
    NOTEBOOKS.update(build_notebooks_more.NOTEBOOKS)
except ImportError:
    pass

if __name__ == '__main__':
    for name, spec in NOTEBOOKS.items():
        notebook(name, **spec)
