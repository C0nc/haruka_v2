#!/usr/bin/env python3
"""Export each figure's input h5ad together with its default Haruka fit, for further analysis.

Adds to the input AnnData (cells matched by obs_names):
  obsm['X_haruka_z']             background latent z (32-d, shared by control and condition cells)
  obsm['X_haruka_s']             salient latent s (32-d); for control cells this is the unused encoder output,
                                 the decoder sets s = 0 there, so analyse s on condition cells only
  obsm['haruka_D_z0']            predicted neighbour cell-type composition from [z, 0] (background)
  obsm['haruka_D_zs']            predicted neighbour composition from [z, s]
  obsm['haruka_salient_shift']   D([z,s]) - D([z,0]), the per-cell salient composition shift
  obsm['haruka_null_shift_k']    the same with s shuffled across cells (k = 0..2), where the fit saved nulls
  obsm['haruka_M_z0'], obsm['haruka_M_shift']
                                 Fig. 2 / Fig. 3 only: predicted per-type neighbour mean expression (16-d SVD per
                                 cell type) from [z, 0], and its salient shift; read by the CellCharter readouts
  obs['haruka_condition_arm']    True for condition-arm cells
  obs['haruka_in_fit']           False for cells the fit did not include (their obsm rows are NaN)
  uns['haruka']                  composition column names (cell types), fit arguments (meta.json), notes
    python export_fitted_h5ad.py <key> [--out DIR]     keys: see DATASETS
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repro import H, WK

RUN = 'haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
B = 'bench_mouse_brain_scviva'
DATASETS = {
    **{f'fig2_{s}': ('02_fig2_benchmark', H / f'{B}/data_composition_pairs_all/mouse_brain_{s}_paired.h5ad',
                     H / f'{B}/results_shared_full_niche_compgated_lambda10/mouse_brain_{s}_paired/haruka_shared_full_niche_compgated_seed0',
                     'condition', 'disease') for s in ['A8', 'A9', 'B17', 'B3', 'B5']},
    'ad_starmap': ('03_fig3_ad_plaque', H / 'bench_real_discovery/data/ad.h5ad',
                   H / f'results_real_compgated_lambda10_samplebatch_sepheads/ad/{RUN}', 'condition', 'disease'),
    'codex_all7': ('03_fig3_ad_plaque', H / 'bench_codex_cns_2026/wholeslice/codex_wholeslice_all7_no_oligo.h5ad',
                   H / f'results_codex_wholeslice/codex_wholeslice_all7_no_oligo/{RUN}', 'condition', 'disease'),
    'codex_ctl3': ('03_fig3_ad_plaque', H / 'bench_codex_cns_2026/wholeslice/codex_wholeslice_ctl3VsCtl14.h5ad',
                   H / f'results_codex_wholeslice/codex_wholeslice_ctl3VsCtl14/{RUN}', 'condition', 'disease'),
    'exercise': ('04_exercise', WK / 'haruka_exercise/data/exercise_old_full.h5ad',
                 WK / f'haruka_exercise/results_w10_s100/exercise_old_full/{RUN}', 'condition', 'exercise'),
    'petracer': ('05_petracer', H / 'bench_petracer_lineage/m2_lineage/petracer_M2-1_lineage.h5ad',
                 H / f'bench_petracer_lineage/m2_lineage/fits/petracer_M2-1_lineage/{RUN}', 'condition', 'disease'),
    'fibrosis': ('06_fibrosis', H / 'bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad',
                 H / f'results_real_compgated_nb_final/gse250346_unaffected_vs_all_fibrosis/{RUN}', 'condition', 'fibrosis'),
    'kras': ('07_kras', H / 'data/mouselung_kras_xenium_clean_noC3.h5ad',
             H / f'results_kras_te_null/mouselung_kras_xenium_clean_noC3/{RUN}', 'condition_label', 'KRASi_treated'),
}
ARRAYS = {'X_haruka_z': 'background', 'X_haruka_s': 'salient', 'haruka_D_z0': 'background_composition',
          'haruka_D_zs': 'condition_composition', 'haruka_salient_shift': 'salient_composition_shift'}

p = argparse.ArgumentParser(); p.add_argument('key', choices=list(DATASETS)); p.add_argument('--out', default=str(WK / 'haruka_v2_fitted'))
a = p.parse_args()
folder, src, fit, ckey, target = DATASETS[a.key]
A = ad.read_h5ad(src)
names = np.load(fit / 'obs_names.npy', allow_pickle=True).astype(str)
idx = A.obs_names.get_indexer(names)
assert (idx >= 0).all(), f'{(idx < 0).sum()} fitted cells not in the input h5ad'
assert len(np.unique(idx)) == len(idx), 'duplicate obs_names'
inside = np.zeros(A.n_obs, bool); inside[idx] = True
meta = json.load(open(fit / 'meta.json'))
cols = [str(c) for c in meta['cell_type_categories']]
null_files = sorted(fit.glob('null_shuffled_s_composition_shift_*.npy'))
EXTRA = {'haruka_M_z0': 'background_neighbor_mean', 'haruka_M_shift': 'salient_neighbor_mean_shift'} if a.key.startswith(('fig2', 'ad_', 'codex')) else {}
for key, f in {**ARRAYS, **EXTRA, **{f'haruka_null_shift_{q.stem.rsplit("_", 1)[1]}': q.stem for q in null_files}}.items():
    v = np.load(fit / f'{f}.npy'); full = np.full((A.n_obs, v.shape[1]), np.nan, np.float32); full[idx] = v
    A.obsm[key] = full
assert A.obsm['haruka_D_zs'].shape[1] == len(cols)
A.obs['haruka_condition_arm'] = (A.obs[ckey].astype(str) == target).to_numpy()
A.obs['haruka_in_fit'] = inside
A.uns['haruka'] = {
    'composition_columns': cols, 'condition_key': ckey, 'condition_value': target,
    'fit_arguments_json': json.dumps(meta, default=str),
    'notes': 'Default Haruka v2 fit used in the paper figure. z/s comparisons use condition-arm cells only: the decoder '
             'sets s = 0 for control cells. haruka_D_* columns follow composition_columns (k = 20 spatial neighbours).'}
out = Path(a.out) / folder / f'{src.stem}_haruka_fitted.h5ad'; out.parent.mkdir(parents=True, exist_ok=True)
A.write_h5ad(out, compression='gzip')
print(a.key, A.shape, 'fitted cells', int(inside.sum()), 'condition', int(A.obs.haruka_condition_arm.sum()),
      'obsm', list(A.obsm), '->', out, f'{out.stat().st_size / 1e6:.0f} MB')
