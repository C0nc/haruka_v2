"""Notebook specs for Fig. 2 (benchmark), Fig. 3 (AD plaque), exercise and PEtracer; imported by build_notebooks.py."""
NOTEBOOKS = {}

# ---------------------------------------------------------------- exercise
FIT_EX = f'haruka_exercise/results_w10_s100/exercise_old_full/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
NOTEBOOKS['04_exercise'] = dict(
    title='Exercise rejuvenation (aged mouse brain MERFISH): s carries the clock shift, z does not',
    intro="""Old control (4 mice) vs old exercise (4 mice), 594,896 cells x 300 genes (Sun et al. aging/exercise atlas).
Ground truth: the published Spatial Aging Clock applied to observed expression (never seen by Haruka).
Out-of-fold ridge predictions of clock age from z or from s (train 3+3 mice, predict the held-out pair) per
cell type x brain region: s reproduces the observed exercise shift (r = +0.85 across strata, rejuvenation strongest in
vascular cells: VLMC, VSMC, endothelium), z does not (r = -0.17).""",
    steps=[
        ('md', '## 1 Data: keep old control / old exercise mice'),
        ('code', "run(f'{PY} bench_exercise/prepare_exercise.py', 'data')  # /home/yancui/virtualorganoid/exercise.h5ad -> exercise_old_full.h5ad"),
        ('md', '## 2 Fit (default model; batch = run; ~15 min on one GPU)'),
        ('code', f"""run(f'bash bench_exercise/run_fit_w10_s100.sh {{GPU}}', 'fit', log='bench_exercise/logs/fit_w10_s100.log')
require(WK / '{FIT_EX}/background.npy', WK / '{FIT_EX}/salient.npy')"""),
        ('md', '## 3 Spatial Aging Clock on observed expression (github sunericd/SpatialAgingClock @ 2b832f4)'),
        ('code', """CLK = 'bench_oskm_reprogramming/external/SpatialAgingClock'
run(f'[ -d {CLK} ] || (git clone -q https://github.com/sunericd/SpatialAgingClock {CLK} && git -C {CLK} checkout -q 2b832f4)', 'download')
run(f'{PY} bench_exercise/run_clock_exercise.py', 'analysis', log='bench_exercise/clock.log')"""),
        ('md', """## 4 Held-out clock prediction from z and s; within-vascular partition; panels
`make_exercise_clock_shift.py` reuses `oof_clock_predictions_w10_s100.csv.gz` if present: move it away for a true rerun."""),
        ('code', """E = H / 'figures_paper/exercise'
run(f'{PY} make_exercise_clock_shift.py', 'figure', cwd=E)
run(f'{PY} intra_vascular.py', 'analysis', cwd=E)
run(f'{PY} make_intra_panel.py', 'figure', cwd=E)
run(f'{PY} make_exercise_blocks.py', 'figure', cwd=E)
run(f'{PY} ../compose_montage.py exercise_clock && {PY} ../compose_montage.py exercise_blocks', 'figure', cwd=E)"""),
        ('code', """import pandas as pd
d = pd.read_csv(E / 'exercise_clock_shift_by_celltype_region.csv')
print('r(observed, z) = %.2f   r(observed, s) = %.2f' % (d.observed.corr(d.z), d.observed.corr(d.s)))"""),
    ],
    figures=['figures_paper/exercise/Exercise_clock_shift_full.png', 'figures_paper/exercise/Exercise_blocks_full.png'],
)

# ---------------------------------------------------------------- PEtracer
P = 'bench_petracer_lineage'
NOTEBOOKS['05_petracer'] = dict(
    title='PEtracer lineage tracing (M2-1): s recovers tumour clones the model never saw',
    intro="""Control = the section's non-tumour cells, condition = malignant cells inside tumours (halo cells are
neighbour-only). Lineage (clone, subclade, fitness from the PEtracer barcode trees) is never given to the model.
Held-out clone classification (multinomial logistic, GroupKFold over 150 um spatial blocks), balanced accuracy:

| test | chance | z | s | scVI |
|---|---|---|---|---|
| T1 clone across tumours | 0.17 | 0.41 | **0.65** | 0.63 |
| T2 clone 2-1 vs 2-2 within tumour 2 | 0.50 | 0.78 | **0.80** | 0.77 |
| T3 subclade within clone 5 | 0.07 | 0.11 | **0.14** | 0.12 |
| T4 fitness within clone 5 (R2) | 0 | 0.06 | **0.18** | 0.12 |""",
    steps=[
        ('md', '## 1 Data: tumour vs non-tumour design from the PEtracer release, then the lineage design'),
        ('code', f"""run(f'{{PY}} {P}/reaudit_2026-10-16/build_m2_tumour_vs_normal.py', 'data', log='{P}/reaudit_2026-10-16/build_m2.log')
run(f'{{PY}} build_m2_lineage.py', 'data', cwd=H / '{P}/m2_lineage')"""),
        ('md', '## 2 Haruka fit (~5 min) and the scVI baseline (~20 min)'),
        ('code', f"""run(f'bash run_fit.sh {{GPU}}', 'fit', cwd=H / '{P}/m2_lineage', log='fit.log')
run(f'{{PY}} fit_scvi.py', 'fit', cwd=H / '{P}/m2_lineage', log='scvi.log')"""),
        ('md', '## 3 Lineage tests T1-T5 (z vs s vs scVI)'),
        ('code', f"""run(f'{{PY}} {P}/m2_lineage/lineage_test.py', 'analysis', log='{P}/m2_lineage/lineage_test.out')
import pandas as pd; pd.read_csv(H / '{P}/m2_lineage/lineage_test_summary.csv').round(3)"""),
        ('md', '## 4 Panels and the page'),
        ('code', """T = H / 'figures_paper/petracer'
run(f'{PY} make_petracer_figure.py', 'figure', cwd=T)
run(f'{PY} ../compose_montage.py petracer', 'figure', cwd=T)"""),
    ],
    figures=['figures_paper/petracer/PEtracer_full.png'],
)

# ---------------------------------------------------------------- Fig. 2 benchmark
NOTEBOOKS['02_fig2_benchmark'] = dict(
    title='Fig. 2: benchmark on paired mouse-brain simulations with planted ground truth',
    intro="""Five coronal sections (A8, A9, B17, B3, B5) each paired with a copy carrying three planted salient domains
(composition shifts) in the disease arm. Readout identical for every method: 3-hop CellCharter aggregation of the saved
embedding, raw KMeans with k = number of planted classes. Means over 5 sections (background / salient ARI):
Haruka 0.540 / 0.854; CINEMA-OT + CC 0.510 / 0.816; expression + aggregation 0.339 / 0.815; scVI + CC 0.479 / 0.139;
ContrastiveVI + CC 0.474 / 0.022. With n = 5 no test can reach p < 0.0625, so section win counts are shown.
Note: the mouse-brain fits used a **Poisson** likelihood and batch = timepoint injected everywhere (pre-dates the
decoder-only batch default); `--gene_likelihood poisson` must now be passed explicitly.""",
    steps=[
        ('md', '## 1 Simulated paired sections'),
        ('code', """B = H / 'bench_mouse_brain_scviva'; SECTIONS = ['A8', 'A9', 'B17', 'B3', 'B5']
run(f'{PY} build_paired_sections.py --source /home/yancui/virtualorganoid/notebook/aging_coronal_spatial_mouse.h5ad --out_dir data_composition_pairs_all --seed 20260815 --n_domains 3 --shift_fraction 0.5 --sections ' + ' '.join(SECTIONS), 'data', cwd=B)"""),
        ('md', '## 2 Haruka fits: default λ = 10 (shown) and the λ = 25 gated / independent ablations (~3 min per section)'),
        ('code', """COMMON = ('--seed 0 --max_epochs 150 --batch_size 2048 --n_latent 32 --n_hidden 256 --n_layers 1 --decoder_layers 2 --k_niche 20 '
          '--neighbor_embedding_dim 16 --composition_loss_weight 1 --neighbor_mean_loss_weight 1 --spatial_weight 100 --direct_only '
          '--batch_key timepoint --sample_key slice_id --condition_key condition --target_label disease --cell_type_key observed_celltype '
          '--num_workers 4 --gene_likelihood poisson')
RUNS = {'results_shared_full_niche_compgated_lambda10': ('composition_gated', 10, 'haruka_shared_full_niche_compgated_seed0'),
        'results_shared_full_niche_compgated_lambda25': ('composition_gated', 25, 'haruka_shared_full_niche_compgated_seed0'),
        'results_shared_full_niche_lambda25': ('independent', 25, 'haruka_shared_full_niche_seed0')}
for root, (mode, lam, run_dir) in RUNS.items():
    for s in SECTIONS:
        d = f'data_composition_pairs_all/mouse_brain_{s}_paired.h5ad'
        run(f'{PY} run_haruka_shared_decoder.py --data {d} --out_root {root} --neighbor_expression_mode {mode} --zs_independence_penalty {lam} {COMMON}', 'fit', cwd=B)
        run(f'{PY} aggregate_haruka_latent_cellcharter.py --data {d} --input_dir {root}/mouse_brain_{s}_paired/{run_dir} --output_dir {root}_cc/mouse_brain_{s}_paired/haruka_z_s_cellcharter_seed0', 'analysis', cwd=B)
        run(f'{PY} score_pair_direct_kmeans.py --data {d} --emb_dir {root}_cc/mouse_brain_{s}_paired/haruka_z_s_cellcharter_seed0', 'analysis', cwd=B)
run(f'{PY} summarize_composition_gated_lambda_sweep.py --base . --lambdas 10 25 --sections ' + ' '.join(SECTIONS) + ' --seed 0 --max_cv_cells 10000 --out_dir results_shared_full_niche_compgated_lambda_sweep', 'analysis', cwd=B)"""),
        ('md', '## 3 Baselines: scVI / ContrastiveVI / expression + CellCharter, CINEMA-OT (env `vio`)'),
        ('code', """for s in SECTIONS:
    d = f'data_composition_pairs_all/mouse_brain_{s}_paired.h5ad'; root = 'results_direct_kmeans' if s == 'A8' else 'results_cross_section'
    run(f'{PY} {H}/bench_merfish/run_baselines_2x2.py --data {d} --methods pca cellcharter contrastivevi --out_root {root}/baselines --seed 0 --max_epochs 150 --batch_size 2048', 'fit', cwd=B)
    cine = f'{root}/cinema_minibatch/mouse_brain_{s}_paired/cinemaot_minibatch_b2048_e3_seed0'
    run(f'{PY_VIO} {H}/bench_merfish/run_cinemaot_minibatch.py --data {d} --out_root {root}/cinema_minibatch --seed 0 --dim 20 --thres 0.15 --transport_batch_size 2048 --n_ensembles 3', 'fit', cwd=B)
    run(f'{PY} {H}/bench_merfish/cc_add_spatial.py --data {d} --result_dir {cine}', 'analysis', cwd=B)
    run(f'{PY} score_composition_controls.py --data {d} --out {root}/composition_controls_{s}.json --k_niche 20 --seed 0 --n_init 20', 'analysis', cwd=B)
    run(f'for e in {root}/baselines/mouse_brain_{s}_paired/*/ {cine}; do {PY} score_pair_direct_kmeans.py --data {d} --emb_dir $e; done', 'analysis', cwd=B)
# per-section haruka/cinema result jsons are listed in results_cross_section/summary/per_section_metrics.csv (result_json)
run(f'{PY} summarize_all_cross_section_methods.py', 'analysis', cwd=B)
run(f'{PY} build_lambda25_comparison.py', 'analysis', cwd=B)"""),
        ('md', '## 4 Figure: per-section table, B5 maps cache (recomputed only if missing), panels'),
        ('code', """run(f'{PY} figures_paper/fig2_benchmark_stats.py', 'figure')
run(f'{PY} figures_paper/fig2_benchmark_maps.py --section B5', 'figure')
run(f'{PY} figures_paper/fig2/make_fig2.py', 'figure')
import pandas as pd
d = pd.read_csv(H / 'figures_paper/fig2_benchmark_per_section.csv'); d.groupby('method')[['bg', 'sal', 'balanced']].mean().round(3).sort_values('balanced', ascending=False)"""),
    ],
    figures=['figures_paper/fig2/fig2_composite.png'],
)

# ---------------------------------------------------------------- Fig. 3 AD plaque
SEP = 'results_real_compgated_lambda10_samplebatch_sepheads'
NOTEBOOKS['03_fig3_ad_plaque'] = dict(
    title='Fig. 3 and Extended Data Figs 3-4: s captures the plaque-associated microglial niche (STARmap + CODEX)',
    intro="""Mouse AD STARmap (2,766 genes; 13-month control vs disease) and human AD CODEX whole slices (34 proteins;
4 AD vs 3 control donors). Amyloid plaques are never given to the model; all readouts use disease-arm cells.
Held-out prediction of plaque proximity is stronger from s than from z or scVI near plaques; over 3 training seeds
rho(s) > rho(z) in 35/36 CODEX and 13/15 STARmap comparisons. The salient composition shift D([z,s]) - D([z,0])
reproduces the observed microglia enrichment toward plaques that the background prediction D([z,0]) misses, and the
s microglia axis separates DAM (Cst7, Apoe) from homeostatic (P2ry12) microglia. The plaque axis is conserved
across species (20 microglial protein-ortholog pairs: rho = +0.47 gradient, +0.54 s loading). Controls: spatially
shifted plaques, a control-vs-control arm swap (Ctl-3 relabelled as disease) and s-beyond-z tests (ED Fig. 3).""",
    steps=[
        ('md', '## 1 Data'),
        ('code', """run(f'{PY} bench_real_discovery/prepare_studies.py ad', 'data')                     # STARmap -> bench_real_discovery/data/ad.h5ad
W = H / 'bench_codex_cns_2026/wholeslice'
run(f'{PY} build_wholeslice.py', 'data', cwd=W)                                   # CODEX all7_no_oligo (+ full3)
run(f'{PY} build_ctl3_vs_ctl14.py', 'data', cwd=W)                               # arm-swap control design"""),
        ('md', '## 2 Fits (default model; normal likelihood on pretransformed input; ~12 min STARmap, ~3.5 min per CODEX design) + seed checks'),
        ('code', f"""run(f'{{PY}} bench_kras_xenium/run_haruka_separate_batch_heads.py --data bench_real_discovery/data/ad.h5ad --batch_size 4096 --sample_key slice_id --batch_key slice_id --condition_key condition --target_label disease --cell_type_key cell_type --gene_likelihood normal --neighbor_input_mode pretransformed --assay_type sequencing --n_top_genes 0 --out_root {SEP} --seed 0 --max_epochs 150 --n_latent 32 --n_hidden 256 --n_layers 1 --decoder_layers 2 --k_niche 20 --neighbor_embedding_dim 16 --composition_loss_weight 1 --neighbor_mean_loss_weight 1 --neighbor_expression_mode composition_gated --spatial_weight 100 --background_wasserstein_penalty 0 --zs_independence_penalty 10 --decoder_batch_dim 8 --batch_injection_mode decoder_only --augmentation_mode sparse --num_workers 4 --direct_only', 'fit')
run(f'{{PY}} bench_kras_xenium/rebuild_condition_only_salient_cc.py --data bench_real_discovery/data/ad.h5ad --input-dir {SEP}/ad/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0 --output-dir {SEP}/ad/haruka_z_s_cellcharter_cc3_condition_only --condition-key condition --condition-value disease --slice-key slice_id', 'analysis')
for v in ['all7_no_oligo', 'ctl3VsCtl14']:
    run(f'bash run_wholeslice_fit.sh {{v}} {{GPU}}', 'fit', cwd=W)
for ds in ['starmap', 'codex']:
    for seed in [1, 2]:
        run(f'bash results_seedcheck/run_seed.sh {{ds}} {{seed}} {{GPU}}', 'fit')"""),
        ('md', '## 3 Plaque distances (model-independent) and CODEX radial / null analyses'),
        ('code', """# STARmap plaque distances: ad_analysis/plot_plaque_distance_spatial.py -> final_results_nb/starmap_ad/biology/plaque_distance/starmap_plaque_distance_cells.csv.gz
require(H / 'final_results_nb/starmap_ad/biology/plaque_distance/starmap_plaque_distance_cells.csv.gz')
PS = W / 'plaque_states'; SEED = H / 'results_seedcheck/codex/codex_wholeslice_all7_no_oligo/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed'
run(f'{PY} build_plaque_niches.py', 'analysis', cwd=PS)
run(f'{PY} radial_generic.py all7_no_oligo && {PY} radial_generic.py all7_no_oligo all && {PY} radial_generic.py ctl3VsCtl14', 'analysis', cwd=PS)
for k in [1, 2]:
    run(f'{PY} radial_generic.py all7_no_oligo', 'analysis', cwd=PS, env={'MODEL_DIR': f'{SEED}{k}', 'TAG': f'_seed{k}'})
run(f'{PY} radial_null.py all7_no_oligo', 'analysis', cwd=PS)"""),
        ('md', '## 4 STARmap shifted-plaque null (with scVI), s beyond z, counterfactual proportions, cross-species markers, domains'),
        ('code', f"""SR = H / 'bench_real_discovery/starmap_plaque_radial'
SSEED = H / 'results_seedcheck/starmap/ad/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed'
run(f'{{PY}} fit_scvi.py', 'fit', cwd=SR)
run(f'{{PY}} starmap_null.py', 'analysis', cwd=SR)
for k in [1, 2]:
    run(f'{{PY}} starmap_null.py', 'analysis', cwd=SR, env={{'MODEL_DIR': f'{{SSEED}}{{k}}', 'TAG': f'_seed{{k}}'}})
run(f'{{PY}} s_beyond_z.py starmap && {{PY}} s_beyond_z.py codex', 'analysis', cwd=SR)
run(f'{{PY}} bench_real_discovery/cc_shift_info/check_cf_proportion.py starmap && {{PY}} bench_real_discovery/cc_shift_info/check_cf_proportion.py codex', 'analysis')
run(f'{{PY}} marker_axes.py starmap && {{PY}} marker_axes.py codex && {{PY}} compare_species.py', 'analysis', cwd=PS / 'cross_species')
run(f'{{PY}} bench_real_discovery/analyze_domain_celltype_enrichment.py --data bench_real_discovery/data/ad.h5ad --cc-dir {SEP}/ad/haruka_z_s_cellcharter_cc3_condition_only --out-dir final_results_nb/starmap_ad/biology/domain_annotation --sample-key slice_id --cell-type-key cell_type --seed 0', 'analysis')"""),
        ('md', '## 5 Panels and pages (letters from `fig3_layout.py`)'),
        ('code', """F3 = H / 'figures_paper/fig3'
run(f'{PY} haruka_domain_plaque_stats.py', 'analysis', cwd=F3)
for s in ['make_fig3_haruka_blocks.py', 'make_fig3_plaque.py', 'make_fig3_cf_proportion.py', 'make_fig3_markers.py', 'make_fig3_full.py']:
    run(f'{PY} {s}', 'figure', cwd=F3)"""),
    ],
    figures=['figures_paper/fig3/final/Fig3_full.png', 'figures_paper/fig3/final/EDFig3_full.png', 'figures_paper/fig3/final/EDFig4_full.png'],
)
