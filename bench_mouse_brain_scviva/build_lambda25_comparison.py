#!/usr/bin/env python3
"""Fig. 2 comparison table: Haruka lambda=25 ablations (composition-gated, independent) + the four baselines.

Writes results_shared_full_niche_compgated_lambda25_comparison/benchmark_with_baselines_per_section.csv, read by
figures_paper/fig2_benchmark_stats.py.  Haruka ARIs: <root>_cc/.../direct_kmeans_ari.json (score_pair_direct_kmeans.py);
baselines: results_cross_section/summary_all_methods/per_section_metrics.csv (summarize_all_cross_section_methods.py).
    python build_lambda25_comparison.py [--check]
"""
import sys, json
import pandas as pd
from pathlib import Path

B = Path(__file__).resolve().parent
SECTIONS = ['A8', 'A9', 'B17', 'B3', 'B5']
HARUKA = {'Haruka λ=25 independent': 'results_shared_full_niche_lambda25_cc',
          'Haruka λ=25 composition-gated': 'results_shared_full_niche_compgated_lambda25_cc'}
BASELINES = ['Mini-batch CINEMA-OT + CellCharter', 'Expression + neighborhood aggregation',
             'scVI + CellCharter', 'ContrastiveVI z/s + CellCharter']
rows = []
for s in SECTIONS:
    for name, root in HARUKA.items():
        j = json.load(open(B / root / f'mouse_brain_{s}_paired/haruka_z_s_cellcharter_seed0/direct_kmeans_ari.json'))['scores']
        rows.append(dict(section=s, method=name, background_subregion_ari=j['background_subregion']['ari'],
                         salient_domain_ari=j['salient_domain']['ari']))
A = pd.read_csv(B / 'results_cross_section/summary_all_methods/per_section_metrics.csv')
A = A[A.method.isin(BASELINES)][['section', 'method', 'background_subregion_ari', 'salient_domain_ari']]
T = pd.concat([pd.DataFrame(rows), A], ignore_index=True)
out = B / 'results_shared_full_niche_compgated_lambda25_comparison/benchmark_with_baselines_per_section.csv'
if '--check' in sys.argv:
    k = ['section', 'method']; R = pd.read_csv(out).set_index(k).sort_index(); T = T.set_index(k).sort_index()
    print('same rows', R.index.equals(T.index), '| max abs diff', float((R - T).abs().max().max()))
else:
    out.parent.mkdir(exist_ok=True); T.to_csv(out, index=False); print('wrote', out, T.shape)
