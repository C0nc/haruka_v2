#!/usr/bin/env python3
"""Published Spatial Aging Clock (Sun et al.) on OBSERVED exercise expression, via the same runner used for OSKM.
Only adds the obs column the runner expects (paired_batch := run_batch)."""
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
_HW = _os.environ.get('HARUKA_WORK') or ('/project/zhihuanglab/yancui' if _os.path.isdir('/project/zhihuanglab/yancui') else _HR + '/work')
import sys, anndata as ad
sys.path.insert(0, f'{_HR}/final_results_agent_oskm/spatial_aging_clock_extension/scripts')
import run_official_clock_parallel as clk
_read = ad.read_h5ad
def read(p, *a, **k):
    d = _read(p, *a, **k); d.obs['paired_batch'] = d.obs['run_batch'].astype(str); return d
clk.ad.read_h5ad = read
sys.argv = ['x', '--data', f'{_HW}/haruka_exercise/data/exercise_old_full.h5ad',
            '--clock-repo', f'{_HR}/bench_oskm_reprogramming/external/SpatialAgingClock',
            '--out', f'{_HW}/haruka_exercise/clock', '--workers', '24']
clk.main()
