#!/usr/bin/env python3
"""Published Spatial Aging Clock (Sun et al.) on OBSERVED exercise expression, via the same runner used for OSKM.
Only adds the obs column the runner expects (paired_batch := run_batch)."""
import sys, anndata as ad
sys.path.insert(0, '/home/yancui/haruka_v2/final_results_agent_oskm/spatial_aging_clock_extension/scripts')
import run_official_clock_parallel as clk
_read = ad.read_h5ad
def read(p, *a, **k):
    d = _read(p, *a, **k); d.obs['paired_batch'] = d.obs['run_batch'].astype(str); return d
clk.ad.read_h5ad = read
sys.argv = ['x', '--data', '/project/zhihuanglab/yancui/haruka_exercise/data/exercise_old_full.h5ad',
            '--clock-repo', '/home/yancui/haruka_v2/bench_oskm_reprogramming/external/SpatialAgingClock',
            '--out', '/project/zhihuanglab/yancui/haruka_exercise/clock', '--workers', '24']
clk.main()
