"""Shared helpers for the Haruka v2 reproduction notebooks.

Every step is a shell command gated by a stage. By default a notebook runs end to end: download the preprocessed
inputs from Hugging Face, fit Haruka, run the analyses, draw the figure (hours on one GPU). `HARUKA_FIGURE_ONLY=1`
redraws the figure from saved analysis outputs; `RUN_<STAGE>=0/1` switches single stages.

Paths: HARUKA_ROOT (default: this repository), HARUKA_WORK (large fit outputs for the exercise and fibrosis H&E
steps; default /project/zhihuanglab/yancui on the development machine, else <root>/work), HARUKA_PY (python of the
haruka2 environment; default: the notebook kernel), HARUKA_GPU (GPU id, default 0), HARUKA_PY_VIO (python with cinemaot, Fig. 2 baseline only).
"""
import os, sys, subprocess, time
from pathlib import Path

H = Path(os.environ.get('HARUKA_ROOT') or Path(__file__).resolve().parent.parent)
WK = Path(os.environ.get('HARUKA_WORK') or ('/project/zhihuanglab/yancui' if os.path.isdir('/project/zhihuanglab/yancui') else H / 'work'))
PY = os.environ.get('HARUKA_PY', sys.executable)
PY_VIO = os.environ.get('HARUKA_PY_VIO', '/home/yancui/.conda/envs/vio/bin/python' if os.path.exists('/home/yancui/.conda/envs/vio/bin/python') else 'python')
HF_REPO = 'C1vy/haruka_v2_data'
GPU = os.environ.get('HARUKA_GPU', os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])   # one GPU id for all fits
ENV = {**os.environ, 'NUMPY_MADVISE_HUGEPAGE': '0', 'OPENBLAS_NUM_THREADS': '8', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
       'HARUKA_ROOT': str(H), 'HARUKA_WORK': str(WK), 'HARUKA_PY': PY,
       'CUDA_VISIBLE_DEVICES': GPU}

# Stage switches. data = rebuild the inputs from the raw public releases (off: the Hugging Face copies replace it).
_ON = {'download': '1', 'data': '0', 'fit': '1', 'analysis': '1', 'figure': '1'}
if os.environ.get('HARUKA_FIGURE_ONLY') == '1':
    _ON = {**{k: '0' for k in _ON}, 'figure': '1'}
STAGES = {s: os.environ.get(f'RUN_{s.upper()}', d) == '1' for s, d in _ON.items()}


def run(cmd, stage, cwd=None, env=None, log=None):
    """Run `cmd` (str, bash) if `stage` is enabled; stream the tail of its output."""
    if not STAGES[stage]:
        print(f'[skip: {stage} stage off] {cmd}')
        return
    t = time.time(); e = {**ENV, **(env or {})}
    if log:
        Path(cwd or H, log).parent.mkdir(parents=True, exist_ok=True)
        cmd = f'{cmd} > {log} 2>&1'
    p = subprocess.run(['bash', '-lc', cmd], cwd=cwd or H, env=e, capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip().splitlines()
    print('\n'.join(out[-15:]))
    print(f'[{stage}] exit {p.returncode} in {time.time() - t:.0f}s: {cmd}')
    if p.returncode:
        raise RuntimeError(f'step failed: {cmd}')


def fetch(hf_file, dst):
    """Download `hf_file` from the Hugging Face dataset (cached under <root>/hf_data) and link it to `dst`,
    the path the scripts read. An existing file at `dst` is kept as is."""
    dst = Path(dst)
    if not STAGES['download']:
        print(f'[skip: download stage off] {hf_file} -> {dst}')
        return dst
    from huggingface_hub import hf_hub_download
    src = Path(hf_hub_download(HF_REPO, hf_file, repo_type='dataset', local_dir=H / 'hf_data'))
    if dst.exists():
        print(f'[download] {hf_file}: {dst} already present, kept')
    else:
        dst.parent.mkdir(parents=True, exist_ok=True); dst.symlink_to(src)
        print(f'[download] {hf_file} -> {dst}')
    return dst


FITTED = {'X_haruka_z': 'background', 'X_haruka_s': 'salient', 'haruka_D_z0': 'background_composition',
          'haruka_D_zs': 'condition_composition', 'haruka_salient_shift': 'salient_composition_shift',
          'haruka_M_z0': 'background_neighbor_mean', 'haruka_M_shift': 'salient_neighbor_mean_shift'}


def fetch_fitted(hf_file, fit_dir):
    """Download a published fitted h5ad and unpack its Haruka outputs into `fit_dir`, the layout a fresh fit writes.
    The fit step then reuses them instead of refitting (set HARUKA_REFIT=1 to refit). An existing fit is kept."""
    import json
    import numpy as np
    fit_dir = Path(fit_dir)
    if not STAGES['download'] or os.environ.get('HARUKA_REFIT') == '1':
        print(f'[skip: {"download stage off" if not STAGES["download"] else "HARUKA_REFIT=1"}] {hf_file}')
        return fit_dir
    if (fit_dir / 'salient.npy').exists():
        print(f'[fitted] {fit_dir} already has a fit, kept'); return fit_dir
    from huggingface_hub import hf_hub_download
    import anndata as ad
    A = ad.read_h5ad(hf_hub_download(HF_REPO, hf_file, repo_type='dataset', local_dir=H / 'hf_data'), backed='r')
    fit_dir.mkdir(parents=True, exist_ok=True)
    m = A.obs['haruka_in_fit'].to_numpy().astype(bool)
    for key in A.obsm.keys():
        name = FITTED.get(key) or (f'null_shuffled_s_composition_shift_{key.rsplit("_", 1)[1]}' if key.startswith('haruka_null_shift_') else None)
        if name: np.save(fit_dir / f'{name}.npy', np.asarray(A.obsm[key])[m].astype(np.float32))
    np.save(fit_dir / 'obs_names.npy', A.obs_names.to_numpy()[m].astype(object), allow_pickle=True)
    if 'haruka_M_z0' in A.obsm:   # direct-decoder spatial readouts = [composition, neighbour mean]
        np.save(fit_dir / 'spatial.npy', np.concatenate([np.load(fit_dir / 'background_composition.npy'), np.load(fit_dir / 'background_neighbor_mean.npy')], 1))
        np.save(fit_dir / 'salient_spatial.npy', np.concatenate([np.load(fit_dir / 'salient_composition_shift.npy'), np.load(fit_dir / 'salient_neighbor_mean_shift.npy')], 1))
    json.dump(json.loads(A.uns['haruka']['fit_arguments_json']), open(fit_dir / 'meta.json', 'w'), indent=2)
    print(f'[fitted] {hf_file} -> {fit_dir} ({int(m.sum())} cells)')
    return fit_dir


def require(*paths):
    """Check that inputs produced by an earlier stage exist."""
    miss = [str(p) for p in paths if not Path(p).exists()]
    print('inputs present' if not miss else 'MISSING:\n  ' + '\n  '.join(miss))
    return not miss


def show(png, width=900):
    from IPython.display import Image, display
    display(Image(filename=str(png), width=width))
