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


def require(*paths):
    """Check that inputs produced by an earlier stage exist."""
    miss = [str(p) for p in paths if not Path(p).exists()]
    print('inputs present' if not miss else 'MISSING:\n  ' + '\n  '.join(miss))
    return not miss


def show(png, width=900):
    from IPython.display import Image, display
    display(Image(filename=str(png), width=width))
