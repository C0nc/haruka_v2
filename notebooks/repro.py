"""Shared helpers for the Haruka v2 reproduction notebooks.

Every step is a shell command. A step runs only when its stage flag is on, so a notebook can be executed
end-to-end (fit + analyses + figure, hours on a GPU) or figure-only (redraw from the saved analysis outputs).
"""
import os, subprocess, time
from pathlib import Path

H = Path(os.environ.get('HARUKA_ROOT', '/home/yancui/haruka_v2'))
PY = os.environ.get('HARUKA_PY', '/home/yancui/.conda/envs/haruka2/bin/python')
ENV = {**os.environ, 'NUMPY_MADVISE_HUGEPAGE': '0', 'OPENBLAS_NUM_THREADS': '8', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'}

# Stage switches. Default: redraw figures only. Set via env before launching jupyter, or edit in the notebook.
STAGES = {s: os.environ.get(f'RUN_{s.upper()}', '1' if s == 'figure' else '0') == '1' for s in ('data', 'fit', 'analysis', 'figure')}


def run(cmd, stage, cwd=None, env=None, log=None):
    """Run `cmd` (str, bash) if `stage` is enabled; stream the tail of its output."""
    if not STAGES[stage]:
        print(f'[skip: {stage} stage off] {cmd}')
        return
    t = time.time(); e = {**ENV, **(env or {})}
    if log:
        cmd = f'{cmd} > {log} 2>&1'
    p = subprocess.run(['bash', '-lc', cmd], cwd=cwd or H, env=e, capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip().splitlines()
    print('\n'.join(out[-15:]))
    print(f'[{stage}] exit {p.returncode} in {time.time() - t:.0f}s: {cmd}')
    if p.returncode:
        raise RuntimeError(f'step failed: {cmd}')


def require(*paths):
    """Check that inputs produced by an earlier stage exist."""
    miss = [str(p) for p in paths if not Path(p).exists()]
    print('inputs present' if not miss else 'MISSING:\n  ' + '\n  '.join(miss))
    return not miss


def show(png, width=900):
    from IPython.display import Image, display
    display(Image(filename=str(png), width=width))
