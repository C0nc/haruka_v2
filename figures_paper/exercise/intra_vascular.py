#!/usr/bin/env python3
"""Exercise, within vascular cell types: what explains the cell-to-cell variation in rejuvenation?

Readouts per cell (held-out, from make_exercise_clock_shift.py; subregion offsets from training mice):
  observed  = clock age from the frozen Sun et al. clock on observed expression (ground truth)
  s, z      = clock age predicted from s / from z (held-out mouse pair)
Candidate sources of within-type variation (vascular types: Endothelial, Pericyte, VLMC, VSMC):
  mouse      eta^2 of the readout on mouse (section)
  region     eta^2 on subregion (within mouse)
  niche      neighbour cell-type proportions (k = 20, within the section): out-of-sample R^2 of a ridge on
             within-mouse-centred proportions, one mouse held out at a time; separately in exercise and in control cells
Exercise-specific niche dependence = niche R^2 in exercise cells - in control cells.
Drivers: per neighbour type, Spearman(readout, proportion) within each mouse; exercise mean - control mean, and in
how many of the 4 exercise mice the sign holds.
"""
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
_HW = _os.environ.get('HARUKA_WORK') or ('/project/zhihuanglab/yancui' if _os.path.isdir('/project/zhihuanglab/yancui') else _HR + '/work')
import os, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8'); os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RidgeCV
warnings.filterwarnings('ignore')
OUT = Path(f'{_HR}/figures_paper/exercise')
VAS = ['Endothelial', 'Pericyte', 'VLMC', 'VSMC']
h = ad.read_h5ad(f'{_HW}/haruka_exercise/data/exercise_old_full.h5ad', backed='r'); o = h.obs
ct = o.celltype.astype(str).to_numpy(); mouse = o.mouse_id.astype(str).to_numpy(); xy = np.asarray(h.obsm['spatial'], float)[:, :2]
types = sorted(set(ct)); code = pd.Series(range(len(types)), index=types)[ct].to_numpy()
names = np.asarray(h.obs_names).astype(str)
D = pd.read_csv(OUT / 'oof_clock_predictions_w10_s100.csv.gz').set_index('cell_id')
D = D[D.cell_type.isin(VAS)]
pos = pd.Index(names).get_indexer(D.index); assert (pos >= 0).all()

# neighbour composition (k = 20 within each mouse section, self excluded) for vascular cells
comp = np.zeros((len(D), len(types)), np.float32)
for m in np.unique(mouse):
    allm = np.flatnonzero(mouse == m); q = np.flatnonzero(mouse[pos] == m)
    if not len(q): continue
    nn = NearestNeighbors(n_neighbors=21).fit(xy[allm]); _, ind = nn.kneighbors(xy[pos[q]])
    nb = code[allm][ind[:, 1:]]
    for j in range(len(types)): comp[q, j] = (nb == j).mean(1)
C = pd.DataFrame(comp, index=D.index, columns=[f'nb_{t}' for t in types])
C.to_csv(OUT / 'vascular_neighbour_composition.csv.gz')

def eta2(v, g):
    v = v - v.mean(); tot = (v ** 2).sum()
    return sum(((g == u).sum() * v[g == u].mean() ** 2) for u in np.unique(g)) / tot
def wcentre(X, g):
    X = np.asarray(X, float).copy()
    for u in np.unique(g): X[g == u] -= X[g == u].mean(0)
    return X
def niche_r2(y, X, g):
    y = wcentre(y[:, None], g).ravel(); X = wcentre(X, g); pred = np.zeros_like(y)
    for u in np.unique(g):
        te = g == u; m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[~te], y[~te]); pred[te] = m.predict(X[te])
    return 1 - ((y - pred) ** 2).sum() / (y ** 2).sum()

part, drv = [], []
for c in VAS + ['all vascular']:
    d = D if c == 'all vascular' else D[D.cell_type == c]
    X = C.loc[d.index].to_numpy()
    for k in ['observed', 'z', 's']:
        rec = dict(cell_type=c, readout=k)
        for arm, am in [('exercise', d.exercise.to_numpy()), ('control', ~d.exercise.to_numpy())]:
            y = d[k].to_numpy()[am]; g = d.mouse.to_numpy()[am]; sb = d.subregion.to_numpy()[am]
            rec[f'mouse_eta2_{arm}'] = eta2(y, g)
            rec[f'region_eta2_{arm}'] = eta2(wcentre(y[:, None], g).ravel(), sb)
            rec[f'niche_r2_{arm}'] = niche_r2(y, X[am], g)
        rec['niche_r2_exercise_minus_control'] = rec['niche_r2_exercise'] - rec['niche_r2_control']
        part.append(rec)
        if c == 'all vascular': continue
        for j, t in enumerate(types):
            if X[:, j].mean() < 0.01: continue
            r = {}
            for arm, am in [('exercise', d.exercise.to_numpy()), ('control', ~d.exercise.to_numpy())]:
                r[arm] = [spearmanr(d[k].to_numpy()[am & (d.mouse.to_numpy() == mm)], X[am & (d.mouse.to_numpy() == mm), j]).statistic
                          for mm in np.unique(d.mouse.to_numpy()[am])]
            ex_spec = np.mean(r['exercise']) - np.mean(r['control'])
            drv.append(dict(cell_type=c, readout=k, neighbour=t, rho_exercise=np.mean(r['exercise']), rho_control=np.mean(r['control']),
                            exercise_specific=ex_spec, ex_mice_same_sign=int(max(np.sum(np.array(r['exercise']) > 0), np.sum(np.array(r['exercise']) < 0))),
                            ex_mice_rho=' '.join(f'{v:+.2f}' for v in r['exercise'])))
Pt = pd.DataFrame(part); Pt.to_csv(OUT / 'intra_vascular_partition.csv', index=False)
Dr = pd.DataFrame(drv); Dr.to_csv(OUT / 'intra_vascular_drivers.csv', index=False)
pd.set_option('display.width', 250)
print('== variance sources of the rejuvenation readout within vascular types ==')
print(Pt.round(3).to_string(index=False))
print('\n== top exercise-specific neighbour drivers (|exercise - control| Spearman), per cell type and readout ==')
for (c, k), g in Dr.groupby(['cell_type', 'readout']):
    g = g.reindex(g.exercise_specific.abs().sort_values(ascending=False).index).head(4)
    print(f'{c:12s} {k:8s} ' + ' | '.join(f'{r.neighbour} {r.exercise_specific:+.2f} (ex {r.rho_exercise:+.2f}, ctrl {r.rho_control:+.2f}, {r.ex_mice_same_sign}/4)' for r in g.itertuples()))
