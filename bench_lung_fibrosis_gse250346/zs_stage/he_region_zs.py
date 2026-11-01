#!/usr/bin/env python3
"""Lung fibrosis, fit Unaffected vs all fibrosis: do z or s separate pathologist H&E regions *within* fibrotic cores?
GT: GSE250346_HE_annotations (cells_partitioned_by_annotation.csv), matched to our cells by barcode + centroid.
Fibrosis cores only; cells with >1 region label in the compared set are dropped.

A  region pairs, same core, same cell type.  For a pair (R1, R2) and a cell type: keep cores that have >= 10 cells of
   that type in each region; latent centred per core (removes core / donor / stage offsets); logistic (C = 0.1,
   balanced) on standardised z or s, GroupKFold(5) over donors; readout = mean over cores of the within-core AUROC
   of the held-out score (cores weighted equally).  Cell type kept if >= 3 donors and >= 100 cells per region.
B  ordinal remodelling score within cell type: minimally remodelled alveoli 0, advanced remodelling 1, fibrosis 2,
   severe fibrosis / microscopic honeycombing 3.  Ridge on z or s, GroupKFold(5) over donors; Spearman(held-out
   prediction, level) pooled, and within core (latent centred per core, cores with >= 2 levels, mean over cores).
"""
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
_HW = _os.environ.get('HARUKA_WORK') or ('/project/zhihuanglab/yancui' if _os.path.isdir('/project/zhihuanglab/yancui') else _HR + '/work')
import os, warnings
for v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ.setdefault(v, '8')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
H = Path(f'{_HR}'); OUT = H / 'bench_lung_fibrosis_gse250346/zs_stage/he_region'; OUT.mkdir(exist_ok=True)
MD = H / 'results_real_compgated_nb_final/gse250346_unaffected_vs_all_fibrosis/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
a = ad.read_h5ad(H / 'bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad', backed='r'); o = a.obs
pos = pd.Series(np.arange(len(o)), index=o.index)
Zm = np.load(MD / 'background.npy', mmap_mode='r'); Sm = np.load(MD / 'salient.npy', mmap_mode='r')
d = pd.read_csv(f'{_HW}/haruka_fibrosis_he/he_annot_matched.csv.gz', usecols=['obs', 'annotation_type', 'stage', 'core'])
d = d[d.stage != 'Unaffected'].drop_duplicates(['obs', 'annotation_type'])
d['i'] = pos.reindex(d.obs).to_numpy(); d['ct'] = o.cell_type.astype(str).to_numpy()[d.i]; d['donor'] = o.donor.astype(str).to_numpy()[d.i]
log = open(OUT / 'he_region_zs.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
pd.set_option('display.width', 250)

def centre(M, g):
    M = M.astype(np.float64).copy()
    for c in np.unique(g): k = g == c; M[k] -= M[k].mean(0)
    return M
def lat(idx, core):
    return {'z': centre(np.asarray(Zm[idx]), core), 's': centre(np.asarray(Sm[idx]), core)}

# A
pairs = [('fibroblastic_focus', 'fibrosis'), ('remodeled_epithelium', 'fibrosis'), ('mixed_inflammation', 'fibrosis'),
         ('severe_fibrosis', 'fibrosis'), ('microscopic_honeycombing', 'fibrosis'), ('hyperplastic_aec', 'fibrosis'),
         ('minimally_remodeled_alveoli', 'fibrosis'), ('advanced_remodeling', 'fibrosis')]
rows = []
for r1, r2 in pairs:
    q = d[d.annotation_type.isin([r1, r2])]
    q = q[~q.obs.duplicated(keep=False)]
    for c, g in q.groupby('ct'):
        n = g.groupby(['core', 'annotation_type']).size().unstack(fill_value=0).reindex(columns=[r1, r2], fill_value=0)
        ok = n.index[(n[r1] >= 10) & (n[r2] >= 10)]; g = g[g.core.isin(ok)]
        if g.donor.nunique() < 3 or (g.annotation_type == r1).sum() < 100 or (g.annotation_type == r2).sum() < 100: continue
        g = g.sort_values('i'); y = (g.annotation_type == r1).to_numpy(); core = g.core.to_numpy(); don = g.donor.to_numpy()
        rec = dict(pair=f'{r1} vs {r2}', cell_type=c, n=len(g), n_r1=int(y.sum()), cores=len(ok), donors=g.donor.nunique())
        for k, X in lat(g.i.to_numpy(), core).items():
            sc = np.zeros(len(y))
            for tr, te in GroupKFold(min(5, g.donor.nunique())).split(X, y, don):
                if len(np.unique(y[tr])) < 2: sc[te] = 0; continue
                ss = StandardScaler().fit(X[tr])
                sc[te] = LogisticRegression(C=0.1, max_iter=500, class_weight='balanced').fit(ss.transform(X[tr]), y[tr]).decision_function(ss.transform(X[te]))
            rec[f'auc_{k}'] = np.mean([roc_auc_score(y[core == cc], sc[core == cc]) for cc in ok])
        rows.append(rec)
A = pd.DataFrame(rows); A.to_csv(OUT / 'region_pairs.csv', index=False)
P('== A. region pairs within core, within cell type: mean within-core held-out-donor AUROC ==')
for p, t in A.groupby('pair', sort=False):
    wp = wilcoxon(t.auc_s, t.auc_z).pvalue if len(t) >= 5 else np.nan
    P(f'\n{p}: {len(t)} cell types | median z {t.auc_z.median():.3f}  s {t.auc_s.median():.3f} | s > z in {(t.auc_s > t.auc_z).sum()}/{len(t)} | Wilcoxon p {wp:.2g}')
    P(t.sort_values('n', ascending=False).head(8)[['cell_type', 'n', 'n_r1', 'cores', 'donors', 'auc_z', 'auc_s']].round(3).to_string(index=False))
P(f'\nALL pair x cell type ({len(A)}): median z {A.auc_z.median():.3f}  s {A.auc_s.median():.3f} | s > z in {(A.auc_s > A.auc_z).sum()}/{len(A)} | Wilcoxon p {wilcoxon(A.auc_s, A.auc_z).pvalue:.2g}')

# B
LV = {'minimally_remodeled_alveoli': 0, 'advanced_remodeling': 1, 'fibrosis': 2, 'severe_fibrosis': 3, 'microscopic_honeycombing': 3}
q = d[d.annotation_type.isin(LV)].copy(); q['lv'] = q.annotation_type.map(LV)
q = q.groupby('obs').filter(lambda t: t.lv.nunique() == 1) if False else q[~q.obs.duplicated(keep=False)]
rows = []; rng = np.random.default_rng(0)
for c, g in q.groupby('ct'):
    if len(g) < 2000 or g.donor.nunique() < 8 or g.lv.nunique() < 3: continue
    if len(g) > 40000: g = g.iloc[np.sort(rng.choice(len(g), 40000, replace=False))]
    g = g.sort_values('i'); y = g.lv.to_numpy(float); don = g.donor.to_numpy(); core = g.core.to_numpy(); idx = g.i.to_numpy()
    raw = {'z': np.asarray(Zm[idx], float), 's': np.asarray(Sm[idx], float)}; cen = lat(idx, core)
    rec = dict(cell_type=c, n=len(g), donors=g.donor.nunique())
    for k in ['z', 's']:
        for tag, X in [('pooled', raw[k]), ('within_core', cen[k])]:
            pr = np.zeros(len(y))
            for tr, te in GroupKFold(5).split(X, y, don):
                ss = StandardScaler().fit(X[tr]); pr[te] = Ridge(alpha=10).fit(ss.transform(X[tr]), y[tr]).predict(ss.transform(X[te]))
            if tag == 'pooled': rec[f'{tag}_{k}'] = spearmanr(pr, y).statistic
            else:
                rs = [spearmanr(pr[core == cc], y[core == cc]).statistic for cc in np.unique(core) if len(np.unique(y[core == cc])) >= 2 and (core == cc).sum() >= 30]
                rec[f'{tag}_{k}'] = np.nanmean(rs) if rs else np.nan; rec['n_cores_multi'] = len(rs)
    rows.append(rec)
B = pd.DataFrame(rows); B.to_csv(OUT / 'remodelling_ordinal.csv', index=False)
P('\n== B. ordinal remodelling score (MRA 0 < advanced 1 < fibrosis 2 < severe/honeycomb 3), held-out donor Spearman ==')
P(B.sort_values('n', ascending=False).round(3).to_string(index=False))
for m in ['pooled', 'within_core']:
    t = B[[f'{m}_z', f'{m}_s']].dropna()
    P(f'  {m}: median z {t[f"{m}_z"].median():.3f}  s {t[f"{m}_s"].median():.3f} | s > z in {(t[f"{m}_s"] > t[f"{m}_z"]).sum()}/{len(t)} | Wilcoxon p {wilcoxon(t[f"{m}_s"], t[f"{m}_z"]).pvalue:.2g}')
P('done')
