#!/usr/bin/env python3
"""GASTON isodepth from z vs from s, per fibrosis core: which latent's spatial topography follows the lesion gradient?
Per core with >= 30 cells of a target lesion (fibroblastic focus / mixed inflammation / TLS / remodelled epithelium):
features A = top-8 PCs of the latent (standardised, centred per cell type within the core, so the isodepth has to come
from cell STATE, not from where each cell type sits); S = centroids.  GASTON (gaston-spatial 0.0.3) f_S [20,20], f_A [20,20],
Adam lr 1e-3, 4000 full-batch epochs, seed 0.  Readout: |Spearman(isodepth, distance to nearest lesion cell)| over the
non-lesion cells within 400 um, per core and lesion; z vs s paired over cores.
"""
import os, sys, warnings
for v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']: os.environ.setdefault(v, '8')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import numpy as np, pandas as pd, anndata as ad, torch
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, wilcoxon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from gaston import neural_net as nn_
warnings.filterwarnings('ignore')
H = Path('/home/yancui/haruka_v2'); OUT = H / 'bench_lung_fibrosis_gse250346/zs_stage/lesion_gradient'; OUT.mkdir(exist_ok=True)
MD = H / 'results_real_compgated_nb_final/gse250346_unaffected_vs_all_fibrosis/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
o = ad.read_h5ad(H / 'bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad', backed='r').obs
log = open(OUT / 'lesion_gaston.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
core = o['sample'].astype(str).to_numpy(); ct = o.cell_type.astype(str).to_numpy(); stage = o.sample_affect.astype(str).to_numpy()
xy = np.c_[o.x_centroid.to_numpy(float), o.y_centroid.to_numpy(float)]
pos = pd.Series(np.arange(len(o)), index=o.index)
he = pd.read_csv('/project/zhihuanglab/yancui/haruka_fibrosis_he/he_annot_matched.csv.gz', usecols=['obs', 'annotation_type']); he['i'] = pos.reindex(he.obs).to_numpy()
LES = ['fibroblastic_focus', 'mixed_inflammation', 'TLS', 'remodeled_epithelium']
Zm = np.load(MD / 'background.npy', mmap_mode='r'); Sm = np.load(MD / 'salient.npy', mmap_mode='r')
cores = sorted({c for L in LES for c, n in pd.Series(core[he.i[he.annotation_type == L]]).value_counts().items() if n >= 30} & set(core[stage != 'Unaffected']))
P(f'{len(cores)} fibrosis cores with >= 30 cells of a target lesion')
rows = []
for c in cores:
    ci = np.flatnonzero(core == c)
    for k, M in [('z', Zm), ('s', Sm)]:
        X = StandardScaler().fit_transform(np.asarray(M[ci]))
        for t in np.unique(ct[ci]): m = ct[ci] == t; X[m] -= X[m].mean(0)
        A = PCA(8, random_state=0).fit_transform(X)
        S_t, A_t = nn_.load_rescale_input_data(xy[ci], A)
        mod, loss = nn_.train(S_t, A_t, S_hidden_list=[20, 20], A_hidden_list=[20, 20], epochs=4000, checkpoint=10**9, optim='adam', lr=1e-3, seed=0)
        with torch.no_grad(): iso = mod.spatial_embedding(S_t.to(next(mod.parameters()).device)).cpu().numpy().ravel()
        for L in LES:
            lc = he.i[(he.annotation_type == L).to_numpy() & (core[he.i] == c)].to_numpy()
            if len(lc) < 30: continue
            nl = ~np.isin(ci, lc); d, _ = cKDTree(xy[lc]).query(xy[ci[nl]]); keep = d <= 400
            if keep.sum() < 200: continue
            rows.append(dict(core=c, lesion=L, latent=k, n=int(keep.sum()), rho=abs(spearmanr(iso[nl][keep], d[keep]).statistic), final_loss=float(loss[-1])))
    P(f'  {c}: done ({len(ci)} cells)')
    pd.DataFrame(rows).to_csv(OUT / 'gaston_isodepth_vs_lesion.csv', index=False)
T = pd.DataFrame(rows).pivot_table(index=['core', 'lesion'], columns='latent', values='rho').dropna().reset_index()
P('\n== |Spearman(GASTON isodepth, distance to lesion)| per core, z vs s ==')
for L, t in T.groupby('lesion'):
    P(f'  {L}: {len(t)} cores | median z {t.z.median():.3f}  s {t.s.median():.3f} | s > z in {(t.s > t.z).sum()}/{len(t)}' + (f' | Wilcoxon p {wilcoxon(t.s, t.z).pvalue:.2g}' if len(t) >= 6 else ''))
P(f'  ALL: {len(T)} core x lesion | median z {T.z.median():.3f}  s {T.s.median():.3f} | s > z in {(T.s > T.z).sum()}/{len(T)} | Wilcoxon p {wilcoxon(T.s, T.z).pvalue:.2g}')
P('done')
