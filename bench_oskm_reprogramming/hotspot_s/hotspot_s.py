#!/usr/bin/env python3
"""OSKM: which gene programmes does Haruka's s organise? Hotspot on the per-cell s graph vs the z graph.

Fit: results_oskm_cf_expression (pair batch; old control = reference, old OSKM = condition).
For one cell type, OSKM cells only, run SEPARATELY in each pair (B9..B14) so pair offsets cannot make a gene
look structured:  Hotspot (DANB, raw counts, 30-NN unweighted) with the graph built on s, and on z.
  per gene: autocorrelation Z on s and on z, per pair -> s-specific = Z_s - Z_z, replicated over pairs
  modules: Hotspot local-correlation modules on the s graph (top autocorrelated genes)
  readout per module: OSKM - control difference of the module's mean log expression (same cell type, same pair)
usage: hotspot_s.py <cell_type>
"""
import os, sys, json, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4'); os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np, pandas as pd, anndata as ad, hotspot
from pathlib import Path
warnings.filterwarnings('ignore')
H = Path('/home/yancui/haruka_v2'); OUT = H / 'bench_oskm_reprogramming/hotspot_s'
MD = H / 'results_oskm_cf_expression/oskm_old_full_684253/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
CT = sys.argv[1]; N_PER_PAIR, KNN, TOP = 6000, 30, 150
tag = CT.replace('-', '_')
log = open(OUT / f'{tag}.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()

h = ad.read_h5ad(H / 'bench_oskm_reprogramming/data/oskm_old_full_684253.h5ad', backed='r')
assert (np.load(MD / 'obs_names.npy', allow_pickle=True).astype(str) == np.asarray(h.obs_names).astype(str)).all()
o = h.obs; ct = o.cell_type.astype(str).to_numpy(); pair = o.paired_batch.astype(str).to_numpy()
oskm = (o.condition.astype(str) == 'oskm').to_numpy()
Z = np.load(MD / 'background.npy'); S = np.load(MD / 'salient.npy')
rng = np.random.default_rng(0)
take = {p: np.sort(rng.choice(np.flatnonzero((ct == CT) & (pair == p) & oskm),
                              min(N_PER_PAIR, int(((ct == CT) & (pair == p) & oskm).sum())), replace=False))
        for p in sorted(set(pair))}
ctrl = np.flatnonzero((ct == CT) & ~oskm)
allrows = np.sort(np.concatenate(list(take.values()) + [ctrl]))
A = h[allrows].to_memory(); h.file.close()
pos = pd.Series(np.arange(len(allrows)), index=allrows)
C = A.layers['counts'].tocsr(); genes = np.asarray(A.var_names).astype(str)
tot = np.asarray(C.sum(1)).ravel()
L = C.multiply(1e3 / np.maximum(tot, 1)[:, None]).tocsr(); L.data = np.log1p(L.data)
P(f'{CT}: OSKM cells per pair ' + ', '.join(f'{p} {len(v)}' for p, v in take.items()) + f'; control cells {len(ctrl)}')

auto, mods_all = [], []
for p, rows in take.items():
    b = ad.AnnData(X=C[pos[rows].to_numpy()].astype(np.float64), var=pd.DataFrame(index=genes))
    b.obs['total_counts'] = tot[pos[rows].to_numpy()]
    keep = np.asarray((b.X > 0).sum(0)).ravel() >= 20; b = b[:, keep].copy()
    b.obsm['s'] = S[rows]; b.obsm['z'] = Z[rows]
    for arm in ['s', 'z']:
        hs = hotspot.Hotspot(b, layer_key=None, model='danb', latent_obsm_key=arm, umi_counts_obs_key='total_counts')
        hs.create_knn_graph(weighted_graph=False, n_neighbors=KNN)
        r = hs.compute_autocorrelations(jobs=4)
        auto.append(r.assign(pair=p, arm=arm, gene=r.index))
        if True:
            g = r[r.FDR < 0.05].sort_values('Z', ascending=False).index[:TOP].tolist()
            if len(g) >= 20:
                hs.compute_local_correlations(g, jobs=4)
                m = hs.create_modules(min_gene_threshold=8, core_only=True, fdr_threshold=0.05)
                mods_all.append(pd.DataFrame(dict(gene=m.index, module=m.to_numpy(), pair=p, arm=arm)))
    P(f'   {p}: done ({b.n_vars} genes)')
AU = pd.concat(auto, ignore_index=True); AU.to_csv(OUT / f'{tag}_autocorrelation.csv.gz', index=False)
W = AU.pivot_table(index=['gene', 'pair'], columns='arm', values='Z').reset_index()
W['s_minus_z'] = W.s - W.z
G = W.groupby('gene').agg(Z_s=('s', 'median'), Z_z=('z', 'median'), s_minus_z=('s_minus_z', 'median'),
                          pairs_s_gt_z=('s_minus_z', lambda v: int((v > 0).sum())), n_pairs=('s', 'size'))
G = G[G.n_pairs == 4].sort_values('s_minus_z', ascending=False)

# OSKM - control difference of each gene, within cell type x pair (mean log expression), median over pairs
def gene_shift(rows_t, rows_c):
    return np.asarray(L[pos[rows_t].to_numpy()].mean(0)).ravel() - np.asarray(L[pos[rows_c].to_numpy()].mean(0)).ravel()
cp = pair[ctrl]
SH = np.median(np.stack([gene_shift(take[p], ctrl[cp == p]) for p in take]), axis=0)
G['oskm_shift_log'] = pd.Series(SH, index=genes).reindex(G.index)
G.to_csv(OUT / f'{tag}_genes.csv')
P(f'\n== per-gene autocorrelation (median over 4 pairs) ==')
P(f'   genes with FDR<0.05: s graph ' + ', '.join(f'{p} {int(((AU.pair == p) & (AU.arm == "s") & (AU.FDR < .05)).sum())}' for p in take)
  + ' | z graph ' + ', '.join(f'{p} {int(((AU.pair == p) & (AU.arm == "z") & (AU.FDR < .05)).sum())}' for p in take))
P(f'   s-specific genes (Z_s > Z_z in 4/4 pairs): {int((G.pairs_s_gt_z == 4).sum())} of {len(G)}')
P('   top s-specific: ' + ', '.join(f'{g} ({r.s_minus_z:+.1f}, shift {r.oskm_shift_log:+.2f})' for g, r in G[G.pairs_s_gt_z == 4].head(20).iterrows()))
P('   top z-specific: ' + ', '.join(f'{g} ({r.s_minus_z:+.1f})' for g, r in G[G.pairs_s_gt_z == 0].sort_values('s_minus_z').head(10).iterrows()))
sp = G[G.pairs_s_gt_z == 4]; ot = G[G.pairs_s_gt_z < 4]
P(f'   |OSKM - control shift|: s-specific genes median {sp.oskm_shift_log.abs().median():.3f} vs other genes {ot.oskm_shift_log.abs().median():.3f}')

from scipy.stats import spearmanr
P('\n== does each graph organise the genes OSKM changes? Spearman(median autocorrelation Z, |OSKM - control shift|) ==')
P(f'   s graph {spearmanr(G.Z_s, G.oskm_shift_log.abs()).statistic:+.2f} | z graph {spearmanr(G.Z_z, G.oskm_shift_log.abs()).statistic:+.2f}')
# modules per graph, replicated across pairs: gene pairs co-assigned in >= 3 of 4 pairs
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
MA = pd.concat(mods_all, ignore_index=True); MA.to_csv(OUT / f'{tag}_modules_by_pair.csv', index=False)
cp = pair[ctrl]; out = []
for arm in ['s', 'z']:
    M = MA[(MA.arm == arm) & (MA.module > 0)]
    gl = sorted(M.gene.unique()); gi = {g: i for i, g in enumerate(gl)}; co = np.zeros((len(gl), len(gl)))
    for _, g in M.groupby('pair'):
        for _, mm in g.groupby('module'):
            ix = [gi[x] for x in mm.gene]; co[np.ix_(ix, ix)] += 1
    adj = (co >= 3).astype(int); np.fill_diagonal(adj, 0)
    _, lab = connected_components(csr_matrix(adj), directed=False)
    rep = sorted([[gl[i] for i in np.flatnonzero(lab == l)] for l in np.unique(lab) if (lab == l).sum() >= 5], key=len, reverse=True)
    P(f'\n== {arm}-graph modules replicated in >= 3 of 4 pairs: {len(rep)} ({sum(map(len, rep))} genes) ==')
    for k, gs in enumerate(rep):
        ix = [list(genes).index(x) for x in gs]
        per = [np.asarray(L[pos[take[p]].to_numpy()][:, ix].mean(1)).ravel().mean() -
               np.asarray(L[pos[ctrl[cp == p]].to_numpy()][:, ix].mean(1)).ravel().mean() for p in take]
        P(f'   {arm.upper()}{k + 1} ({len(gs)} genes, OSKM - control {np.median(per):+.3f}, same sign {max(sum(np.array(per) > 0), sum(np.array(per) < 0))}/4): ' + ', '.join(gs))
        out.append(dict(arm=arm, module=f'{arm.upper()}{k + 1}', n=len(gs), oskm_shift=np.median(per), genes=';'.join(gs)))
R = pd.DataFrame(out); R.to_csv(OUT / f'{tag}_replicated_modules.csv', index=False)
for arm in ['s', 'z']:
    r = R[R.arm == arm]
    if len(r): P(f'   {arm}: gene-weighted mean |module OSKM shift| {np.average(r.oskm_shift.abs(), weights=r.n):.3f}')
P('done')
