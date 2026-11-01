#!/usr/bin/env python3
"""Hotspot gene programmes on three graphs: real space, z-CC, s-CC.

Hotspot (DeTomaso & Yosef) finds genes with autocorrelation on a cell-cell KNN
graph, then groups them into modules by local correlation.  Swapping only the
graph asks: which gene programmes does each embedding consider 'spatially'
structured?

Real spatial coordinates are the reference arm — the programmes that are
genuinely spatially organised.  Everything runs PER SECTION so all three arms use
within-section neighbourhoods and are directly comparable.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scanpy as sc, hotspot
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr

CC = ('/home/yancui/haruka_v2/results_real_compgated_nb_final/mouselung_kras_xenium_clean_noC3/'
      'haruka_z_s_cellcharter_cc3_condition_only/')
BIO = '/home/yancui/haruka_v2/bench_kras_xenium/biological_responder/'
TUM = ['Tumor_EMT', 'Tumor_epithelial', 'Tumor_epithelial_Itgb6']


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', type=Path,
                   default=Path('/home/yancui/haruka_v2/bench_kras_xenium/hotspot_arms'))
    p.add_argument('--n-cells', type=int, default=25000)
    p.add_argument('--n-neighbors', type=int, default=30)
    p.add_argument('--n-genes-module', type=int, default=500)
    p.add_argument('--min-gene-threshold', type=int, default=20)
    p.add_argument('--fdr', type=float, default=0.05)
    args = p.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)

    h = ad.read_h5ad('/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad', backed='r')
    rows = np.load(CC + 'source_rows.npy')
    obs = h.obs.iloc[rows][['slice_id', 'celltype', 'x_centroid', 'y_centroid', 'cell_id']].copy()
    sl = obs.slice_id.astype(str).to_numpy()
    zcc = np.load(CC + 'spatial.npy'); scc = np.load(CC + 'salient_spatial.npy')

    take, sec_of = [], []
    for s in sorted(np.unique(sl)):
        idx = np.where(sl == s)[0]
        rng = np.random.default_rng(0)
        t = np.sort(rng.choice(idx, min(args.n_cells, len(idx)), replace=False))
        take.append(t); sec_of += [s] * len(t)
    take = np.concatenate(take); sec_of = np.array(sec_of)
    print(f'loading {len(take)} cells positionally...', flush=True)
    a = h[rows[take]].to_memory(); h.file.close()
    a.obs['sec'] = sec_of
    a.obs['cid_idx'] = a.obs_names.astype(str)
    a.X = a.layers['counts'].copy()
    a.obsm['zcc'] = zcc[take]; a.obsm['scc'] = scc[take]
    a.obsm['xy'] = obs.iloc[take][['x_centroid', 'y_centroid']].to_numpy(np.float64)
    a.obs['total_counts'] = np.asarray(a.X.sum(1)).ravel()
    print(f'loaded {a.shape}', flush=True)

    bio = pd.read_csv(BIO + 'treated_tumor_biological_response.csv.gz').set_index('cell_id')
    a.obs['MAPK'] = bio.reindex(a.obs['cell_id'].astype(str))['MAPK_MPAS'].to_numpy()
    ct = a.obs.celltype.astype(str).to_numpy(); a.obs['is_tumor'] = np.isin(ct, TUM)

    auto_all, mod_all, summary = [], [], {}
    for sec in sorted(np.unique(sec_of)):
        b = a[a.obs.sec == sec].copy()
        sc.pp.filter_genes(b, min_cells=args.min_gene_threshold)
        # local tumour fraction on the real spatial graph, for interpreting modules
        from sklearn.neighbors import NearestNeighbors
        XY = b.obsm['xy']
        nn = NearestNeighbors(n_neighbors=21).fit(XY).kneighbors(XY, return_distance=False)[:, 1:]
        b.obs['local_tumor_frac'] = b.obs.is_tumor.to_numpy()[nn].mean(1)

        for arm, key in [('space', 'xy'), ('z-CC', 'zcc'), ('s-CC', 'scc')]:
            print(f'--- {sec} / {arm} ---', flush=True)
            hs = hotspot.Hotspot(b, layer_key=None, model='danb',
                                 latent_obsm_key=key, umi_counts_obs_key='total_counts')
            hs.create_knn_graph(weighted_graph=False, n_neighbors=args.n_neighbors)
            res = hs.compute_autocorrelations()
            res = res.assign(section=sec, arm=arm, gene=res.index)
            auto_all.append(res)
            sig = res[res.FDR < args.fdr].sort_values('Z', ascending=False)
            genes = sig.index[:args.n_genes_module].tolist()
            print(f'    {len(sig)} genes FDR<{args.fdr}; modules on top {len(genes)}', flush=True)
            if len(genes) < 20:
                summary[f'{sec}|{arm}'] = dict(n_sig=int(len(sig)), n_modules=0); continue
            hs.compute_local_correlations(genes, jobs=8)
            mods = hs.create_modules(min_gene_threshold=15, core_only=True, fdr_threshold=0.05)
            scores = hs.calculate_module_scores()
            mod_all.append(pd.DataFrame(dict(gene=mods.index, module=mods.to_numpy(),
                                             section=sec, arm=arm)))
            cors = {}
            for m in sorted(set(mods[mods > 0])):
                v = scores[m].to_numpy()
                tm = np.isfinite(b.obs.MAPK.to_numpy())
                cors[int(m)] = dict(
                    n_genes=int((mods == m).sum()),
                    rho_local_tumor=float(spearmanr(v, b.obs.local_tumor_frac).statistic),
                    rho_MAPK=float(spearmanr(v[tm], b.obs.MAPK.to_numpy()[tm]).statistic)
                        if tm.sum() > 100 else float('nan'),
                    top=mods[mods == m].index[:10].tolist())
            summary[f'{sec}|{arm}'] = dict(n_sig=int(len(sig)),
                                           n_modules=int(len(cors)), modules=cors)
            print(f'    {len(cors)} modules', flush=True)

    pd.concat(auto_all, ignore_index=True).to_csv(args.out_dir/'autocorrelation.csv.gz',
                                                  index=False, compression='gzip')
    if mod_all:
        pd.concat(mod_all, ignore_index=True).to_csv(args.out_dir/'modules.csv', index=False)
    (args.out_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('\nwrote', args.out_dir, flush=True)

    A = pd.concat(auto_all, ignore_index=True)
    print('\n=== genes with significant spatial autocorrelation (FDR<0.05) ===')
    print(A[A.FDR < args.fdr].groupby(['section','arm']).size().unstack().to_string())
    print('\n=== rank agreement of per-gene autocorrelation Z, vs the real-space arm ===')
    for sec in sorted(A.section.unique()):
        s = A[A.section == sec].pivot_table(index='gene', columns='arm', values='Z')
        s = s.dropna()
        print(f'  {sec}: z-CC vs space rho={spearmanr(s["z-CC"], s["space"]).statistic:+.3f} | '
              f's-CC vs space rho={spearmanr(s["s-CC"], s["space"]).statistic:+.3f} | '
              f'z-CC vs s-CC rho={spearmanr(s["z-CC"], s["s-CC"]).statistic:+.3f}')


if __name__ == '__main__':
    main()
