#!/usr/bin/env python3
"""Hotspot module heatmaps + pathway enrichment for the three graphs.

The main run did not persist `local_correlation_z` (the gene x gene local-correlation
matrix the Hotspot heatmap is drawn from), so it is recomputed here for ONE
representative section across all three arms, then each module's genes are tested
against GO Biological Process and MSigDB Hallmark via Enrichr (gseapy).
"""
from __future__ import annotations
import argparse, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scanpy as sc, hotspot, gseapy
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

CC = ('/home/yancui/haruka_v2/results_real_compgated_nb_final/mouselung_kras_xenium_clean_noC3/'
      'haruka_z_s_cellcharter_cc3_condition_only/')
D = Path('/home/yancui/haruka_v2/bench_kras_xenium/hotspot_arms')
OUT = Path('/home/yancui/haruka_v2/bench_kras_xenium/figures')
ARMS = [('space','xy'), ('z-CC','zcc'), ('s-CC','scc')]
INK, INK2, MUTED, GRID = '#0b0b0b','#52514e','#8a8a86','#e3e3df'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.edgecolor':MUTED,
  'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'figure.facecolor':'white',
  'axes.facecolor':'white'})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--section', default='JB_M1')
    p.add_argument('--n-cells', type=int, default=25000)
    p.add_argument('--n-genes', type=int, default=500)
    p.add_argument('--top-modules', type=int, default=5)
    p.add_argument('--skip-enrichr', action='store_true')
    args = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True)

    h = ad.read_h5ad('/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad', backed='r')
    rows = np.load(CC + 'source_rows.npy')
    obs = h.obs.iloc[rows][['slice_id','celltype','x_centroid','y_centroid','cell_id']].copy()
    sl = obs.slice_id.astype(str).to_numpy()
    idx = np.where(sl == args.section)[0]
    rng = np.random.default_rng(0)                      # same draw as the main run
    take = np.sort(rng.choice(idx, min(args.n_cells, len(idx)), replace=False))
    a = h[rows[take]].to_memory(); h.file.close()
    a.X = a.layers['counts'].copy()
    a.obsm['zcc'] = np.load(CC+'spatial.npy')[take]
    a.obsm['scc'] = np.load(CC+'salient_spatial.npy')[take]
    a.obsm['xy'] = obs.iloc[take][['x_centroid','y_centroid']].to_numpy(np.float64)
    a.obs['total_counts'] = np.asarray(a.X.sum(1)).ravel()
    sc.pp.filter_genes(a, min_cells=20)
    panel = a.var_names.astype(str).tolist()
    print(f'{args.section}: {a.shape}', flush=True)

    store, modstore = {}, {}
    for arm, key in ARMS:
        print(f'--- {arm}: local correlations ---', flush=True)
        hs = hotspot.Hotspot(a, layer_key=None, model='danb', latent_obsm_key=key,
                             umi_counts_obs_key='total_counts')
        hs.create_knn_graph(weighted_graph=False, n_neighbors=30)
        res = hs.compute_autocorrelations()
        genes = res[res.FDR < 0.05].sort_values('Z', ascending=False).index[:args.n_genes].tolist()
        hs.compute_local_correlations(genes, jobs=8)
        mods = hs.create_modules(min_gene_threshold=15, core_only=True, fdr_threshold=0.05)
        store[arm] = hs.local_correlation_z.copy(); modstore[arm] = mods.copy()
        store[arm].to_pickle(D / f'localcorr_{args.section}_{arm}.pkl')
        print(f'    {len(genes)} genes, {len(set(mods[mods>0]))} modules', flush=True)

    # ---------------- heatmaps ----------------
    fig = plt.figure(figsize=(14.4, 5.3))
    gs = gridspec.GridSpec(1, 3, wspace=0.22, left=0.045, right=0.97, top=0.85, bottom=0.08)
    for i,(arm,_) in enumerate(ARMS):
        Z = store[arm]; mods = modstore[arm]
        keep = mods[mods > 0].sort_values()
        Zk = Z.loc[keep.index, keep.index]
        # order genes within each module by hierarchical clustering
        order = []
        for m in sorted(set(keep)):
            g = keep[keep == m].index.tolist()
            if len(g) > 2:
                sub = Zk.loc[g, g].to_numpy(); d = sub.max() - sub; np.fill_diagonal(d, 0)
                g = [g[j] for j in leaves_list(linkage(squareform(d, checks=False), 'average'))]
            order += g
        Zo = Zk.loc[order, order].to_numpy()
        ax = fig.add_subplot(gs[0, i])
        v = np.nanpercentile(np.abs(Zo), 97)
        im = ax.imshow(Zo, cmap='RdBu_r', vmin=-v, vmax=v, interpolation='nearest')
        bounds, c = [], 0
        for m in sorted(set(keep)):
            c += (keep == m).sum(); bounds.append(c)
        for b in bounds[:-1]:
            ax.axhline(b-.5, color='#111', lw=.7); ax.axvline(b-.5, color='#111', lw=.7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{arm}  ·  {len(set(keep))} modules, {len(order)} genes',
                     loc='left', fontsize=9.5, color=INK)
        cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.02); cb.set_label('local correlation Z', fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(f'Hotspot module structure on three graphs — section {args.section}',
                 x=0.045, ha='left', fontsize=12, color=INK, fontweight='bold')
    hp = OUT/'hotspot_heatmaps.png'
    fig.savefig(hp, dpi=200, bbox_inches='tight'); fig.savefig(str(hp).replace('.png','.pdf'), bbox_inches='tight')
    print('wrote', hp, flush=True)

    if args.skip_enrichr: return
    # ---------------- enrichment ----------------
    print('\n--- Enrichr ---', flush=True)
    recs = []
    for arm,_ in ARMS:
        mods = modstore[arm]
        sizes = mods[mods>0].value_counts().head(args.top_modules)
        for m in sizes.index:
            genes = mods[mods==m].index.tolist()
            try:
                e = gseapy.enrichr(gene_list=genes, background=panel,
                                   gene_sets=['GO_Biological_Process_2026','MSigDB_Hallmark_2020'],
                                   organism='Mouse', outdir=None, no_plot=True)
                r = e.results.sort_values('Adjusted P-value').head(5)
                for _,row in r.iterrows():
                    recs.append(dict(arm=arm, module=int(m), n_genes=len(genes),
                                     term=row['Term'], padj=row['Adjusted P-value'],
                                     overlap=row['Overlap'], genes=row['Genes']))
                top = r.iloc[0]
                print(f'  {arm} mod {int(m)} (n={len(genes)}): {top["Term"]}  padj={top["Adjusted P-value"]:.2e}', flush=True)
            except Exception as ex:
                print(f'  {arm} mod {int(m)}: enrichr failed {type(ex).__name__}', flush=True)
            time.sleep(1.2)
    if recs:
        E = pd.DataFrame(recs); E.to_csv(D/f'enrichment_{args.section}.csv', index=False)
        sig = E[E.padj < 0.05]
        fig2, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
        for ax,(arm,_) in zip(axes, ARMS):
            s = sig[sig.arm==arm].drop_duplicates('term').nsmallest(10, 'padj')
            if len(s)==0:
                ax.text(.5,.5,'no term at padj < 0.05',ha='center',va='center',
                        transform=ax.transAxes,color=MUTED); ax.axis('off')
                ax.set_title(arm, loc='left', fontsize=9.5, color=INK); continue
            y = np.arange(len(s))[::-1]
            ax.barh(y, -np.log10(s.padj), color={'space':'#0f6e7a','z-CC':'#2a78d6','s-CC':'#eb6834'}[arm], height=.62)
            ax.set_yticks(y); ax.set_yticklabels([t[:52] for t in s.term], fontsize=7.4)
            ax.set_xlabel('−log10 adjusted P'); ax.set_title(f'{arm}', loc='left', fontsize=9.5, color=INK)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.grid(axis='x', color=GRID, lw=.7); ax.set_axisbelow(True)
        fig2.suptitle(f'Module pathway enrichment by graph — section {args.section}',
                      x=0.02, ha='left', fontsize=12, color=INK, fontweight='bold')
        fig2.tight_layout(rect=[0,0,1,.94])
        ep = OUT/'hotspot_enrichment.png'
        fig2.savefig(ep, dpi=200, bbox_inches='tight'); fig2.savefig(str(ep).replace('.png','.pdf'), bbox_inches='tight')
        print('wrote', ep, flush=True)


if __name__ == '__main__':
    main()
