#!/usr/bin/env python3
"""GASTON on RAW GENE EXPRESSION (its intended input), for comparison with the
salient-CC fit.

If gene-based GASTON yields a clean isodepth where the salient-CC fit did not,
the embedding was the problem.  If both are weak, the tissue is not topographic.
Uses the SAME subsampled cells as run_gaston_on_scc.py so the two are comparable.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scanpy as sc, torch
warnings.filterwarnings('ignore')
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from gaston import neural_net, dp_related

CC = ('/home/yancui/haruka_v2/results_real_compgated_nb_final/mouselung_kras_xenium_clean_noC3/'
      'haruka_z_s_cellcharter_cc3_condition_only/')
BIO = '/home/yancui/haruka_v2/bench_kras_xenium/biological_responder/'
TUM = ['Tumor_EMT', 'Tumor_epithelial', 'Tumor_epithelial_Itgb6']


def segment(A, iso, lmax=6):
    for L in range(lmax, 1, -1):
        for nb in (50, 30, 20):
            try:
                be = np.linspace(iso.min(), iso.max() + 0.01, nb + 1)
                em, sm = dp_related.dp_bucketized(A.T, be, L, xcoords=iso)
                segs = dp_related.find_segments_from_dp(em, sm, L)
                bl = np.digitize(iso, be); lab = np.zeros(len(iso))
                for c, sg in enumerate(segs):
                    for s_ in sg:
                        lab[np.where(bl == s_ + 1)[0]] = c
                if len(np.unique(lab)) == L:
                    return lab, L, nb
            except Exception:
                continue
    return None, None, None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', type=Path,
                   default=Path('/home/yancui/haruka_v2/bench_kras_xenium/gaston_genes'))
    p.add_argument('--n-cells', type=int, default=30000)
    p.add_argument('--n-pcs', type=int, default=15)
    p.add_argument('--n-hvg', type=int, default=2000)
    p.add_argument('--epochs', type=int, default=4000)
    p.add_argument('--seeds', type=int, default=3)
    args = p.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    h = ad.read_h5ad('/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad', backed='r')
    rows = np.load(CC + 'source_rows.npy')
    obs = h.obs.iloc[rows][['slice_id','celltype','x_centroid','y_centroid','cell_id']].copy()
    obs['cell_id'] = obs['cell_id'].astype(str)
    sl = obs.slice_id.astype(str).to_numpy()

    take_all, sec_of = [], []
    for sec in sorted(np.unique(sl)):                       # identical subsample to the s-CC run
        idx = np.where(sl == sec)[0]
        rng = np.random.default_rng(0)
        t = np.sort(rng.choice(idx, min(args.n_cells, len(idx)), replace=False))
        take_all.append(t); sec_of += [sec] * len(t)
    take_all = np.concatenate(take_all); sec_of = np.array(sec_of)
    # cell_id is NOT globally unique (52 IDs collide ACROSS arms); the obs index is.
    # Select positionally so order is preserved and no join is needed.
    global_rows = rows[take_all]
    print(f'loading {len(global_rows)} cells x genes (positional)...', flush=True)
    a = h[global_rows].to_memory(); h.file.close()
    a.obs['cid'] = a.obs_names.astype(str)          # unique obs index, not cell_id
    a.obs['sec'] = sec_of
    a.X = a.layers['counts'].copy()
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    print(f'loaded {a.shape}', flush=True)

    bio = pd.read_csv(BIO + 'treated_tumor_biological_response.csv.gz').set_index('cell_id')
    res, frames = {}, []
    for sec in sorted(np.unique(sec_of)):
        m = (a.obs.sec == sec).to_numpy()
        b = a[m].copy()
        sc.pp.highly_variable_genes(b, n_top_genes=args.n_hvg)
        bb = b[:, b.var.highly_variable].copy()
        sc.pp.scale(bb, max_value=10); sc.tl.pca(bb, n_comps=args.n_pcs, random_state=0)
        A = np.asarray(bb.obsm['X_pca'], dtype=np.float64)
        S = b.obs[['x_centroid','y_centroid']].to_numpy(np.float64)
        S_t, A_t = neural_net.load_rescale_input_data(S, A)
        best = (np.inf, None)
        for sd in range(args.seeds):
            neural_net.set_seeds(sd)
            mdl, loss = neural_net.train(S_t, A_t, S_hidden_list=[20,20], A_hidden_list=[20,20],
                                         epochs=args.epochs, checkpoint=args.epochs,
                                         optim='adam', lr=1e-3, seed=sd, device=dev)
            fl = float(np.asarray(loss)[-1]) if np.ndim(loss) else float(loss)
            print(f'  {sec} seed {sd} loss {fl:.5f}', flush=True)
            if fl < best[0]: best = (fl, mdl)
        mdl = best[1].to('cpu').eval()
        with torch.no_grad():
            iso = mdl.spatial_embedding(S_t.cpu().float()).numpy().ravel()
        lab, L, nb = segment(A, iso)
        if lab is None:
            print(f'  {sec}: DP failed for every L'); continue
        ct = b.obs.celltype.astype(str).to_numpy(); is_t = np.isin(ct, TUM)
        nn = NearestNeighbors(n_neighbors=21).fit(S).kneighbors(S, return_distance=False)[:,1:]
        lfrac = is_t[nn].mean(1)
        mp = bio.reindex(b.obs['cell_id'].astype(str))['MAPK_MPAS'].to_numpy()
        ok = np.isfinite(mp)
        r_lf = spearmanr(iso, lfrac).statistic
        r_mp = spearmanr(iso[ok], mp[ok]).statistic if ok.sum() > 100 else np.nan
        res[sec] = dict(n_domains=int(L), num_buckets=int(nb), final_loss=best[0],
                        rho_isodepth_local_tumor_frac=float(r_lf),
                        rho_isodepth_MAPK=float(r_mp), n_tumor=int(ok.sum()))
        print(f'  {sec}: L={L} | rho(iso,tumourfrac) {r_lf:+.3f} | rho(iso,MAPK) {r_mp:+.3f}', flush=True)
        frames.append(pd.DataFrame(dict(cell_id=b.obs.cid.to_numpy(), slice_id=sec,
                                        x_centroid=S[:,0], y_centroid=S[:,1], celltype=ct,
                                        isodepth=iso, domain=lab, local_tumor_frac=lfrac,
                                        MAPK_MPAS=mp)))
        np.savez(args.out_dir/f'gaston_genes_{sec}.npz', isodepth=iso, labels=lab,
                 cell_id=b.obs.cid.to_numpy(), xy=S)

    # expression of the top domain markers vs isodepth, for the piecewise-fit panel
    cells = pd.concat(frames, ignore_index=True)
    cells.to_csv(args.out_dir/'gaston_genes_cells.csv.gz', index=False, compression='gzip')
    (args.out_dir/'gaston_genes_summary.json').write_text(json.dumps(res, indent=2)+'\n')

    print('\n=== domain annotation (pooled) ===', flush=True)
    _save_expr = True
    a.obs['domain'] = pd.Categorical(
        pd.Series(cells.set_index('cell_id')['domain']).reindex(a.obs.cid).astype('Int64').astype(str))
    a2 = a[a.obs.domain != '<NA>'].copy()
    sc.tl.rank_genes_groups(a2, 'domain', method='wilcoxon', n_genes=30)
    mk = sc.get.rank_genes_groups_df(a2, None)
    mk.to_csv(args.out_dir/'gaston_genes_domain_markers.csv', index=False)
    comp = pd.crosstab(a2.obs.domain, a2.obs.celltype.astype(str), normalize='index')*100
    comp.to_csv(args.out_dir/'gaston_genes_domain_celltype.csv')
    for g in sorted(mk.group.unique(), key=int):
        t = mk[(mk.group==g)&(mk.pvals_adj<0.05)].nlargest(8,'logfoldchanges')
        top = comp.loc[g].nlargest(3)
        print(f'  domain {g} (n={(a2.obs.domain==g).sum():6d}): '
              + ', '.join(t.names.tolist()), flush=True)
        print(f'            cell types: ' + ', '.join(f'{k} {v:.0f}%' for k,v in top.items()), flush=True)
    if _save_expr:
        # expression of the top per-domain markers vs isodepth, for the piecewise-fit panel
        top = (mk[mk.pvals_adj < 0.05].sort_values('logfoldchanges', ascending=False)
               .drop_duplicates('group').nlargest(6, 'logfoldchanges').names.tolist())
        keep = [g for g in top if g in a2.var_names]
        X = a2[:, keep].X
        X = np.asarray(X.todense()) if hasattr(X, 'todense') else np.asarray(X)
        E = pd.DataFrame(X, columns=keep)
        by_cell = cells.set_index('cell_id')
        cid_key = a2.obs.cid.astype(str)   # cells['cell_id'] holds the obs index
        E['slice_id'] = a2.obs.sec.to_numpy()
        E['isodepth'] = by_cell['isodepth'].reindex(cid_key).to_numpy()
        E['domain'] = by_cell['domain'].reindex(cid_key).to_numpy()
        E.to_csv(args.out_dir / 'gaston_genes_expression.csv.gz', index=False, compression='gzip')
        print(f'  saved expression table for {keep}', flush=True)
    print('\nwrote', args.out_dir)


if __name__ == '__main__':
    main()
