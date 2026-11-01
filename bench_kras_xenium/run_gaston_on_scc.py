#!/usr/bin/env python3
"""GASTON isodepth on the KRAS salient-CC embedding.

GASTON (Chitra et al., raphael-group) learns a 1-D 'isodepth' scalar field
S(x,y) together with an expression function of that field, giving a topographic
map of the tissue plus a domain segmentation as level sets of isodepth.

Here the input is the SALIENT-CC embedding (its top PCs) rather than gene PCs, so
the isodepth formalises the salient spatial gradient we characterised: a
tumour-depth coordinate.  Three things are then asked of it:
  (a) what does isodepth correspond to  -> local tumour fraction, cell type
  (b) what marks each domain            -> per-domain gene markers from REAL counts
  (c) does drug response vary along it  -> MAPK MPAS vs isodepth  <-- the real test
(c) is the point: a drug-penetration / depth gradient in response would be a
genuine spatial-pharmacology result, and is not something (a) or (b) can fake.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scanpy as sc, torch
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from gaston import neural_net, dp_related

CC = ('/home/yancui/haruka_v2/results_real_compgated_nb_final/mouselung_kras_xenium_clean_noC3/'
      'haruka_z_s_cellcharter_cc3_condition_only/')
BIO = '/home/yancui/haruka_v2/bench_kras_xenium/biological_responder/'
TUM = ['Tumor_EMT', 'Tumor_epithelial', 'Tumor_epithelial_Itgb6']


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', type=Path,
                   default=Path('/home/yancui/haruka_v2/bench_kras_xenium/gaston_scc'))
    p.add_argument('--n-cells', type=int, default=30000)
    p.add_argument('--n-pcs', type=int, default=15)
    p.add_argument('--domains', type=int, default=5)
    p.add_argument('--epochs', type=int, default=4000)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--rep', default='salient_spatial.npy')
    args = p.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device {dev}', flush=True)

    h = ad.read_h5ad('/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad', backed='r')
    rows = np.load(CC + 'source_rows.npy')
    obs = h.obs.iloc[rows][['slice_id','celltype','x_centroid','y_centroid','cell_id']].copy()
    obs['cell_id'] = obs['cell_id'].astype(str)
    rep = np.load(CC + args.rep)
    print(f'treated cells {len(obs)}, rep {rep.shape}', flush=True)

    bio = pd.read_csv(BIO + 'treated_tumor_biological_response.csv.gz').set_index('cell_id')
    results, per_cell = {}, []
    for sec in sorted(obs.slice_id.astype(str).unique()):
        m = (obs.slice_id.astype(str) == sec).to_numpy()
        idx = np.where(m)[0]
        rng = np.random.default_rng(0)
        take = np.sort(rng.choice(idx, min(args.n_cells, len(idx)), replace=False))
        S = obs.iloc[take][['x_centroid','y_centroid']].to_numpy(np.float64)
        A = PCA(n_components=args.n_pcs, random_state=0).fit_transform(
            StandardScaler().fit_transform(rep[take]))
        S_t, A_t = neural_net.load_rescale_input_data(S, A)
        best = (np.inf, None)
        for sd in range(args.seeds):
            neural_net.set_seeds(sd)
            mdl, loss = neural_net.train(S_t, A_t, S_hidden_list=[20,20], A_hidden_list=[20,20],
                                         epochs=args.epochs, checkpoint=args.epochs,
                                         optim='adam', lr=1e-3, seed=sd, device=dev)
            fl = float(np.asarray(loss)[-1]) if np.ndim(loss) else float(loss)
            print(f'  {sec} seed {sd} final loss {fl:.5f}', flush=True)
            if fl < best[0]: best = (fl, mdl)
        mdl = best[1].to('cpu').eval()      # tensors from load_rescale_input_data are on CPU
        with torch.no_grad():
            iso = mdl.spatial_embedding(S_t.cpu().float()).detach().numpy().ravel()
        # get_isodepth_labels returns (isodepth, labels) -- labels is [1].
        # Its DP can fail (KeyError) when many isodepth buckets are empty, so
        # step the bucket count down, then fall back to quantile bins.
        lab = None
        for nb_ in (50, 30, 20, 15, 10):
            try:
                seg = dp_related.get_isodepth_labels(mdl, A_t, S_t, args.domains, num_buckets=nb_)
                lab = np.asarray(seg[1]).ravel()
                print(f'    segmented with num_buckets={nb_}', flush=True)
                break
            except Exception as e:
                print(f'    segmentation failed at num_buckets={nb_}: {type(e).__name__} {e}', flush=True)
        if lab is None:
            q = np.quantile(iso, np.linspace(0, 1, args.domains + 1)[1:-1])
            lab = np.digitize(iso, q).astype(float)
            print('    FALLBACK: equal-count isodepth bins (DP failed at every bucket count)', flush=True)
        np.savez(args.out_dir / f'gaston_{sec}.npz', isodepth=iso, labels=lab,
                 xy=S, cell_id=obs.iloc[take].cell_id.to_numpy())

        sub = obs.iloc[take].copy()
        sub['isodepth'] = iso; sub['domain'] = lab
        ct = sub.celltype.astype(str).to_numpy(); is_t = np.isin(ct, TUM)
        nb = NearestNeighbors(n_neighbors=21).fit(S).kneighbors(S, return_distance=False)[:,1:]
        sub['local_tumor_frac'] = is_t[nb].mean(1)
        sub['MAPK_MPAS'] = bio.reindex(sub.cell_id)['MAPK_MPAS'].to_numpy()
        per_cell.append(sub)

        rho_lf = spearmanr(sub.isodepth, sub.local_tumor_frac).statistic
        tm = np.isfinite(sub.MAPK_MPAS.to_numpy())
        rho_mp = spearmanr(sub.isodepth[tm], sub.MAPK_MPAS[tm]).statistic if tm.sum() > 100 else np.nan
        results[sec] = dict(n=int(len(sub)), final_loss=best[0],
                            rho_isodepth_local_tumor_frac=float(rho_lf),
                            rho_isodepth_MAPK=float(rho_mp), n_tumor_scored=int(tm.sum()),
                            domain_sizes={int(k): int(v) for k, v in
                                          pd.Series(lab).value_counts().sort_index().items()})
        print(f'  {sec}: rho(isodepth, local tumour frac) {rho_lf:+.3f} | '
              f'rho(isodepth, MAPK MPAS) {rho_mp:+.3f} (n={tm.sum()})', flush=True)

    allc = pd.concat(per_cell, ignore_index=True)
    allc.to_csv(args.out_dir/'gaston_isodepth_cells.csv.gz', index=False, compression='gzip')
    (args.out_dir/'gaston_summary.json').write_text(json.dumps(results, indent=2)+'\n')

    print('\n=== per-domain gene markers (real counts, pooled over sections) ===', flush=True)
    sel = h.obs['cell_id'].astype(str).isin(allc.cell_id).to_numpy()
    a = h[sel].to_memory(); h.file.close()
    a.obs['cid'] = a.obs['cell_id'].astype(str)
    dom = allc.set_index('cell_id')['domain'].reindex(a.obs.cid).to_numpy()
    a = a[np.isfinite(dom)].copy(); dom = dom[np.isfinite(dom)]
    a.X = a.layers['counts'].copy(); sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    a.obs['domain'] = pd.Categorical(dom.astype(int).astype(str))
    sc.tl.rank_genes_groups(a, 'domain', method='wilcoxon', n_genes=25)
    mk = sc.get.rank_genes_groups_df(a, None)
    mk.to_csv(args.out_dir/'domain_markers.csv', index=False)
    for g in sorted(mk.group.unique(), key=int):
        top = mk[mk.group == g].nlargest(8, 'logfoldchanges')
        print(f'  domain {g}: ' + ', '.join(top.names.tolist()), flush=True)
    print(f'\nwrote {args.out_dir}')


if __name__ == '__main__':
    main()
