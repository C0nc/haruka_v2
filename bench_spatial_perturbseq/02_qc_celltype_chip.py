"""QC, barcode calling, spatial sparsity, cell-type annotation per chip.
Usage: python -I 02_qc_celltype.py <h5ad_dir> <out_dir>
Coordinates are in DNB units (GEF resolution 500 nm) -> um = coord * 0.5
"""
import sys, os, json
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
from scipy.spatial import cKDTree

hd, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
CHIPS = sys.argv[3].split(',') if len(sys.argv) > 3 else ['C02943C3', 'B03018A2', 'A03599E2']
import time; T0 = time.time()
def tick(m): print(f'[{time.time()-T0:7.0f}s] {m}', flush=True)
BC2GENE = {'sgrna_C9orf72': 'C9orf72', 'sgrna_Cfap410': 'Cfap410', 'sgrna_clu': 'Clu', 'sgrna_dpp5': 'Dpp6',
           'sgrna_fasn': 'Fasn', 'sgrna_flcn': 'Flcn', 'sgrna_gfap': 'Gfap', 'sgrna_lrrk2': 'Lrrk2',
           'sgrna_msafe': 'mSafe', 'sgrna_ndufaf': 'Ndufaf2', 'sgrna_oligo2': 'Olig2', 'sgrna_rbfox': 'Rbfox3',
           'sgrna_rraga': 'Rraga', 'sgrna_sh3gl2': 'Sh3gl2', 'sgrna_srf': 'Srf', 'sgrna_stk39': 'Stk39',
           'sgrna_tbk1': 'Tbk1', 'sgrna_trem2': 'Trem2'}
# targets (Gfap, Olig2, Rbfox3, Clu) deliberately NOT used as markers
MARKERS = {
    'ExN': ['Slc17a7', 'Neurod6', 'Satb2', 'Camk2a', 'Nrgn', 'Prox1', 'Tbr1'],
    'InN': ['Gad1', 'Gad2', 'Slc32a1', 'Sst', 'Pvalb', 'Vip'],
    'Astro': ['Aqp4', 'Slc1a3', 'Aldh1l1', 'Gja1', 'Slc1a2', 'Atp1a2'],
    'Oligo': ['Mbp', 'Plp1', 'Mobp', 'Mog', 'Cldn11', 'Mal'],
    'OPC': ['Pdgfra', 'Cspg4', 'Vcan', 'Gpr17'],
    'Micro': ['Cx3cr1', 'P2ry12', 'Tmem119', 'Csf1r', 'Hexb', 'C1qa'],
    'Endo': ['Cldn5', 'Flt1', 'Pecam1', 'Slco1a4', 'Ly6c1'],
    'ChP': ['Ttr', 'Kl', 'Folr1', 'Clic6'],
}
summ = {}
bc_rows = []
for chip in CHIPS:
    a = sc.read_h5ad(os.path.join(hd, f'{chip}.h5ad'))
    a = a[:, a.var_names != ''].copy()
    a = a[:, ~a.var_names.str.match(r'^-\d+$')].copy()  # empty names made unique
    bcs = [b for b in BC2GENE if b in a.var_names]
    B = a[:, bcs].layers['counts']
    B = B.toarray() if sp.issparse(B) else B
    nbc = (B > 0).sum(1)
    a.obs['n_barcodes'] = nbc
    a.obs['bc_umis'] = B.sum(1)
    a.obs['perturbation'] = 'none'
    single = nbc == 1
    a.obs.loc[single, 'perturbation'] = [BC2GENE[bcs[j]] for j in B[single].argmax(1)]
    a.obs.loc[nbc > 1, 'perturbation'] = 'multi'
    a.uns['barcode_features'] = bcs
    # gene QC on endogenous genes only
    endo = ~a.var_names.isin(bcs)
    C = a.layers['counts'][:, endo]
    a.obs['n_genes_endo'] = np.asarray((C > 0).sum(1)).ravel()
    a.obs['umi_endo'] = np.asarray(C.sum(1)).ravel()
    # ---- spatial sparsity: NN distance among single-barcode cells vs random cells
    xy = a.obsm['spatial'] * 0.5  # um
    pos = np.where(a.obs['perturbation'].values != 'none')[0]
    pos_s = np.where(~a.obs['perturbation'].isin(['none', 'multi']).values)[0]
    tr_all = cKDTree(xy)
    d_all = tr_all.query(xy, k=2)[0][:, 1]
    trp = cKDTree(xy[pos])
    dpp = trp.query(xy[pos], k=2)[0][:, 1]
    # how many of the 15/20 NN of a perturbed cell are barcode+ ?
    nn20 = tr_all.query(xy[pos_s], k=21)[1][:, 1:]
    isbc = (a.obs['perturbation'].values != 'none')
    frac_bc_nn20 = isbc[nn20].mean()
    # random expectation under same positions: fraction of barcode+ cells overall
    rng = np.random.default_rng(0)
    rnd = rng.choice(len(xy), size=len(pos), replace=False)
    d_rnd = cKDTree(xy[rnd]).query(xy[rnd], k=2)[0][:, 1]
    # ---- cell typing
    sc.pp.filter_cells(a, min_counts=1)
    b = a[:, endo].copy()
    b.X = b.layers['counts'].copy()
    sc.pp.normalize_total(b, target_sum=1e4)
    sc.pp.log1p(b)
    tick('hvg'); sc.pp.highly_variable_genes(b, n_top_genes=3000, flavor='seurat')
    bh = b[:, b.var.highly_variable].copy()
    sc.pp.scale(bh, max_value=10)
    tick('pca'); sc.tl.pca(bh, n_comps=30)
    tick('nbrs'); sc.pp.neighbors(bh, n_neighbors=15)
    tick('leiden'); sc.tl.leiden(bh, resolution=1.5, flavor='igraph', n_iterations=2, key_added='leiden')
    for t, g in MARKERS.items():
        gg = [x for x in g if x in b.var_names]
        sc.tl.score_genes(b, gg, score_name=f'sc_{t}', random_state=0)
    S = b.obs[[f'sc_{t}' for t in MARKERS]].copy()
    S.columns = list(MARKERS)
    S['leiden'] = bh.obs['leiden'].values
    cm = S.groupby('leiden').mean()
    cz = (cm - cm.mean()) / cm.std()
    lab = cz.idxmax(1)
    a.obs['leiden'] = bh.obs['leiden'].values
    a.obs['cell_type'] = a.obs['leiden'].map(lab).astype(str).values
    a.obs['cell_type_cellscore'] = S[list(MARKERS)].idxmax(1).values
    for t in MARKERS:
        a.obs[f'sc_{t}'] = S[t].values
    cz.assign(label=lab, n=S.groupby('leiden').size()).to_csv(os.path.join(out, f'{chip}_cluster_marker_z.csv'))
    # barcode table
    ct = pd.crosstab(a.obs['perturbation'], a.obs['cell_type'])
    ct.to_csv(os.path.join(out, f'{chip}_perturbation_x_celltype.csv'))
    for p, n in a.obs['perturbation'].value_counts().items():
        bc_rows.append(dict(chip=chip, perturbation=p, n=int(n)))
    summ[chip] = dict(
        n_cells=int(a.n_obs), median_umi=float(np.median(a.obs['umi_endo'])),
        median_genes=float(np.median(a.obs['n_genes_endo'])),
        n_barcode_pos=int(len(pos)), frac_barcode_pos=float(len(pos) / a.n_obs),
        n_single=int(len(pos_s)), n_multi=int((nbc > 1).sum()),
        frac_multi_of_pos=float((nbc > 1).sum() / max(len(pos), 1)),
        bc_umis_single_median=float(np.median(a.obs.loc[a.obs.n_barcodes == 1, 'bc_umis'])),
        frac_single_with_1umi=float((a.obs.loc[a.obs.n_barcodes == 1, 'bc_umis'] == 1).mean()),
        nn_dist_any_cell_um_median=float(np.median(d_all)),
        nn_dist_bcpos_to_bcpos_um_median=float(np.median(dpp)),
        nn_dist_random_same_n_um_median=float(np.median(d_rnd)),
        frac_bcpos_among_20nn_of_perturbed=float(frac_bc_nn20),
        celltype_counts=a.obs['cell_type'].value_counts().to_dict(),
        celltype_of_single_bc=a.obs.loc[~a.obs.perturbation.isin(['none', 'multi']), 'cell_type'].value_counts().to_dict(),
        celltype_agreement_cluster_vs_cell=float((a.obs.cell_type == a.obs.cell_type_cellscore).mean()),
    )
    print(chip, json.dumps(summ[chip], indent=1), flush=True)
    a.write_h5ad(os.path.join(out, f'{chip}.annot.h5ad'), compression='gzip')
pd.DataFrame(bc_rows).pivot(index='perturbation', columns='chip', values='n').fillna(0).astype(int)\
    .to_csv(os.path.join(out, 'barcode_counts_per_chip.csv'))
json.dump(summ, open(os.path.join(out, 'qc_summary.json'), 'w'), indent=1)
