"""Target engagement + model-free neighbourhood effects + cross-chip replication.
Usage: python -I 03_knockdown_neighbourhood.py <annot_dir> <out_dir>
"""
import sys, os, json
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy import stats
from scipy.spatial import cKDTree
import anndata as ad
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ind, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
CHIPS = ['C02943C3', 'B03018A2', 'A03599E2']
TYPES = ['ExN', 'InN', 'Astro', 'Oligo', 'OPC', 'Micro', 'Endo', 'ChP']
COMP_TYPES = TYPES + ['LowQ']
CLAIMS = {'Lrrk2': {'Sparc': 'less', 'Vps35l': 'greater', 'Dock10': 'greater', 'Gpr37': 'greater'},
          'Srf': {'Gad1': 'greater', 'Arhgap12': 'two-sided', 'Ssrp1': 'two-sided'}}
RNG = np.random.default_rng(0)
NPERM = 20000
MINN = 5  # min single-barcode cells per KO per chip to enter profiles


def stouffer(ps, ws):
    ps, ws = np.asarray(ps, float), np.asarray(ws, float)
    ok = np.isfinite(ps)
    if ok.sum() == 0:
        return np.nan, np.nan
    z = stats.norm.isf(np.clip(ps[ok], 1e-300, 1 - 1e-16))
    Z = (ws[ok] * z).sum() / np.sqrt((ws[ok] ** 2).sum())
    return float(Z), float(stats.norm.sf(Z))


ZMIN = 0.5  # clusters whose best marker z < ZMIN are called LowQ (all-negative clusters otherwise get argmax label)


def load(chip):
    a = ad.read_h5ad(os.path.join(ind, f'{chip}.annot.h5ad'))
    cz = pd.read_csv(os.path.join(ind, f'{chip}_cluster_marker_z.csv'), index_col=0)
    zmax = cz[TYPES].max(1)
    lab = cz['label'].where(zmax >= ZMIN, 'LowQ')
    lab.index = lab.index.astype(str)
    a.obs['cell_type_orig'] = a.obs['cell_type']
    a.obs['cell_type'] = a.obs['leiden'].astype(str).map(lab).values
    bcs = list(a.uns['barcode_features'])
    endo = ~a.var_names.isin(bcs)
    C = a.layers['counts'][:, endo].tocsr().astype(np.float32)
    lib = np.asarray(C.sum(1)).ravel()
    L = sp.diags(1e4 / np.maximum(lib, 1)) @ C
    L.data = np.log1p(L.data)
    return a.obs.copy(), a.var_names[endo], C, L.tocsr(), lib, a.obsm['spatial'] * 0.5


chipdata = {c: load(c) for c in CHIPS}
common_genes = None
for c in CHIPS:
    g = set(chipdata[c][1])
    common_genes = g if common_genes is None else common_genes & g

# ---------------- 1. target engagement ----------------
ke_rows, glm_rows = [], []
KOS = sorted(set().union(*[set(chipdata[c][0].perturbation) for c in CHIPS]) - {'none', 'multi', 'mSafe'})
for ko in KOS:
    for c in CHIPS:
        obs, genes, C, L, lib, xy = chipdata[c]
        if ko not in set(genes):
            continue
        gi = int(np.where(genes == ko)[0][0])
        x_cnt = np.asarray(C[:, gi].todense()).ravel()
        x_log = np.asarray(L[:, gi].todense()).ravel()
        for ref in ['mSafe', 'none']:
            for t in TYPES:
                m_ko = ((obs.perturbation == ko) & (obs.cell_type == t)).values
                m_rf = ((obs.perturbation == ref) & (obs.cell_type == t)).values
                if m_ko.sum() < 3 or m_rf.sum() < 3:
                    continue
                p = stats.mannwhitneyu(x_log[m_ko], x_log[m_rf], alternative='less').pvalue
                ke_rows.append(dict(ko=ko, chip=c, ref=ref, cell_type=t, n_ko=int(m_ko.sum()), n_ref=int(m_rf.sum()),
                                    mean_cp10k_ko=float(np.expm1(x_log[m_ko]).mean()),
                                    mean_cp10k_ref=float(np.expm1(x_log[m_rf]).mean()),
                                    det_ko=float((x_cnt[m_ko] > 0).mean()), det_ref=float((x_cnt[m_rf] > 0).mean()),
                                    p_less=float(p)))
            # Poisson GLM with offset, cell type covariate, robust SE (one-sided KO<ref)
            m = ((obs.perturbation == ko) | (obs.perturbation == ref)).values
            if ref == 'none':  # subsample barcode-negative cells for speed
                idx_ref = np.where((obs.perturbation == 'none').values)[0]
                idx_ref = RNG.choice(idx_ref, size=min(20000, len(idx_ref)), replace=False)
                m = np.zeros(len(obs), bool); m[idx_ref] = True; m[(obs.perturbation == ko).values] = True
            df = pd.DataFrame({'y': x_cnt[m], 'ko': (obs.perturbation.values[m] == ko).astype(float),
                               'ct': obs.cell_type.values[m], 'off': np.log(np.maximum(lib[m], 1))})
            if df.ko.sum() < 5 or (1 - df.ko).sum() < 5 or df.y.sum() < 5:
                continue
            X = pd.get_dummies(df.ct, drop_first=True, dtype=float)
            X.insert(0, 'ko', df.ko)
            X = sm.add_constant(X)
            try:
                r = sm.GLM(df.y, X, family=sm.families.Poisson(), offset=df.off).fit(cov_type='HC0')
                b, se = r.params['ko'], r.bse['ko']
                glm_rows.append(dict(ko=ko, chip=c, ref=ref, n_ko=int(df.ko.sum()), n_ref=int((1 - df.ko).sum()),
                                     log_rr=float(b), se=float(se), p_less=float(stats.norm.cdf(b / se))))
            except Exception as e:
                glm_rows.append(dict(ko=ko, chip=c, ref=ref, err=str(e)))
ke = pd.DataFrame(ke_rows); ke.to_csv(os.path.join(out, 'knockdown_strata.csv'), index=False)
glm = pd.DataFrame(glm_rows); glm.to_csv(os.path.join(out, 'knockdown_glm_per_chip.csv'), index=False)
comb = []
for (ko, ref), g in ke.groupby(['ko', 'ref']):
    w = np.sqrt(g.n_ko * g.n_ref / (g.n_ko + g.n_ref))
    Z, p = stouffer(g.p_less, w)
    gg = glm[(glm.ko == ko) & (glm.ref == ref) & glm.se.notna()] if 'se' in glm else glm.iloc[:0]
    if len(gg):
        wi = 1 / gg.se ** 2
        b = float((wi * gg.log_rr).sum() / wi.sum()); se = float(1 / np.sqrt(wi.sum()))
        pg = float(stats.norm.cdf(b / se))
    else:
        b = se = pg = np.nan
    comb.append(dict(ko=ko, ref=ref, n_ko=int(g.groupby('chip').n_ko.sum().sum()), strata=len(g),
                     stouffer_Z=Z, p_mwu_stouffer=p, glm_ivw_log_rr=b, glm_rr=float(np.exp(b)) if np.isfinite(b) else np.nan,
                     glm_se=se, p_glm_ivw=pg))
comb = pd.DataFrame(comb)
for ref in ['mSafe', 'none']:
    m = comb.ref == ref
    comb.loc[m, 'q_mwu'] = multipletests(comb.loc[m, 'p_mwu_stouffer'].fillna(1), method='fdr_bh')[1]
    comb.loc[m, 'q_glm'] = multipletests(comb.loc[m, 'p_glm_ivw'].fillna(1), method='fdr_bh')[1]
comb.to_csv(os.path.join(out, 'knockdown_combined.csv'), index=False)
print(comb.to_string(), flush=True)

# ---------------- 2. neighbourhood ----------------
# profiles for single-barcode cells: neighbour composition, claimed-gene neighbour expr, genome-wide panel
claim_genes = sorted({g for d in CLAIMS.values() for g in d})
# genome-wide panel: genes in all chips with decent detection (top 500 by mean in first chip)
obs0, genes0, C0, L0, _, _ = chipdata[CHIPS[0]]
mu0 = pd.Series(np.asarray(L0.mean(0)).ravel(), index=genes0)
panel = [g for g in mu0.sort_values(ascending=False).index if g in common_genes][:500]

res_store = {}
for k in [15, 20]:
    prof = {}
    for c in CHIPS:
        obs, genes, C, L, lib, xy = chipdata[c]
        gidx = pd.Series(np.arange(len(genes)), index=genes)
        cells = np.where(~obs.perturbation.isin(['none', 'multi']).values)[0]
        nn = cKDTree(xy).query(xy[cells], k=k + 1)[1][:, 1:]
        ct = obs.cell_type.values
        comp = np.stack([(ct[nn] == t).mean(1) for t in COMP_TYPES], 1)
        W = sp.csr_matrix((np.full(nn.size, 1.0 / k), nn.ravel(), np.arange(0, nn.size + 1, k)),
                          shape=(len(cells), len(obs)))
        Lc = L[:, gidx[claim_genes].values].toarray()
        Lp = L[:, gidx[panel].values]
        nb_claim = W @ Lc
        nb_panel = np.asarray((W @ Lp).todense())
        own_target = np.full(len(cells), np.nan)
        pert = obs.perturbation.values[cells]
        for i, p in enumerate(pert):
            if p in gidx:
                own_target[i] = L[cells[i], gidx[p]]
        df = pd.DataFrame(comp, columns=[f'comp_{t}' for t in COMP_TYPES])
        df[[f'nbx_{g}' for g in claim_genes]] = nb_claim
        P = pd.DataFrame(nb_panel, columns=[f'nbp_{g}' for g in panel])
        df = pd.concat([df, P], axis=1)
        df['own_type'] = ct[cells]
        df['pert'] = pert
        df['own_target'] = own_target
        prof[c] = df
    res_store[k] = prof

feat_sets = {'composition': [f'comp_{t}' for t in COMP_TYPES], 'claimed_genes': [f'nbx_{g}' for g in claim_genes],
             'panel500': [f'nbp_{g}' for g in panel]}


def residualise(df, cols):
    """subtract own-type mean (over all single-barcode cells of that type in the chip) -> type-matched"""
    r = df[cols] - df.groupby('own_type')[cols].transform('mean')
    return r


summary = {}
pert_rows = []
for k, prof in res_store.items():
    for fs, cols in feat_sets.items():
        Ds, Ts = {}, {}
        for c in CHIPS:
            df = prof[c]
            r = residualise(df, cols); r['pert'] = df.pert.values
            g = r.groupby('pert')
            M = g[cols].mean(); S = g[cols].std(ddof=1); N = g.size()
            keep = N[N >= MINN].index
            M, S, N = M.loc[keep], S.loc[keep], N.loc[keep]
            D = M - M.mean(0)  # deviation from mean over KOs (incl. mSafe)
            T = D / (S / np.sqrt(N.values[:, None])).replace(0, np.nan)
            Ds[c], Ts[c] = D, T
            # KO vs mSafe tests (type-matched residuals), per feature
            if fs != 'panel500' and 'mSafe' in keep:
                for ko in keep:
                    if ko == 'mSafe':
                        continue
                    a_ = r.loc[r.pert == ko, cols]; b_ = r.loc[r.pert == 'mSafe', cols]
                    for col in cols:
                        alt = 'two-sided'
                        gname = col.split('_', 1)[1]
                        if fs == 'claimed_genes' and ko in CLAIMS and gname in CLAIMS[ko]:
                            alt = CLAIMS[ko][gname]
                        p = stats.mannwhitneyu(a_[col], b_[col], alternative=alt).pvalue
                        pert_rows.append(dict(k=k, featset=fs, chip=c, ko=ko, feature=gname, alt=alt,
                                              n_ko=len(a_), n_msafe=len(b_), delta=float(a_[col].mean() - b_[col].mean()), p=p))
        # replication across chips
        kos = sorted(set.intersection(*[set(Ds[c].index) for c in CHIPS]))
        out_fs = {}
        for name, Mats in [('raw', Ds), ('t', Ts)]:
            A = [Mats[c].loc[kos, cols].values for c in CHIPS]
            def stat(As):
                rs = []
                for i in range(3):
                    for j in range(i + 1, 3):
                        x, y = As[i].ravel(), As[j].ravel()
                        ok = np.isfinite(x) & np.isfinite(y)
                        rs.append(stats.pearsonr(x[ok], y[ok])[0])
                return np.array(rs)
            obs_r = stat(A)
            null = np.empty((NPERM, 3))
            for b in range(NPERM):
                null[b] = stat([A[0], A[1][RNG.permutation(len(kos))], A[2][RNG.permutation(len(kos))]])
            pm = float((1 + (null.mean(1) >= obs_r.mean()).sum()) / (NPERM + 1))
            pp = [float((1 + (null[:, i] >= obs_r[i]).sum()) / (NPERM + 1)) for i in range(3)]
            out_fs[name] = dict(n_kos=len(kos), r_pairs=dict(zip(['C0-B0', 'C0-A0', 'B0-A0'], np.round(obs_r, 4).tolist())),
                                r_mean=float(obs_r.mean()), p_perm_mean=pm, p_perm_pairs=pp)
        summary[f'k{k}_{fs}'] = out_fs
        print(k, fs, json.dumps(out_fs), flush=True)

# own-cell target positive-control replication (only diagonal: KO cell's own target gene, residualised by type)
for k in [15]:
    vals = {}
    for c in CHIPS:
        df = res_store[k][c].copy()
        df = df[df.own_target.notna()]
        df['r'] = df.own_target - df.groupby('own_type').own_target.transform('mean')
        vals[c] = df.groupby('pert').r.mean()
    kos = sorted(set.intersection(*[set(v.index) for v in vals.values()]))
    summary['own_target_residual_per_ko'] = {c: vals[c].loc[kos].round(4).to_dict() for c in CHIPS}

pr = pd.DataFrame(pert_rows)
comb = []
for (k, fs, ko, f, alt), g in pr.groupby(['k', 'featset', 'ko', 'feature', 'alt']):
    # signed Stouffer for two-sided: convert to one-sided via sign of delta
    if alt == 'two-sided':
        z = np.sign(g.delta) * stats.norm.isf(g.p / 2)
    else:
        z = stats.norm.isf(np.clip(g.p, 1e-300, 1 - 1e-16))
    w = np.sqrt(g.n_ko * g.n_msafe / (g.n_ko + g.n_msafe))
    Z = float((w * z).sum() / np.sqrt((w ** 2).sum()))
    p = float(2 * stats.norm.sf(abs(Z))) if alt == 'two-sided' else float(stats.norm.sf(Z))
    comb.append(dict(k=k, featset=fs, ko=ko, feature=f, alt=alt, n_chips=len(g), n_ko=int(g.n_ko.sum()),
                     deltas=';'.join(f'{d:+.4f}' for d in g.delta), same_sign_all=bool(len(set(np.sign(g.delta))) == 1),
                     Z=Z, p=p))
comb = pd.DataFrame(comb)
for (k, fs), idx in comb.groupby(['k', 'featset']).groups.items():
    comb.loc[idx, 'q_bh'] = multipletests(comb.loc[idx, 'p'], method='fdr_bh')[1]
comb.to_csv(os.path.join(out, 'neighbour_ko_vs_msafe_combined.csv'), index=False)
pr.to_csv(os.path.join(out, 'neighbour_ko_vs_msafe_per_chip.csv'), index=False)
print(comb[(comb.featset == 'claimed_genes') & comb.ko.isin(CLAIMS) & comb.apply(lambda r: r.feature in CLAIMS.get(r.ko, {}), axis=1)].to_string())
print(comb[comb.featset == 'composition'].sort_values('p').head(20).to_string())
json.dump(summary, open(os.path.join(out, 'replication_summary.json'), 'w'), indent=1, default=float)
