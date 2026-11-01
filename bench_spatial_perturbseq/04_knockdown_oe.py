"""Target engagement via observed/expected counts (exact Poisson, conditional on library size).
For KO g, chip c: O = sum of g-counts in g-KO cells; E = sum_i lib_i * rate_ref(g | own type, chip).
rate_ref from (a) barcode-negative cells, (b) mSafe cells pooled over chips (type-matched).
One-sided Poisson test P(X <= O | E). Calibration: the same O/E for each KO computed on the
OTHER 16 target genes (off-target controls) -> empirical rank of the on-target O/E.
Usage: python -I 04_knockdown_oe.py <annot_dir> <out_dir>
"""
import sys, os
import numpy as np, pandas as pd, anndata as ad
from scipy import stats

ind, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
CHIPS = ['C02943C3', 'B03018A2', 'A03599E2']
TYPES = ['ExN', 'InN', 'Astro', 'Oligo', 'OPC', 'Micro', 'Endo', 'ChP']
TARGETS = ['C9orf72', 'Cfap410', 'Clu', 'Dpp6', 'Fasn', 'Flcn', 'Gfap', 'Lrrk2', 'Ndufaf2', 'Olig2', 'Rbfox3',
           'Rraga', 'Sh3gl2', 'Srf', 'Stk39', 'Tbk1', 'Trem2']
ZMIN = 0.5
rows = []
store = {}
for c in CHIPS:
    a = ad.read_h5ad(os.path.join(ind, f'{c}.annot.h5ad'))
    cz = pd.read_csv(os.path.join(ind, f'{c}_cluster_marker_z.csv'), index_col=0)
    lab = cz['label'].where(cz[TYPES].max(1) >= ZMIN, 'LowQ'); lab.index = lab.index.astype(str)
    ct = a.obs['leiden'].astype(str).map(lab).values
    lib = a.obs['umi_endo'].values.astype(float)
    Y = pd.DataFrame(a[:, TARGETS].layers['counts'].toarray(), columns=TARGETS)
    store[c] = (a.obs['perturbation'].values, ct, lib, Y)

def rates(mask_fn):
    """per (chip,type) rate = sum counts / sum lib over reference cells; returns dict[(chip,type)] -> Series over genes"""
    R = {}
    for c, (pert, ct, lib, Y) in store.items():
        for t in set(ct):
            m = mask_fn(pert) & (ct == t)
            R[(c, t)] = (Y[m].sum(0), lib[m].sum(), int(m.sum()))
    return R

R_neg = rates(lambda p: p == 'none')
# mSafe: pool chips per type (too few per chip)
R_ms = {}
ms = rates(lambda p: p == 'mSafe')
for t in TYPES + ['LowQ']:
    keys = [(c, t) for c in CHIPS if (c, t) in ms]
    if keys:
        R_ms[t] = (sum(ms[k][0] for k in keys), sum(ms[k][1] for k in keys), sum(ms[k][2] for k in keys))

def oe(ko, gene, chips, ref):
    O = E = 0.0; n = 0
    for c in chips:
        pert, ct, lib, Y = store[c]
        m = pert == ko
        for t in set(ct[m]):
            mm = m & (ct == t)
            if ref == 'neg':
                cnt, L, nref = R_neg.get((c, t), (None, 0, 0))
            else:
                cnt, L, nref = R_ms.get(t, (None, 0, 0))
            if L == 0 or nref < 3:
                continue
            O += Y.loc[mm, gene].sum(); E += lib[mm].sum() * cnt[gene] / L; n += int(mm.sum())
    return O, E, n

for ref in ['neg', 'msafe']:
    for ko in TARGETS:
        for scope in CHIPS + ['all']:
            chips = CHIPS if scope == 'all' else [scope]
            O, E, n = oe(ko, ko, chips, ref)
            if n == 0:
                continue
            p = stats.poisson.cdf(O, E) if E > 0 else np.nan
            ctrl = []
            for g in TARGETS:
                if g == ko:
                    continue
                o2, e2, _ = oe(ko, g, chips, ref)
                if e2 > 0:
                    ctrl.append(o2 / e2)
            ctrl = np.array(ctrl)
            r = O / E if E > 0 else np.nan
            rows.append(dict(ref=ref, ko=ko, scope=scope, n_ko=n, O=int(O), E=round(E, 2), OE=r, p_poisson_less=p,
                             offtarget_OE_median=float(np.median(ctrl)) if len(ctrl) else np.nan,
                             rank_vs_offtarget=float((ctrl <= r).mean()) if len(ctrl) else np.nan))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(out, 'knockdown_OE.csv'), index=False)
from statsmodels.stats.multitest import multipletests
for ref in ['neg', 'msafe']:
    s = df[(df.ref == ref) & (df.scope == 'all')].copy()
    s['q_bh'] = multipletests(s.p_poisson_less.fillna(1), method='fdr_bh')[1]
    # sign test: how many KOs have on-target O/E below their own off-target median
    below = int((s.OE < s.offtarget_OE_median).sum())
    sp = stats.binomtest(below, len(s), 0.5, alternative='greater').pvalue
    # Fisher-combined empirical rank (rank fraction ~ U under null, coarse)
    print(f'\n=== ref={ref} pooled over chips; on-target O/E below off-target median in {below}/{len(s)} KOs, sign-test p={sp:.3g}')
    print(s[['ko', 'n_ko', 'O', 'E', 'OE', 'offtarget_OE_median', 'rank_vs_offtarget', 'p_poisson_less', 'q_bh']].round(4).to_string(index=False))
    s.to_csv(os.path.join(out, f'knockdown_OE_pooled_{ref}.csv'), index=False)
# per-chip view for ref=neg
pc = df[(df.ref == 'neg') & (df.scope != 'all')].pivot(index='ko', columns='scope', values='OE').round(3)
print('\nper-chip O/E vs barcode-negative:\n', pc.to_string())
