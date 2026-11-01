#!/usr/bin/env python3
"""Define KRASi response from drug pharmacodynamics instead of model-output magnitude.

KRAS inhibition acts by shutting down MAPK/ERK signalling.  Its canonical
pharmacodynamic readout is the MAPK Pathway Activity Score (MPAS; Dusp4/6,
Spry2/4, Etv4/5, Ccnd1, Epha2 -- Wagle et al. / Genentech MPAS), which falls when
the pathway is inhibited.  A responding tumour cell is therefore one whose MAPK
output has been driven BELOW what untreated tumour cells of the same type show.

This is control-anchored (uses the 46,971 untreated tumour cells), direction-aware
(low MAPK = response, not |shift|), and independent of the Haruka latents.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scanpy as sc, warnings
warnings.filterwarnings('ignore')

TUMOR = ['Tumor_EMT', 'Tumor_epithelial', 'Tumor_epithelial_Itgb6']
SIGS = {
    'MAPK_MPAS':      ['Dusp4','Dusp6','Spry2','Spry4','Etv4','Etv5','Ccnd1','Epha2'],
    'MAPK_extended':  ['Dusp4','Dusp5','Dusp6','Spry1','Spry2','Spry4','Etv1','Etv4','Etv5',
                       'Ccnd1','Epha2','Fosl1','Myc','Egr1'],
    'Proliferation':  ['Mki67','Top2a','Ccnb1','Birc5','Ccna2','Aurkb','Cdk1','Pcna','Mcm2'],
    'EMT':            ['Zeb1','Zeb2','Snai1','Snai2','Twist1','Cdh2'],
    'Stress_p21':     ['Cdkn1a','Cdkn2a','Mdm2','Gdf15'],
    'AT2_diff':       ['Lamp3','Abca3','Nkx2-1','Cebpa'],
}

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad')
    p.add_argument('--out-dir', type=Path,
                   default=Path('/home/yancui/haruka_v2/bench_kras_xenium/biological_responder'))
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)

    h = ad.read_h5ad(args.data, backed='r')
    keep = h.obs['celltype'].astype(str).isin(TUMOR).to_numpy()
    print(f'tumour cells: {keep.sum()} of {h.n_obs}', flush=True)
    a = h[keep].to_memory(); h.file.close()
    a.obs['arm'] = a.obs['condition_label'].astype(str)
    a.obs['sl'] = a.obs['slice_id'].astype(str)
    a.obs['ct'] = a.obs['celltype'].astype(str)

    # X in this object is ALREADY normalized+log1p; raw counts live in layers['counts'].
    a.X = a.layers['counts'].copy()
    raw_total = np.asarray(a.X.sum(1)).ravel()
    print(f'using layers[counts]: max {a.X.max():.0f} median total {np.median(raw_total):.0f}',
          flush=True)
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)

    for name, genes in SIGS.items():
        present = [g for g in genes if g in a.var_names]
        sc.tl.score_genes(a, present, score_name=name, random_state=args.seed)
        print(f'  {name}: {len(present)}/{len(genes)} genes', flush=True)
    a.obs['n_counts_raw'] = raw_total

    o = a.obs
    print('\n=== 1. Is MAPK output actually suppressed by treatment? (per section) ===')
    t = o.groupby(['arm','sl'],observed=True)[['MAPK_MPAS','MAPK_extended','Proliferation',
                                               'EMT','Stress_p21']].median()
    print(t.round(3).to_string())
    ctl = o.loc[o.arm=='Control','MAPK_MPAS']; trt = o.loc[o.arm=='KRASi_treated','MAPK_MPAS']
    print(f"\n  median MPAS  control {ctl.median():+.3f}  treated {trt.median():+.3f}")
    cs = o[o.arm=='Control'].groupby('sl',observed=True)['MAPK_MPAS'].median()
    ts = o[o.arm=='KRASi_treated'].groupby('sl',observed=True)['MAPK_MPAS'].median()
    print(f"  every treated section below every control section: {ts.max() < cs.min()}"
          f"  (treated max {ts.max():+.3f} vs control min {cs.min():+.3f})")
    print(f"  rho(MPAS, n_counts) = {pd.Series(o.MAPK_MPAS).corr(pd.Series(o.n_counts_raw),method='spearman'):+.3f}"
          "   (sanity: should be small)")

    print('\n=== 2. Control-anchored responder: MPAS below control 25th pct, within cell type ===')
    o['bio_responder'] = np.nan
    for ct in TUMOR:
        thr = o.loc[(o.arm=='Control') & (o.ct==ct), 'MAPK_MPAS'].quantile(0.25)
        m = (o.arm=='KRASi_treated') & (o.ct==ct)
        o.loc[m,'bio_responder'] = (o.loc[m,'MAPK_MPAS'] < thr).astype(float)
        base = ((o.arm=='Control') & (o.ct==ct)).sum()
        print(f'  {ct:26s} control p25 threshold {thr:+.3f} | treated responders '
              f'{o.loc[m,"bio_responder"].mean()*100:5.1f}%  (control baseline 25.0%, n_ctl={base})')
    trt_mask = o.arm=='KRASi_treated'
    print(f'\n  overall treated responder rate {o.loc[trt_mask,"bio_responder"].mean()*100:.1f}% '
          f'vs 25.0% expected if treatment did nothing')
    print('\n  by section:')
    print((o[trt_mask].groupby('sl',observed=True)['bio_responder'].agg(['mean','size'])
           .assign(pct=lambda d:(d['mean']*100).round(1))[['pct','size']]).to_string())
    print('\n  by tumour subtype:')
    print((o[trt_mask].groupby('ct',observed=True)['bio_responder'].mean()*100).round(1).to_string())

    out = o.loc[trt_mask, ['cell_id','sl','ct','MAPK_MPAS','MAPK_extended','Proliferation',
                           'EMT','Stress_p21','AT2_diff','bio_responder','n_counts_raw']] \
           if 'cell_id' in o else o.loc[trt_mask].assign(cell_id=o.index[trt_mask])[
              ['cell_id','sl','ct','MAPK_MPAS','MAPK_extended','Proliferation','EMT',
               'Stress_p21','AT2_diff','bio_responder','n_counts_raw']]
    out.to_csv(args.out_dir/'treated_tumor_biological_response.csv.gz', index=False, compression='gzip')
    o[['arm','sl','ct','MAPK_MPAS','MAPK_extended','Proliferation','EMT','Stress_p21','AT2_diff']] \
        .to_csv(args.out_dir/'all_tumor_signature_scores.csv.gz', index=False, compression='gzip')
    # ---- 3/4: agreement with the model label, and a REAL prediction test ----
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from scipy.stats import fisher_exact
        CC = ('/home/yancui/haruka_v2/results_real_compgated_nb_final/'
              'mouselung_kras_xenium_clean_noC3/haruka_z_s_cellcharter_cc3_condition_only/')
        mt = pd.read_csv('/home/yancui/haruka_v2/final_results_nb/kras/biology/responder/tables/'
                         'treated_tumor_relative_responder_assignments.csv.gz')
        bio = o.loc[trt_mask].copy(); bio['cid'] = bio.index.astype(str)
        mt['cid'] = mt.cell_id.astype(str)
        j = mt.merge(bio[['cid','bio_responder','MAPK_MPAS','Proliferation']], on='cid')
        print(f'\n=== 3. Does the model label agree with the biological one? (n={len(j)}) ===')
        print(f'  rho(model response_percentile, MAPK_MPAS)   = '
              f'{j.response_percentile.corr(j.MAPK_MPAS, method="spearman"):+.3f}  '
              '(response should mean LOW MAPK -> expect negative)')
        ex = j[j.response_class.ne('intermediate')]
        tab = pd.crosstab(ex.response_class.eq('relative_responder'), ex.bio_responder.astype(bool))
        orr, pv = fisher_exact(tab.to_numpy())
        print(f'  model-responder vs bio-responder odds ratio {orr:.2f} (p={pv:.2e})')
        print(f'  bio-responder rate among model responders    '
              f'{ex[ex.response_class.eq("relative_responder")].bio_responder.mean()*100:.1f}%')
        print(f'  bio-responder rate among model NON-responders '
              f'{ex[ex.response_class.eq("relative_nonresponder")].bio_responder.mean()*100:.1f}%')

        print('\n=== 4. REAL test: predict the BIOLOGICAL responder from z-CC vs s-CC ===')
        names = pd.Index(np.load(CC+'obs_names.npy', allow_pickle=True).astype(str))
        loc = names.get_indexer(j.cid.to_numpy())
        ok = loc >= 0
        y = j.bio_responder.to_numpy()[ok].astype(int); sl_ = j.slice_id.to_numpy()[ok]
        for tag, f in [('z-CC background','spatial.npy'), ('s-CC salient','salient_spatial.npy'),
                       ('z intrinsic','background.npy'), ('s intrinsic','salient.npy')]:
            X = np.load(CC+f)[loc[ok]].astype(np.float64); sc_ = []
            for s_ in np.unique(sl_):
                tr, te = sl_ != s_, sl_ == s_
                ss = StandardScaler().fit(X[tr])
                clf = LogisticRegression(max_iter=3000, random_state=args.seed,
                                         class_weight='balanced').fit(ss.transform(X[tr]), y[tr])
                sc_.append(roc_auc_score(y[te], clf.predict_proba(ss.transform(X[te]))[:, 1]))
            print(f'  {tag:18s} leave-section-out AUROC {np.mean(sc_):.3f}  {np.round(sc_,3)}')
    except Exception as e:
        print('comparison step failed:', repr(e))

    (args.out_dir/'definition.json').write_text(json.dumps({
        'definition': 'treated tumour cell with MAPK MPAS below the 25th percentile of UNTREATED '
                      'tumour cells of the same celltype',
        'rationale': 'KRAS inhibition suppresses MAPK/ERK output; MPAS is its canonical PD biomarker',
        'control_anchored': True, 'direction_aware': True,
        'independent_of_haruka_latents': True,
        'signatures': {k: [g for g in v if g in a.var_names] for k, v in SIGS.items()},
    }, indent=2) + '\n')
    print(f'\nwrote {args.out_dir}')


if __name__ == '__main__':
    main()
