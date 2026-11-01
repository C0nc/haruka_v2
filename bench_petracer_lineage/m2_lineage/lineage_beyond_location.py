# Does z or s add lineage information beyond location? coordinates+X vs coordinates alone (same blocked CV).
exec(open('lineage_test.py').read().split("res = []")[0])
cl = ob.lin_clone.astype(str).to_numpy(); cd = ob.lin_clade.astype(str).to_numpy()
fit = pd.to_numeric(ob.lin_fitness, errors='coerce').to_numpy()
keep = np.isin(cl, [c for c, n in pd.Series(cl).value_counts().items() if c != 'nan' and n >= 100])
k2 = np.isin(cl, ['2-1', '2-2']); k3 = (cl == '5') & (cd != 'nan'); big = [c for c, n in pd.Series(cd[k3]).value_counts().items() if n >= 20]; k3 &= np.isin(cd, big)
k4 = (cl == '5') & np.isfinite(fit)
Rc = {'coords': R['coordinates'], 'coords+z': np.hstack([R['coordinates'], R['z']]), 'coords+s': np.hstack([R['coordinates'], R['s']]),
      'coords+expr': np.hstack([R['coordinates'], R['expression PCA32']])}
rows = []
for name, f in [('T1 clone', lambda X: cv_cls(X[keep], cl[keep], blocks[keep])), ('T2 2-1 vs 2-2', lambda X: cv_cls(X[k2], cl[k2] == '2-2', blocks[k2], 'auc')),
                ('T3 subclade', lambda X: cv_cls(X[k3], cd[k3], blocks[k3])), ('T4 fitness', lambda X: cv_reg(X[k4], fit[k4], blocks[k4]))]:
    v = {k: f(X) for k, X in Rc.items()}; rows.append(dict(test=name, **v))
    P(f'   {name:14s} ' + '  '.join(f'{k} {x:.3f}' for k, x in v.items()))
pd.DataFrame(rows).to_csv(D / 'lineage_beyond_location.csv', index=False)
