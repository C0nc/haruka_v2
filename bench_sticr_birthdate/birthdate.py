"""Task 2: labelling-time ("birthdate") profile per cluster from STICR scRNA-seq metadata.

Normalisation follows the paper: counts at each injection age are divided by the
total at that injection age (so ages with more captured cells/clones are not
over-weighted), then renormalised to a distribution over ages per cluster.

Two units:
  cells  : barcoded cells
  mclones: multicellular clones (clone_index shared by >=2 cells, same mouse);
           a clone counts once for every cluster it contains.
Run: python -I birthdate.py <geo_metadata.tsv> <outdir>
"""
import sys, os
import numpy as np, pandas as pd

AGES = ["E11", "E12", "E13", "E14", "E15", "E16", "E17", "P0"]
AGE_NUM = dict(zip(AGES, [11, 12, 13, 14, 15, 16, 17, 19.5]))
LATE = ["E16", "E17", "P0"]

meta_p, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
m = pd.read_csv(meta_p, sep="\t")
m["clone"] = m["mouse"] + "|" + m["clone_index"]
cs = m["clone"].map(m["clone"].value_counts())
m["multi"] = cs >= 2


def profile(df, unit, group_cols, norm_cols):
    """Return per-group normalised age distribution."""
    if unit == "cells":
        cnt = df.groupby(group_cols + ["injection_date"]).size()
        tot = df.groupby("injection_date").size()
    else:
        d = df[df.multi]
        cnt = d.drop_duplicates(group_cols + ["clone"]).groupby(group_cols + ["injection_date"]).size()
        tot = d.drop_duplicates(["clone"]).groupby("injection_date").size()
    cnt = cnt.unstack("injection_date").reindex(columns=AGES).fillna(0)
    tot = tot.reindex(AGES)
    assert not norm_cols  # per-region handled by subsetting
    totm = pd.DataFrame(np.repeat(tot.values.reshape(1, -1), len(cnt), 0), index=cnt.index, columns=AGES)
    rate = cnt / totm.replace(0, np.nan)
    rate = rate.fillna(0)
    p = rate.div(rate.sum(1), axis=0)
    res = pd.DataFrame(index=cnt.index)
    res["n_" + unit] = cnt.sum(1).astype(int)
    for a in AGES:
        res["p_" + a] = p[a]
    res["peak"] = p.idxmax(1)
    res["frac_ge_E16"] = p[LATE].sum(1)
    res["mean_age"] = (p * pd.Series(AGE_NUM)).sum(1)
    res["n_ages_present"] = (cnt > 0).sum(1)
    return res


# Clusters are region-specific (region_celltype_cluster); celltype_cluster numbers are
# re-used across regions. Broad class = majority celltype_broad of the cluster.
# Normalisation: per injection-age total WITHIN the region.
pd.set_option("display.width", 250, "display.max_rows", 700)
out_all = {}
for key in ["region_celltype_cluster", "region_celltype_sub"]:
    broad = m.groupby(key).celltype_broad.agg(lambda x: x.value_counts().index[0])
    bpur = m.groupby(key).celltype_broad.agg(lambda x: x.value_counts(normalize=True).iloc[0])
    res = {}
    for unit in ["cells", "mclones"]:
        r = pd.concat({rg: profile(m[m.region == rg], unit, [key], [])
                       for rg in m.region.unique()}, names=["region"]).droplevel(0)
        res[unit] = r
    j = res["cells"].join(res["mclones"], lsuffix="_cell", rsuffix="_mc")
    j.insert(0, "broad_purity", bpur.reindex(j.index).values)
    j.insert(0, "celltype_broad", broad.reindex(j.index).values)
    j.insert(0, "region", [x.split(":")[0] for x in j.index])
    # leave-one-mouse-out stability of mclone mean_age
    loo = {}
    for mo in m.mouse.unique():
        d = m[m.mouse != mo]
        loo[mo] = pd.concat([profile(d[d.region == rg], "mclones", [key], [])["mean_age"]
                             for rg in m.region.unique()])
    j["loo_sd_mean_age_mc"] = pd.DataFrame(loo).std(1).reindex(j.index)
    j.to_csv(f"{out}/birthdate_{key}.tsv", sep="\t")
    out_all[key] = j
    print("==", key, j.shape)
    print("cell vs mclone mean_age spearman:", j[["mean_age_cell", "mean_age_mc"]].corr("spearman").iloc[0, 1])
    print("cell vs mclone fracLate spearman:", j[["frac_ge_E16_cell", "frac_ge_E16_mc"]].corr("spearman").iloc[0, 1])
j = out_all["region_celltype_sub"]
cols = ["region", "celltype_broad", "n_cells", "n_mclones", "peak_cell", "peak_mc", "frac_ge_E16_cell",
        "frac_ge_E16_mc", "mean_age_mc", "loo_sd_mean_age_mc"]
print(j[j.n_mclones >= 20].sort_values(["celltype_broad", "region", "mean_age_mc"])[cols].round(2).to_string())
print(pd.crosstab(m.injection_date, m.multi))
