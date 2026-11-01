#!/usr/bin/env python3
"""Counterfactual-proportion check for the M2-1 tumour-vs-non-tumour Haruka fit.

Mirrors bench_real_discovery/cc_shift_info/check_cf_proportion.py.
dp = 0 on reference (control) cells by construction: dp is NEVER tested on them.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

D = Path("/home/yancui/haruka_v2/bench_petracer_lineage/reaudit_2026-10-16")
MD = Path(sys.argv[1]) if len(sys.argv) > 1 else next((D / "fits").glob("*/haruka_shared_full_niche_compgated*"))
K = 20
log = open(D / "check_cf_m2.log", "w")


def P(*x):
    s = " ".join(str(v) for v in x)
    print(s, flush=True)
    log.write(s + "\n")


a = ad.read_h5ad(D / "petracer_M2-1_tumour_vs_nontumour.h5ad")
o = a.obs
names = np.load(MD / "obs_names.npy", allow_pickle=True).astype(str)
assert (names == a.obs_names.astype(str)).all()
meta = json.load(open(MD / "meta.json"))
cats = meta["cell_type_categories"]
ct = o.cell_type.astype(str).to_numpy()
arm = o.condition.astype(str).to_numpy()
dis, ref = arm == "disease", arm == "control"
sd = o.signed_boundary_dist.to_numpy(float)
xy = np.asarray(a.obsm["spatial"])
code = pd.Categorical(ct, categories=cats).codes
_, nb = cKDTree(xy).query(xy, k=K + 1)
obs = np.zeros((len(ct), len(cats)), np.float32)
for c in range(1, K + 1):
    np.add.at(obs, (np.arange(len(ct)), code[nb[:, c]]), 1.0)
obs /= K
pbg = np.load(MD / "background_composition.npy")
pf = np.load(MD / "condition_composition.npy")
dp = np.load(MD / "salient_composition_shift.npy")
iM = cats.index("Malignant")
malf = obs[:, iM]
P(f"== M2-1 fit {MD}\n   cells {len(ct)}: control {ref.sum()}, disease {dis.sum()}, halo {(arm=='halo').sum()}")

P("\n-- C1 bookkeeping --")
P("   rows sum to 1: p_bg %.1e, p_full %.1e" % (np.abs(pbg.sum(1) - 1).max(), np.abs(pf.sum(1) - 1).max()))
P("   dp == p_full - p_bg on disease: %.1e; dp on non-disease max |.| %.1e (zero by construction; never tested)"
  % (np.abs(dp[dis] - (pf - pbg)[dis]).max(), np.abs(dp[~dis]).max()))


def r2(y, f):
    return 1 - ((y - f) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()


def tv(p, q):
    return 0.5 * np.abs(p - q).sum(-1)


P("\n-- C2 fit to observed k=20 composition (in-sample) --")
P("   control : p_bg   R2 %.3f TV %.3f" % (r2(obs[ref], pbg[ref]), tv(obs[ref], pbg[ref]).mean()))
P("   disease : p_bg   R2 %.3f TV %.3f" % (r2(obs[dis], pbg[dis]), tv(obs[dis], pbg[dis]).mean()))
P("   disease : p_full R2 %.3f TV %.3f" % (r2(obs[dis], pf[dis]), tv(obs[dis], pf[dis]).mean()))
P("   disease : dp L1 mean %.3f, median %.3f; dp Malignant column mean %+.4f (observed malignant frac in disease %.3f)"
  % (np.abs(dp[dis]).sum(1).mean(), np.median(np.abs(dp[dis]).sum(1)), dp[dis, iM].mean(), malf[dis].mean()))
share_mal = np.abs(dp[dis, iM]).sum() / np.abs(dp[dis]).sum()
P("   fraction of |dp| mass on the Malignant column: %.3f" % share_mal)

P("\n-- C3 matched on own type: share of the tumour-vs-normal change carried by p_bg vs p_full --")
rows = []
for c in np.unique(ct[dis]):
    r, d = np.flatnonzero(ref & (ct == c)), np.flatnonzero(dis & (ct == c))
    if len(r) < 30 or len(d) < 30:
        continue
    rows.append(dict(own=c, n_ref=len(r), n_dis=len(d), obs_ref=obs[r].mean(0), obs_dis=obs[d].mean(0),
                     pbg_dis=pbg[d].mean(0), pf_dis=pf[d].mean(0), pbg_ref=pbg[r].mean(0)))
S = pd.DataFrame(rows)
w = S.n_dis.to_numpy(float) / S.n_dis.sum()
for k, (x_, y_) in {"TV(obs dis, obs ref)  real tumour-vs-normal change": ("obs_dis", "obs_ref"),
                    "TV(p_bg dis, obs ref) background vs reference": ("pbg_dis", "obs_ref"),
                    "TV(p_bg dis, obs dis) background vs tumour": ("pbg_dis", "obs_dis"),
                    "TV(p_full dis, obs dis) salient vs tumour": ("pf_dis", "obs_dis"),
                    "TV(p_bg ref, obs ref) calibration on reference": ("pbg_ref", "obs_ref")}.items():
    P(f"   {k:55s} {np.dot(w, [tv(S[x_][i], S[y_][i]) for i in range(len(S))]):.4f}")
M = lambda col: np.einsum("i,ij->j", w, np.stack(S[col]))
T = pd.DataFrame({c: M(c) for c in ["obs_ref", "obs_dis", "pbg_dis", "pf_dis"]}, index=cats)
T["effect"] = T.obs_dis - T.obs_ref
T["bg_share"] = (T.pbg_dis - T.obs_ref) / T.effect
T["full_share"] = (T.pf_dis - T.obs_ref) / T.effect
T.loc[T.effect.abs() < 0.003, ["bg_share", "full_share"]] = np.nan
P(T.round(4).to_string())
T.to_csv(D / "check_cf_m2_C3_own_type.csv")

# Beyond the malignant fraction: renormalise over non-malignant neighbour types.
P("\n-- C3b beyond malignant: non-malignant neighbour composition renormalised (Malignant column dropped) --")
nm = [j for j in range(len(cats)) if j != iM]


def renorm(p):
    q = p[:, nm]
    return q / np.clip(q.sum(1, keepdims=True), 1e-6, None)


ob_r, pbg_r, pf_r = renorm(obs), renorm(pbg), renorm(pf)
dp_r = pf_r - pbg_r
rows = []
for c in S.own:
    r, d = ref & (ct == c), dis & (ct == c)
    rows.append(pd.DataFrame({"own": c, "type": [cats[j] for j in nm], "n": d.sum(),
                              "obs_ref": ob_r[r].mean(0), "obs_dis": ob_r[d].mean(0),
                              "pbg_dis": pbg_r[d].mean(0), "pf_dis": pf_r[d].mean(0)}))
R = pd.concat(rows)
Rw = R.groupby("type").apply(lambda x: pd.Series({k: np.average(x[k], weights=x.n) for k in ["obs_ref", "obs_dis", "pbg_dis", "pf_dis"]}))
Rw["effect"] = Rw.obs_dis - Rw.obs_ref
Rw["bg_share"] = (Rw.pbg_dis - Rw.obs_ref) / Rw.effect
Rw["full_share"] = (Rw.pf_dis - Rw.obs_ref) / Rw.effect
Rw.loc[Rw.effect.abs() < 0.003, ["bg_share", "full_share"]] = np.nan
P(Rw.round(4).to_string())
Rw.to_csv(D / "check_cf_m2_C3b_nonmalignant_renorm.csv")

P("\n-- C3c does per-cell dp track per-cell observed TME deviation beyond malignant fraction? (disease cells only) --")
P("   partial Spearman(dp_renorm[:,j], obs_renorm[:,j] | own type x malignant-frac decile), and same for p_bg")
mbin = pd.qcut(malf[dis], 10, labels=False, duplicates="drop")
grp = pd.Series(ct[dis]).astype(str) + "_" + pd.Series(mbin).astype(str)


def resid(v):
    s = pd.Series(v)
    return (s - s.groupby(grp.values).transform("mean")).to_numpy()


out = []
for k, j in enumerate(nm):
    yo = resid(ob_r[dis][:, k])
    out.append(dict(type=cats[j], rho_dp=spearmanr(resid(dp_r[dis][:, k]), yo)[0],
                    rho_pbg=spearmanr(resid(pbg_r[dis][:, k]), yo)[0],
                    rho_pfull=spearmanr(resid(pf_r[dis][:, k]), yo)[0],
                    sd_obs_resid=float(np.std(yo))))
C3c = pd.DataFrame(out).set_index("type")
P(C3c.round(3).to_string())
C3c.to_csv(D / "check_cf_m2_C3c_partial_rho.csv")

P("\n-- C4 dp vs distance inside the tumour (disease cells; um from boundary) --")
bins = [50, 100, 150, 250, 400, np.inf]
tb = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = dis & (sd >= lo) & (sd < hi)
    row = dict(bin=f"[{lo:g},{hi:g})", n=int(m.sum()), obs_malignant=malf[m].mean(), dp_malignant=dp[m, iM].mean(),
               dp_L1=np.abs(dp[m]).sum(1).mean())
    for t in ("Exhausted CD8 T cell", "CD4 T cell", "Treg", "CD11c macrophage", "Cancer fibroblast", "Endothelial", "AT1/AT2", "B cell"):
        j = cats.index(t)
        row[f"obs_{t}"] = obs[m, j].mean()
        row[f"dp_{t}"] = dp[m, j].mean()
    tb.append(row)
TB = pd.DataFrame(tb)
P(TB.round(4).T.to_string())
TB.to_csv(D / "check_cf_m2_C4_distance_bins.csv", index=False)
rho = {cats[j]: spearmanr(dp[dis, j], sd[dis])[0] for j in range(len(cats))}
rho_o = {cats[j]: spearmanr(obs[dis, j], sd[dis])[0] for j in range(len(cats))}
P("   Spearman with inside distance, dp vs observed:")
P(pd.DataFrame({"dp": rho, "observed": rho_o}).round(3).to_string())

P("\n-- C5 marker sanity for the types dp moves (mean log1p CPM, disease vs control, own type) --")
X = a.layers["counts"]
lib = np.asarray(X.sum(1)).ravel()
genes = a.var_names.astype(str)
markers = {"Exhausted CD8 T cell": ["Cd8a", "Cd8b1", "Pdcd1", "Havcr2", "Tox", "Gzmb"],
           "CD11c macrophage": ["Itgax", "Cd68", "Spp1", "Arg1", "C1qa", "Mrc1"],
           "Cancer fibroblast": ["Col1a1", "Postn", "Acta2", "Fap", "Pdgfrb"],
           "Endothelial": ["Pecam1", "Cldn5", "Plvap", "Gpihbp1", "Car4"],
           "CD4 T cell": ["Cd4", "Foxp3", "Il7r", "Cd3e"],
           "AT1/AT2": ["Sftpc", "Ager", "Lamp3"]}
for t, gl in markers.items():
    gl = [g for g in gl if g in genes]
    if not gl:
        continue
    idx = [list(genes).index(g) for g in gl]
    for nm_, m in (("control", ref & (ct == t)), ("disease", dis & (ct == t))):
        v = np.log1p(np.asarray(X[m][:, idx].todense()) * 1e4 / lib[m, None])
        P(f"   {t:22s} {nm_:8s} n={m.sum():5d} " + " ".join(f"{g}={x:.2f}" for g, x in zip(gl, v.mean(0))))
P("done")
