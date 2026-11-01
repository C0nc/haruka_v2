"""Turn MapMyCells output into per-cell confidence and compare with the paper's labels.
Paper filter: per supertype (here: mapped broad class) MAD of correlations below the median (MADlow);
cells with corr < median - 3*MADlow removed.
Run: python -I postproc_mapping.py <bench_dir>
"""
import sys, json
import numpy as np, pandas as pd

B = sys.argv[1]
d = json.load(open(f"{B}/work/ctm_result.json"))
r = pd.DataFrame({"cellId": [int(x["cell_id"]) for x in d["results"]],
                  "ctm_cluster": [x["cluster"]["assignment"] for x in d["results"]],
                  "ctm_cls": [x["cls"]["assignment"] for x in d["results"]],
                  "ctm_boot_prob": [x["cluster"]["bootstrapping_probability"] for x in d["results"]],
                  "ctm_corr": [x["cluster"]["avg_correlation"] for x in d["results"]]})
med = r.groupby("ctm_cls").ctm_corr.transform("median")
madlow = r.groupby("ctm_cls").ctm_corr.transform(lambda x: np.median(np.abs(x[x < x.median()] - x.median())))
r["ctm_mad_pass"] = r.ctm_corr >= med - 3 * madlow
fb = pd.read_parquet(f"{B}/results/merfish_fb_arms.parquet")
r = r.merge(fb[["cellId", "gcluster", "cls", "arm", "region"]], on="cellId", how="left")
r["ctm_agree_cluster"] = r.ctm_cluster == r.gcluster
bdg = pd.read_csv(f"{B}/results/cluster_arm_calls.tsv", sep="\t", index_col=0)
gc2cls = bdg.assign(g=[i.split(":")[1] for i in bdg.index]).groupby("g").cls.first()
r["ctm_broad"] = r.ctm_cluster.map(gc2cls)
r["ctm_agree_broad"] = r.ctm_broad == r.cls
r[["cellId", "ctm_cluster", "ctm_boot_prob", "ctm_corr", "ctm_mad_pass", "ctm_agree_cluster", "ctm_agree_broad"]] \
    .to_parquet(f"{B}/results/mapping_confidence.parquet")
pd.set_option("display.width", 220)
cand = ((r.cls == "OPC_Oligo") & r.region.isin(["Isocortex", "fiber tracts"])) | \
       ((r.cls == "Astro") & r.region.isin(["HIP", "RHP", "HY"]))
a = r[cand & r.arm.isin(["early", "late"])]
print("== arm cells (candidate designs)")
print(a.groupby(["cls", "region", "arm"]).agg(n=("cellId", "size"), agree_cluster=("ctm_agree_cluster", "mean"),
      agree_broad=("ctm_agree_broad", "mean"), boot_prob_med=("ctm_boot_prob", "median"),
      frac_boot_ge_0_5=("ctm_boot_prob", lambda x: (x >= 0.5).mean()), mad_pass=("ctm_mad_pass", "mean"),
      corr_med=("ctm_corr", "median")).round(3).to_string())
# would our mapping put the cell in the same ARM? (arm of our cluster under the same dissection key)
key = a.region.map({"Isocortex": "Cortex", "HIP": "Hippocampus", "RHP": "Hippocampus", "HY": "Thalamus"})
key = key.fillna("Cortex") + ":" + a.ctm_cluster
a = a.assign(ctm_arm=key.map(bdg.arm).fillna("NA"))
print("== arm agreement (paper label arm vs our-mapping arm)")
print(a.groupby(["cls", "region", "arm"]).apply(lambda x: (x.ctm_arm == x.arm.iloc[0]).mean()).round(3).to_string())
print("== random forebrain sample")
s = r[~cand | ~r.arm.isin(["early", "late"])]
print("n", len(s), "agree_cluster", s.ctm_agree_cluster.mean().round(3), "agree_broad", s.ctm_agree_broad.mean().round(3),
      "boot>=0.5", (s.ctm_boot_prob >= 0.5).mean().round(3), "mad_pass", s.ctm_mad_pass.mean().round(3))
print(s.groupby("cls").agg(n=("cellId", "size"), agree_cluster=("ctm_agree_cluster", "mean"),
                           agree_broad=("ctm_agree_broad", "mean"), boot=("ctm_boot_prob", "median")).round(3).to_string())
