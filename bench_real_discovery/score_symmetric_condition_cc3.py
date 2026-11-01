"""Direct K-means ARI audit for corrected SPARC and PEtracer CC3 readouts."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "bench_real_discovery/results_symmetric_condition_cc3"


def align(data_path: Path, emb_path: Path):
    data = ad.read_h5ad(data_path, backed="r")
    names = np.load(emb_path / "obs_names.npy", allow_pickle=True).astype(str)
    missing = pd.Index(names).difference(data.obs_names.astype(str))
    if len(missing):
        raise ValueError(f"Missing embedding observations: {missing[:5].tolist()}")
    return data.obs.loc[names].copy()


def score(x: np.ndarray, labels: np.ndarray, seed: int = 0) -> dict:
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels).astype(str)
    variable = np.var(x, axis=0) > 1e-10
    x = x[:, variable]
    k = len(np.unique(labels))
    prediction = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(x)
    return {
        "ari": float(adjusted_rand_score(labels, prediction)),
        "nmi": float(normalized_mutual_info_score(labels, prediction)),
        "n_cells": int(len(labels)),
        "n_classes": int(k),
        "n_features": int(x.shape[1]),
    }


rows = []
spac_data = ROOT / "bench_spacseq/data/spac_lung_functional_guides_k8_hvg3000.h5ad"
spac_emb = RESULT / "spacseq_functional/haruka_symmetric_condition_cc3_seed0"
spac_obs = align(spac_data, spac_emb)
target = spac_obs["condition"].astype(str).to_numpy() == "disease"
exact_guide = target & (spac_obs["guide_identity"].astype(str).to_numpy() != "")
for representation, filename in (("z_cc", "spatial.npy"), ("s_cc", "salient_spatial.npy")):
    values = np.load(spac_emb / filename)
    for truth, mask, labels in (
        ("target_gene", target, spac_obs["perturbation"].astype(str).to_numpy()),
        ("exact_sgRNA", exact_guide, spac_obs["guide_identity"].astype(str).to_numpy()),
    ):
        rows.append({
            "dataset": "SPARC", "section": "all_slices", "truth": truth,
            "representation": representation, **score(values[mask], labels[mask]),
        })

for section in ("M1-1", "M1-2", "M1-3", "M1-4"):
    data_path = ROOT / (
        "bench_petracer_lineage/data_m1t1_pairs_22k/"
        f"petracer_M1T1_{section}_paired_22k.h5ad"
    )
    emb_path = RESULT / f"petracer_{section}/haruka_symmetric_condition_cc3_seed0"
    obs = align(data_path, emb_path)
    eligible = (
        (obs["condition"].astype(str).to_numpy() == "disease")
        & obs["lineage_effect_eligible"].astype(bool).to_numpy()
    )
    for representation, filename in (("z_cc", "spatial.npy"), ("s_cc", "salient_spatial.npy")):
        values = np.load(emb_path / filename)[eligible]
        for truth, column in (
            ("lineage_family", "gt_lineage_family"),
            ("lineage_clade", "gt_lineage_clade"),
        ):
            rows.append({
                "dataset": "PEtracer", "section": section, "truth": truth,
                "representation": representation,
                **score(values, obs.loc[eligible, column].astype(str).to_numpy()),
            })

table = pd.DataFrame(rows)
summary = (
    table.groupby(["dataset", "truth", "representation"], as_index=False)
    .agg(
        ari_mean=("ari", "mean"), ari_std=("ari", "std"),
        nmi_mean=("nmi", "mean"), n_sections=("section", "nunique"),
    )
)
out = RESULT / "metrics"
out.mkdir(parents=True, exist_ok=True)
table.to_csv(out / "cc3_kmeans_ari_by_section.csv", index=False)
summary.to_csv(out / "cc3_kmeans_ari_summary.csv", index=False)
print(table.to_string(index=False))
print("\nSUMMARY")
print(summary.to_string(index=False))
