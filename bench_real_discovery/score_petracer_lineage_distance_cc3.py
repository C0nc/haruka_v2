"""Test whether z-CC or s-CC better preserves PEtracer lineage distance."""

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "bench_real_discovery/results_symmetric_condition_cc3"
SOURCE = Path("/home/yancui/Jill/fourteen/petracer_data/M1_tumor_tracing.h5td")


def barcode_distance(characters: np.ndarray, left: np.ndarray, right: np.ndarray):
    a = characters[left]
    b = characters[right]
    observed = (a >= 0) & (b >= 0)
    denominator = observed.sum(1)
    distance = np.full(len(left), np.nan, dtype=np.float64)
    valid = denominator > 0
    distance[valid] = ((a[valid] != b[valid]) & observed[valid]).sum(1) / denominator[valid]
    return distance


with h5py.File(SOURCE, "r") as handle:
    source_characters = ad.io.read_elem(handle["obsm/characters"])

rows = []
for section_i, section in enumerate(("M1-1", "M1-2", "M1-3", "M1-4")):
    data_path = ROOT / (
        "bench_petracer_lineage/data_m1t1_pairs_22k/"
        f"petracer_M1T1_{section}_paired_22k.h5ad"
    )
    emb_path = RESULT / f"petracer_{section}/haruka_symmetric_condition_cc3_seed0"
    data = ad.read_h5ad(data_path, backed="r")
    names = np.load(emb_path / "obs_names.npy", allow_pickle=True).astype(str)
    obs = data.obs.loc[names].copy()
    eligible = (
        (obs["condition"].astype(str).to_numpy() == "disease")
        & obs["lineage_effect_eligible"].astype(bool).to_numpy()
    )
    focal_obs = obs.loc[eligible]
    characters = source_characters.loc[focal_obs["pair_id"].astype(str)].to_numpy(np.int16)
    clade = focal_obs["gt_lineage_clade"].astype(str).to_numpy()
    rng = np.random.default_rng(20260820 + section_i)
    n_pairs = 200_000
    random_left = rng.integers(0, len(focal_obs), size=n_pairs)
    random_right = rng.integers(0, len(focal_obs), size=n_pairs)
    different = random_left != random_right
    random_left, random_right = random_left[different], random_right[different]
    random_character_distance = barcode_distance(characters, random_left, random_right)
    random_valid = np.isfinite(random_character_distance)
    random_mean = float(np.nanmean(random_character_distance))

    for representation, filename in (("z_cc", "spatial.npy"), ("s_cc", "salient_spatial.npy")):
        values = np.load(emb_path / filename)[eligible]
        values = StandardScaler().fit_transform(values)
        norm = np.linalg.norm(values, axis=1, keepdims=True)
        normalized = values / np.maximum(norm, 1e-12)

        neighbors = NearestNeighbors(n_neighbors=21, metric="cosine").fit(normalized)
        _, index = neighbors.kneighbors(normalized)
        neighbor_left = np.repeat(np.arange(len(values)), 20)
        neighbor_right = index[:, 1:].reshape(-1)
        neighbor_character_distance = barcode_distance(
            characters, neighbor_left, neighbor_right
        )
        same_clade_neighbor = clade[neighbor_left] == clade[neighbor_right]

        random_embedding_distance = 1.0 - np.sum(
            normalized[random_left] * normalized[random_right], axis=1
        )
        all_valid = random_valid & np.isfinite(random_embedding_distance)
        same_clade_random = all_valid & (clade[random_left] == clade[random_right])
        rho_all = spearmanr(
            random_embedding_distance[all_valid],
            random_character_distance[all_valid],
        ).statistic
        rho_within = spearmanr(
            random_embedding_distance[same_clade_random],
            random_character_distance[same_clade_random],
        ).statistic
        rows.append({
            "section": section,
            "representation": representation,
            "n_cells": int(len(values)),
            "random_barcode_distance_mean": random_mean,
            "embedding_knn_barcode_distance_mean": float(np.nanmean(neighbor_character_distance)),
            "embedding_knn_lineage_enrichment": float(
                (random_mean - np.nanmean(neighbor_character_distance)) / random_mean
            ),
            "embedding_knn_same_clade_fraction": float(np.mean(same_clade_neighbor)),
            "embedding_knn_within_clade_barcode_distance_mean": float(
                np.nanmean(neighbor_character_distance[same_clade_neighbor])
            ),
            "pairwise_spearman_all": float(rho_all),
            "pairwise_spearman_within_clade": float(rho_within),
            "n_random_pairs": int(all_valid.sum()),
            "n_within_clade_random_pairs": int(same_clade_random.sum()),
        })

table = pd.DataFrame(rows)
summary = table.groupby("representation", as_index=False).agg(
    knn_lineage_enrichment_mean=("embedding_knn_lineage_enrichment", "mean"),
    knn_lineage_enrichment_std=("embedding_knn_lineage_enrichment", "std"),
    knn_same_clade_fraction_mean=("embedding_knn_same_clade_fraction", "mean"),
    pairwise_spearman_all_mean=("pairwise_spearman_all", "mean"),
    pairwise_spearman_within_clade_mean=("pairwise_spearman_within_clade", "mean"),
)
out = RESULT / "metrics"
out.mkdir(parents=True, exist_ok=True)
table.to_csv(out / "cc3_lineage_distance_by_section.csv", index=False)
summary.to_csv(out / "cc3_lineage_distance_summary.csv", index=False)
print(table.to_string(index=False))
print("\nSUMMARY")
print(summary.to_string(index=False))
