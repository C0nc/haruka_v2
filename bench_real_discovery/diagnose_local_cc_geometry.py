"""Small audit for local-only CellCharter geometry."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


data_path = Path("bench_spacseq/data/spac_lung_functional_guides_k8_hvg3000.h5ad")
emb_path = Path(
    "bench_real_discovery/results_symmetric_local_cc15/spacseq_functional/"
    "haruka_symmetric_local_cc_seed0"
)
adata = ad.read_h5ad(data_path)
names = np.load(emb_path / "obs_names.npy", allow_pickle=True).astype(str)
positions = pd.Index(adata.obs_names.astype(str)).get_indexer(names)
obs = adata.obs.iloc[positions]
coordinates = np.asarray(adata.obsm["spatial"])[positions]
labels = obs["perturbation"].astype(str).to_numpy()
target = obs["condition"].astype(str).to_numpy() == "disease"
slices = obs["slice_id"].astype(str).to_numpy()

same_label_fraction = []
target_neighbor_fraction = []
distances = []
for sample in pd.unique(slices):
    index = np.flatnonzero(slices == sample)
    distance, neighbors = NearestNeighbors(n_neighbors=16).fit(
        coordinates[index]
    ).kneighbors(coordinates[index])
    neighbors = index[neighbors[:, 1:]]
    keep = target[index]
    same_label_fraction.extend(
        (labels[neighbors] == labels[index, None]).mean(1)[keep].tolist()
    )
    target_neighbor_fraction.extend(target[neighbors].mean(1)[keep].tolist())
    distances.extend(distance[:, 1:].reshape(-1).tolist())

print(
    "same_sgrna_neighbor_fraction",
    float(np.mean(same_label_fraction)),
    float(np.median(same_label_fraction)),
)
print(
    "target_neighbor_fraction",
    float(np.mean(target_neighbor_fraction)),
    float(np.median(target_neighbor_fraction)),
)
print("neighbor_distance", float(np.median(distances)), float(np.quantile(distances, 0.95)))

for name, filename in (
    ("raw_z", "background.npy"),
    ("z_neighbor_only", "spatial.npy"),
    ("raw_s", "salient.npy"),
    ("s_neighbor_only", "salient_spatial.npy"),
):
    values = np.load(emb_path / filename)[target]
    values = values - values.mean(0)
    singular = np.linalg.svd(values, compute_uv=False)
    variance = singular**2
    effective_rank = variance.sum() ** 2 / np.square(variance).sum()
    print(
        name,
        "mean_feature_sd",
        float(np.std(values, axis=0).mean()),
        "effective_rank",
        float(effective_rank),
        "top2_fraction",
        float(variance[:2].sum() / variance.sum()),
    )
