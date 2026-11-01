"""Audit which BE CODEX annotation should define MENDER cell composition."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from build_pairs import DEFAULT_SECTIONS, stratified_subsample
from inspect_be import categorical


SOURCE = "/home/yancui/trajectory/EsophagusAtlas_CODEX.h5mu"
FIELDS = [
    "Cell Type",
    "MS_CL",
    "MS_CL_coarse",
    "leiden",
    "community",
    "community common",
    "neigh_sub",
]


def gmm_ari(x, truth, seed=0):
    x = StandardScaler().fit_transform(x)
    x = PCA(n_components=min(50, x.shape[1]), random_state=seed).fit_transform(x)
    pred = GaussianMixture(
        len(np.unique(truth)), covariance_type="full", random_state=seed
    ).fit_predict(x)
    return float(adjusted_rand_score(truth, pred))


def multiscale_composition(labels, coords, ks=(12, 36, 72)):
    onehot = OneHotEncoder(sparse_output=False).fit_transform(labels[:, None])
    width = min(max(ks) + 1, len(labels))
    nbr = NearestNeighbors(n_neighbors=width).fit(coords).kneighbors(
        return_distance=False
    )[:, 1:]
    return np.concatenate(
        [onehot[nbr[:, : min(k, nbr.shape[1])]].mean(1) for k in ks], axis=1
    )


def main():
    rows, counts = [], {}
    with h5py.File(SOURCE, "r") as handle:
        for section_i, section in enumerate(DEFAULT_SECTIONS):
            group = handle[f"mod/{section}"]
            niche = categorical(group["obs/neigh_name"])
            composition = categorical(group["obs/neigh_sub"])
            coords = np.asarray(group["obsm/spatial"][:], dtype=np.float32)
            values, sizes = np.unique(niche, return_counts=True)
            top = values[np.argsort(-sizes)[:6]]
            keep = np.flatnonzero(np.isin(niche, top))
            niche, composition, coords = niche[keep], composition[keep], coords[keep]
            take = stratified_subsample(
                niche, composition, 6000, 20260815 + section_i
            )
            niche, coords = niche[take], coords[take]

            for field in FIELDS:
                labels = categorical(group[f"obs/{field}"])[keep][take]
                local = multiscale_composition(labels.astype(str), coords)
                nbr = NearestNeighbors(n_neighbors=13).fit(coords).kneighbors(
                    return_distance=False
                )[:, 1:]
                rows.append(
                    {
                        "section": section,
                        "annotation": field,
                        "n_classes": len(np.unique(labels)),
                        "NMI_annotation_vs_niche": normalized_mutual_info_score(
                            labels, niche
                        ),
                        "ARI_annotation_vs_niche": adjusted_rand_score(labels, niche),
                        "annotation_12nn_agreement": np.mean(
                            labels[nbr] == labels[:, None]
                        ),
                        "niche_ARI_from_multiscale_composition": gmm_ari(local, niche),
                    }
                )
                unique, n = np.unique(labels, return_counts=True)
                counts[f"{section}::{field}"] = dict(
                    sorted(zip(unique.astype(str), n.astype(int)), key=lambda v: -v[1])
                )
    out = Path(__file__).resolve().parent / "summary_2domain_fixed"
    out.mkdir(exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "background_annotation_audit.csv", index=False)
    (out / "background_annotation_counts.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n"
    )
    summary = table.groupby("annotation").agg(
        n_classes_mean=("n_classes", "mean"),
        NMI_mean=("NMI_annotation_vs_niche", "mean"),
        local_agreement_mean=("annotation_12nn_agreement", "mean"),
        niche_composition_ARI_mean=("niche_ARI_from_multiscale_composition", "mean"),
        niche_composition_ARI_std=("niche_ARI_from_multiscale_composition", "std"),
    ).sort_values("niche_composition_ARI_mean", ascending=False)
    summary.to_csv(out / "background_annotation_audit_summary.csv")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
