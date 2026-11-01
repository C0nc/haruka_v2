"""Build the corrected BE two-domain simulation using fine `Cell Type` composition."""

from __future__ import annotations

import argparse
import json
import os

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

from build_pairs import (
    DEFAULT_SECTIONS,
    balanced_spatial_domains,
    feature_background_association,
    stratified_subsample,
)
from inspect_be import categorical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="/home/yancui/trajectory/EsophagusAtlas_CODEX.h5mu"
    )
    parser.add_argument("--sections", nargs="+", default=list(DEFAULT_SECTIONS))
    parser.add_argument(
        "--out_dir", default="bench_be_protein/data_pairs_2domain_celltype"
    )
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--top_niches", type=int, default=6)
    parser.add_argument("--max_cells", type=int, default=6000)
    parser.add_argument("--n_salient_domains", type=int, default=3)
    parser.add_argument("--proteins_per_domain", type=int, default=3)
    parser.add_argument("--effect_scale", type=float, default=1.5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    with h5py.File(args.source, "r") as handle:
        for section_i, section in enumerate(args.sections):
            group = handle[f"mod/{section}"]
            x = np.asarray(group["X"][:], dtype=np.float32)
            coords = np.asarray(group["obsm/spatial"][:], dtype=np.float32)
            niche = categorical(group["obs/neigh_name"])
            cell_type = categorical(group["obs/Cell Type"])
            niche_sub = categorical(group["obs/neigh_sub"])
            names = np.asarray(group["obs/cellid"][:]).astype(str)
            proteins = np.asarray(group["var/_index"][:]).astype(str)

            values, counts = np.unique(niche, return_counts=True)
            top = values[np.argsort(-counts)[: args.top_niches]]
            keep = np.flatnonzero(np.isin(niche, top))
            x, coords = x[keep], coords[keep]
            niche, cell_type, niche_sub, names = (
                niche[keep], cell_type[keep], niche_sub[keep], names[keep]
            )
            take = stratified_subsample(
                niche, cell_type, args.max_cells, args.seed + section_i
            )
            x, coords = x[take], coords[take]
            niche, cell_type, niche_sub, names = (
                niche[take], cell_type[take], niche_sub[take], names[take]
            )

            natural_joint = np.char.add(
                np.char.add(niche.astype(str), "|"), cell_type.astype(str)
            )
            salient, angle = balanced_spatial_domains(
                coords,
                natural_joint,
                args.seed + 100 + section_i,
                args.n_salient_domains,
            )
            n_effect = args.n_salient_domains * args.proteins_per_domain
            association = feature_background_association(x, niche, cell_type)
            effect_cols = np.argsort(association, kind="mergesort")[:n_effect]
            condition = x.copy()
            weights = np.linspace(1.0, 0.65, args.proteins_per_domain)
            programs = {}
            for domain in range(args.n_salient_domains):
                cols = effect_cols[
                    domain * args.proteins_per_domain :
                    (domain + 1) * args.proteins_per_domain
                ]
                condition[np.ix_(salient == domain, cols)] += (
                    args.effect_scale * weights[None, :]
                )
                programs[f"domain_{domain}"] = proteins[cols].tolist()

            control = pd.DataFrame(index=[f"{section}_control_{v}" for v in names])
            target = pd.DataFrame(index=[f"{section}_condition_{v}" for v in names])
            for obs, condition_name, sample in (
                (control, "control", f"{section}_control"),
                (target, "disease", f"{section}_condition"),
            ):
                obs["condition"] = condition_name
                obs["section"] = section
                obs["slice_id"] = sample
                obs["cell_type_label"] = cell_type
                obs["niche_sub_label"] = niche_sub
                obs["gt_background_domain"] = niche
            control["gt_salient_domain"] = "control"
            target["gt_salient_domain"] = [f"domain_{v}" for v in salient]
            obs = pd.concat([control, target])
            pair_x = np.vstack([x, condition]).astype(np.float32)
            pair = ad.AnnData(
                X=pair_x,
                obs=obs,
                var=pd.DataFrame(index=pd.Index(proteins, name="protein")),
            )
            pair.layers["protein"] = pair_x.copy()
            pair.obsm["spatial"] = np.vstack([coords, coords]).astype(np.float32)

            nmi_niche = float(normalized_mutual_info_score(salient, niche))
            nmi_cell_type = float(normalized_mutual_info_score(salient, cell_type))
            ari_niche = float(adjusted_rand_score(salient, niche))
            delta = condition[:, effect_cols] - x[:, effect_cols]
            oracle = float(adjusted_rand_score(
                salient,
                GaussianMixture(
                    args.n_salient_domains, covariance_type="full",
                    reg_covar=1e-6, random_state=0,
                ).fit_predict(delta.astype(np.float64)),
            ))
            if max(abs(nmi_niche), abs(nmi_cell_type), abs(ari_niche)) > 0.01:
                raise RuntimeError(f"{section}: salient/background labels correlated")
            pair.uns["be_protein_2domain_simulation"] = {
                "source": args.source,
                "section": section,
                "background_truth": "neigh_name",
                "cell_composition_annotation": "Cell Type",
                "neigh_sub_is_not_used_as_cell_type": True,
                "condition_is_paired_control_plus_salient_domain_program": True,
                "salient_balanced_within_niche_x_cell_type": True,
                "domain_proteins": programs,
                "effect_scale_standard_deviations": args.effect_scale,
                "nmi_salient_vs_background": nmi_niche,
                "nmi_salient_vs_cell_type": nmi_cell_type,
                "ari_salient_vs_background": ari_niche,
                "paired_delta_oracle_ari": oracle,
                "projection_angle_radians": angle,
            }
            path = os.path.join(args.out_dir, f"be_{section}_paired.h5ad")
            pair.write_h5ad(path, compression="gzip")
            row = {
                "section": section,
                "n_source_cells": len(x),
                "n_pair_cells": pair.n_obs,
                "n_background_domains": len(np.unique(niche)),
                "n_cell_types": len(np.unique(cell_type)),
                "n_salient_domains": args.n_salient_domains,
                "nmi_salient_vs_background": nmi_niche,
                "nmi_salient_vs_cell_type": nmi_cell_type,
                "paired_delta_oracle_ari": oracle,
                "path": os.path.abspath(path),
            }
            manifest.append(row)
            print(json.dumps(row), flush=True)
    pd.DataFrame(manifest).to_csv(os.path.join(args.out_dir, "manifest.csv"), index=False)


if __name__ == "__main__":
    main()
