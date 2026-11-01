"""Build BE CODEX pairs with only a natural background and simulated salient domain."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="/home/yancui/trajectory/EsophagusAtlas_CODEX.h5mu"
    )
    parser.add_argument("--sections", nargs="+", default=list(DEFAULT_SECTIONS))
    parser.add_argument("--out_dir", default="bench_be_protein/data_pairs_2domain")
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
            composition = categorical(group["obs/neigh_sub"])
            names = np.asarray(group["obs/cellid"][:]).astype(str)
            proteins = np.asarray(group["var/_index"][:]).astype(str)

            values, counts = np.unique(niche, return_counts=True)
            top = values[np.argsort(-counts)[: args.top_niches]]
            keep = np.flatnonzero(np.isin(niche, top))
            x, coords = x[keep], coords[keep]
            niche, composition, names = niche[keep], composition[keep], names[keep]
            take = stratified_subsample(
                niche, composition, args.max_cells, args.seed + section_i
            )
            x, coords = x[take], coords[take]
            niche, composition, names = niche[take], composition[take], names[take]

            natural_joint = np.char.add(
                np.char.add(niche.astype(str), "|"), composition.astype(str)
            )
            salient, angle = balanced_spatial_domains(
                coords,
                natural_joint,
                args.seed + 100 + section_i,
                args.n_salient_domains,
            )

            n_effect = args.n_salient_domains * args.proteins_per_domain
            association = feature_background_association(x, niche, composition)
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

            obs_control = pd.DataFrame(index=[f"{section}_control_{v}" for v in names])
            obs_condition = pd.DataFrame(index=[f"{section}_condition_{v}" for v in names])
            for obs, condition_name, sample in (
                (obs_control, "control", f"{section}_control"),
                (obs_condition, "disease", f"{section}_condition"),
            ):
                obs["condition"] = condition_name
                obs["section"] = section
                obs["slice_id"] = sample
                obs["composition_label"] = composition
                obs["gt_background_domain"] = niche
            obs_control["gt_salient_domain"] = "control"
            obs_condition["gt_salient_domain"] = [f"domain_{v}" for v in salient]
            obs = pd.concat([obs_control, obs_condition])

            pair_x = np.vstack([x, condition]).astype(np.float32)
            pair = ad.AnnData(
                X=pair_x,
                obs=obs,
                var=pd.DataFrame(index=pd.Index(proteins, name="protein")),
            )
            pair.layers["protein"] = pair_x.copy()
            pair.obsm["spatial"] = np.vstack([coords, coords]).astype(np.float32)

            nmi_niche = float(normalized_mutual_info_score(salient, niche))
            nmi_composition = float(
                normalized_mutual_info_score(salient, composition)
            )
            ari_niche = float(adjusted_rand_score(salient, niche))
            if max(abs(nmi_niche), abs(nmi_composition), abs(ari_niche)) > 0.01:
                raise RuntimeError(f"{section}: salient/background labels correlated")

            delta = condition[:, effect_cols] - x[:, effect_cols]
            oracle = float(
                adjusted_rand_score(
                    salient,
                    GaussianMixture(
                        args.n_salient_domains,
                        covariance_type="full",
                        reg_covar=1e-6,
                        random_state=0,
                    ).fit_predict(delta.astype(np.float64)),
                )
            )
            pair.uns["be_protein_2domain_simulation"] = {
                "source": args.source,
                "section": section,
                "background_truth": "neigh_name",
                "composition_label": "neigh_sub",
                "composition_is_not_an_evaluation_target": True,
                "condition_is_paired_control_plus_salient_domain_program": True,
                "n_salient_domains": args.n_salient_domains,
                "projection_angle_radians": angle,
                "salient_balanced_within_niche_x_composition": True,
                "effect_scale_standard_deviations": args.effect_scale,
                "domain_proteins": programs,
                "nmi_salient_vs_background": nmi_niche,
                "nmi_salient_vs_composition": nmi_composition,
                "ari_salient_vs_background": ari_niche,
                "paired_delta_oracle_ari": oracle,
                "seed": args.seed + section_i,
            }
            path = os.path.join(args.out_dir, f"be_{section}_paired.h5ad")
            pair.write_h5ad(path, compression="gzip")
            row = {
                "section": section,
                "n_source_cells": len(x),
                "n_pair_cells": pair.n_obs,
                "n_background_domains": len(np.unique(niche)),
                "n_composition_labels": len(np.unique(composition)),
                "n_salient_domains": args.n_salient_domains,
                "n_effect_proteins": n_effect,
                "nmi_salient_vs_background": nmi_niche,
                "nmi_salient_vs_composition": nmi_composition,
                "paired_delta_oracle_ari": oracle,
                "path": os.path.abspath(path),
            }
            manifest.append(row)
            print(json.dumps(row), flush=True)

    pd.DataFrame(manifest).to_csv(os.path.join(args.out_dir, "manifest.csv"), index=False)


if __name__ == "__main__":
    main()
