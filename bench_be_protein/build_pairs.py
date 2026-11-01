"""Build paired control/condition CODEX benchmarks from five BE regions.

Natural factors
---------------
* ``neigh_sub``  -> background intrinsic composition (Epithelial/Stromal)
* ``neigh_name`` -> background spatial niche

Condition-only factors are injected into an exact copy of each control section.
Both are balanced within the joint natural background strata; the intrinsic state
is additionally balanced within the simulated spatial domain.
"""

from __future__ import annotations

import argparse
import json
import os

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from inspect_be import categorical


DEFAULT_SECTIONS = (
    "E08_reg006",
    "E11_reg002",
    "E12_reg003",
    "E17_reg006",
    "E19_reg001",
)


def balanced_spatial_domains(coords, strata, seed, n_domains=3):
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(0, np.pi))
    projection = coords[:, 0] * np.cos(angle) + coords[:, 1] * np.sin(angle)
    labels = np.empty(len(coords), dtype=np.int16)
    for group in np.unique(strata):
        rows = np.flatnonzero(strata == group)
        order = rows[np.argsort(projection[rows], kind="mergesort")]
        labels[order] = np.minimum(
            np.arange(len(order)) * n_domains // len(order), n_domains - 1
        )
    return labels, angle


def balanced_intrinsic_states(strata, spatial_domain, seed, n_states=3):
    rng = np.random.default_rng(seed)
    labels = np.empty(len(strata), dtype=np.int16)
    joint = np.char.add(np.char.add(strata.astype(str), "|"), spatial_domain.astype(str))
    for group in np.unique(joint):
        rows = np.flatnonzero(joint == group)
        order = rng.permutation(rows)
        labels[order] = np.arange(len(order)) % n_states
    return labels


def feature_background_association(x, *labels):
    """Sum one-way eta-squared over natural categorical background truths."""
    score = np.zeros(x.shape[1], dtype=np.float64)
    for label in labels:
        overall = x.mean(0)
        between = np.zeros(x.shape[1], dtype=np.float64)
        for group in np.unique(label):
            mask = label == group
            between += mask.mean() * np.square(x[mask].mean(0) - overall)
        score += between / (x.var(0) + 1e-8)
    return score


def stratified_subsample(niche, composition, max_cells, seed):
    if len(niche) <= max_cells:
        return np.arange(len(niche))
    rng = np.random.default_rng(seed)
    strata = np.char.add(np.char.add(niche.astype(str), "|"), composition.astype(str))
    chosen = []
    groups, counts = np.unique(strata, return_counts=True)
    quota = np.floor(max_cells * counts / counts.sum()).astype(int)
    remainder = max_cells - int(quota.sum())
    fractional = max_cells * counts / counts.sum() - quota
    quota[np.argsort(-fractional)[:remainder]] += 1
    for group, take in zip(groups, quota):
        rows = np.flatnonzero(strata == group)
        chosen.extend(rng.choice(rows, size=min(take, len(rows)), replace=False))
    return np.sort(np.asarray(chosen, dtype=int))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="/home/yancui/trajectory/EsophagusAtlas_CODEX.h5mu"
    )
    parser.add_argument("--sections", nargs="+", default=list(DEFAULT_SECTIONS))
    parser.add_argument("--out_dir", default="bench_be_protein/data_pairs")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--top_niches", type=int, default=6)
    parser.add_argument("--max_cells", type=int, default=6000)
    parser.add_argument("--n_salient_states", type=int, default=3)
    parser.add_argument("--n_salient_domains", type=int, default=3)
    parser.add_argument("--proteins_per_class", type=int, default=3)
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

            niche_labels, niche_counts = np.unique(niche, return_counts=True)
            top = niche_labels[np.argsort(-niche_counts)[: args.top_niches]]
            keep = np.flatnonzero(np.isin(niche, top))
            x, coords = x[keep], coords[keep]
            niche, composition, names = niche[keep], composition[keep], names[keep]
            take = stratified_subsample(
                niche, composition, args.max_cells, args.seed + section_i
            )
            x, coords = x[take], coords[take]
            niche, composition, names = niche[take], composition[take], names[take]

            background_strata = np.char.add(
                np.char.add(niche.astype(str), "|"), composition.astype(str)
            )
            sal_domain, angle = balanced_spatial_domains(
                coords,
                background_strata,
                args.seed + 100 + section_i,
                args.n_salient_domains,
            )
            sal_state = balanced_intrinsic_states(
                background_strata,
                sal_domain,
                args.seed + 200 + section_i,
                args.n_salient_states,
            )

            n_effect = (
                args.n_salient_states + args.n_salient_domains
            ) * args.proteins_per_class
            if n_effect > x.shape[1]:
                raise ValueError("Requested salient programs exceed protein panel")
            association = feature_background_association(x, niche, composition)
            effect_proteins = np.argsort(association, kind="mergesort")[:n_effect]
            split = args.n_salient_states * args.proteins_per_class
            state_proteins, domain_proteins = effect_proteins[:split], effect_proteins[split:]

            condition = x.copy()
            weights = np.linspace(1.0, 0.65, args.proteins_per_class)
            for state in range(args.n_salient_states):
                cols = state_proteins[
                    state * args.proteins_per_class : (state + 1) * args.proteins_per_class
                ]
                condition[np.ix_(sal_state == state, cols)] += (
                    args.effect_scale * weights[None, :]
                )
            for domain in range(args.n_salient_domains):
                cols = domain_proteins[
                    domain * args.proteins_per_class : (domain + 1) * args.proteins_per_class
                ]
                condition[np.ix_(sal_domain == domain, cols)] += (
                    args.effect_scale * weights[None, :]
                )

            obs_control = pd.DataFrame(index=[f"{section}_control_{x}" for x in names])
            obs_condition = pd.DataFrame(index=[f"{section}_condition_{x}" for x in names])
            for obs, condition_name, sample in (
                (obs_control, "control", f"{section}_control"),
                (obs_condition, "disease", f"{section}_condition"),
            ):
                obs["condition"] = condition_name
                obs["section"] = section
                obs["slice_id"] = sample
                obs["gt_background_composition"] = composition
                obs["gt_background_niche"] = niche
            obs_control["gt_salient_intrinsic"] = "control"
            obs_control["gt_salient_domain"] = "control"
            obs_condition["gt_salient_intrinsic"] = [f"state_{v}" for v in sal_state]
            obs_condition["gt_salient_domain"] = [f"domain_{v}" for v in sal_domain]
            obs = pd.concat([obs_control, obs_condition])

            pair_x = np.vstack([x, condition]).astype(np.float32)
            pair = ad.AnnData(
                X=pair_x,
                obs=obs,
                var=pd.DataFrame(index=pd.Index(proteins, name="protein")),
            )
            pair.layers["protein"] = pair_x.copy()
            pair.obsm["spatial"] = np.vstack([coords, coords]).astype(np.float32)

            diagnostics = {
                "nmi_sal_domain_vs_niche": float(
                    normalized_mutual_info_score(sal_domain, niche)
                ),
                "nmi_sal_domain_vs_composition": float(
                    normalized_mutual_info_score(sal_domain, composition)
                ),
                "nmi_sal_state_vs_niche": float(
                    normalized_mutual_info_score(sal_state, niche)
                ),
                "nmi_sal_state_vs_composition": float(
                    normalized_mutual_info_score(sal_state, composition)
                ),
                "nmi_sal_state_vs_sal_domain": float(
                    normalized_mutual_info_score(sal_state, sal_domain)
                ),
                "ari_sal_domain_vs_niche": float(
                    adjusted_rand_score(sal_domain, niche)
                ),
            }
            if max(abs(v) for v in diagnostics.values()) > 0.01:
                raise RuntimeError(
                    f"{section}: simulated factors are not independent: {diagnostics}"
                )
            pair.uns["be_protein_simulation"] = {
                "source": args.source,
                "section": section,
                "control_is_exact_source_standardized_codex": True,
                "condition_is_control_plus_additive_protein_programs": True,
                "background_intrinsic_truth": "neigh_sub",
                "background_spatial_truth": "neigh_name",
                "top_niches": top.tolist(),
                "n_salient_states": args.n_salient_states,
                "n_salient_domains": args.n_salient_domains,
                "projection_angle_radians": angle,
                "salient_balanced_within_background_joint_strata": True,
                "salient_intrinsic_balanced_within_spatial_domain": True,
                "effect_scale_standard_deviations": args.effect_scale,
                "state_proteins": {
                    f"state_{state}": proteins[
                        state_proteins[
                            state * args.proteins_per_class :
                            (state + 1) * args.proteins_per_class
                        ]
                    ].tolist()
                    for state in range(args.n_salient_states)
                },
                "domain_proteins": {
                    f"domain_{domain}": proteins[
                        domain_proteins[
                            domain * args.proteins_per_class :
                            (domain + 1) * args.proteins_per_class
                        ]
                    ].tolist()
                    for domain in range(args.n_salient_domains)
                },
                "mean_selected_protein_background_association": float(
                    association[effect_proteins].mean()
                ),
                **diagnostics,
                "seed": args.seed + section_i,
            }
            path = os.path.join(args.out_dir, f"be_{section}_paired.h5ad")
            pair.write_h5ad(path, compression="gzip")
            row = {
                "section": section,
                "n_cells_original": len(x),
                "n_cells_pair": pair.n_obs,
                "n_background_niches": len(np.unique(niche)),
                "n_background_compositions": len(np.unique(composition)),
                "effect_scale": args.effect_scale,
                **diagnostics,
                "path": os.path.abspath(path),
            }
            manifest.append(row)
            print(json.dumps(row), flush=True)
    pd.DataFrame(manifest).to_csv(os.path.join(args.out_dir, "manifest.csv"), index=False)


if __name__ == "__main__":
    main()
