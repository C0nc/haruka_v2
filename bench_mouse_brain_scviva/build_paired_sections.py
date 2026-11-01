"""Create paired mouse-brain sections with spatial composition shifts.

The control arm is the exact original section. The condition arm keeps the same
coordinates and background annotations, but each salient spatial domain applies
a different additive shift to the local cell-type proportions. Its observed
expression is sampled from real control cells of the requested type and the same
anatomical subregion. Consequently the simulation changes cell composition
without inventing an additive gene program or destroying background anatomy.
"""

from __future__ import annotations

import argparse
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def reconstruct_counts(log_x: np.ndarray, barcode_count: np.ndarray) -> np.ndarray:
    raw = np.expm1(log_x) * barcode_count[:, None] / 1e4
    rounded = np.rint(raw)
    max_error = float(np.max(np.abs(raw - rounded)))
    if max_error > 1e-4:
        raise ValueError(f"Raw-count inversion failed; max rounding error={max_error:.3g}")
    return rounded.astype(np.int32)


def balanced_spatial_domains(coords, subregion, seed, n_domains=3):
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(0, np.pi))
    projection = coords[:, 0] * np.cos(angle) + coords[:, 1] * np.sin(angle)
    labels = np.empty(len(coords), dtype=np.int16)
    for group in np.unique(subregion):
        rows = np.flatnonzero(subregion == group)
        # Rank bins give equal domain proportions inside each subregion while
        # preserving spatial contiguity along a randomly oriented tissue axis.
        order = rows[np.argsort(projection[rows], kind="mergesort")]
        labels[order] = np.minimum(
            (np.arange(len(order)) * n_domains) // len(order), n_domains - 1
        )
    return labels, angle


def choose_domain_promotions(cell_type, subregion, n_domains):
    """Choose distinct cell types supported in every anatomical subregion."""
    categories, counts = np.unique(cell_type, return_counts=True)
    order = np.argsort(-counts, kind="mergesort")
    categories, counts = categories[order], counts[order]
    minimum_support = np.zeros(len(categories), dtype=np.float64)
    for code, category in enumerate(categories):
        support = []
        for region in np.unique(subregion):
            rows = subregion == region
            support.append(np.mean(cell_type[rows] == category))
        minimum_support[code] = min(support)
    eligible = np.flatnonzero(minimum_support > 0)
    eligible = eligible[np.argsort(-minimum_support[eligible], kind="mergesort")]
    if len(eligible) < n_domains:
        raise ValueError(
            f"Only {len(eligible)} cell types occur in every subregion; "
            f"need {n_domains} domain-specific promoted types"
        )
    return categories, counts, eligible[:n_domains], minimum_support


def exact_composition_sample(probabilities, n, rng):
    """Allocate exactly n labels with largest-remainder integer rounding."""
    expected = np.asarray(probabilities, dtype=np.float64) * int(n)
    allocation = np.floor(expected).astype(np.int64)
    remainder = int(n - allocation.sum())
    if remainder:
        priority = expected - allocation + rng.uniform(0, 1e-12, len(expected))
        allocation[np.argsort(-priority)[:remainder]] += 1
    labels = np.repeat(np.arange(len(probabilities)), allocation)
    rng.shuffle(labels)
    return labels


def simulate_condition(
    counts,
    subregion,
    original_cell_type,
    domain,
    categories,
    promoted_codes,
    shift_fraction,
    rng,
):
    """Apply additive composition shifts using a minimum-change transport."""
    category_index = {value: i for i, value in enumerate(categories)}
    original_code = np.asarray([category_index[x] for x in original_cell_type])
    observed_code = original_code.copy()
    donor_index = np.arange(len(counts), dtype=np.int64)
    shift_records = []

    for region in np.unique(subregion):
        region_rows = np.flatnonzero(subregion == region)
        donor_pools = {
            code: region_rows[original_code[region_rows] == code]
            for code in range(len(categories))
        }
        for salient_domain in np.unique(domain):
            rows = np.flatnonzero((subregion == region) & (domain == salient_domain))
            current = np.bincount(
                original_code[rows], minlength=len(categories)
            ).astype(np.int64)
            baseline = current / current.sum()
            promoted = int(promoted_codes[int(salient_domain)])
            # Exact additive form: q = p + alpha * (one_hot(promoted) - p).
            target = (1.0 - shift_fraction) * baseline
            target[promoted] += shift_fraction
            desired_labels = exact_composition_sample(target, len(rows), rng)
            desired = np.bincount(
                desired_labels, minlength=len(categories)
            ).astype(np.int64)

            surplus = current - desired
            changed_rows = []
            for code in np.flatnonzero(surplus > 0):
                pool = rows[original_code[rows] == code].copy()
                rng.shuffle(pool)
                changed_rows.extend(pool[: surplus[code]])
            destination_codes = np.repeat(
                np.arange(len(categories)), np.maximum(-surplus, 0)
            )
            rng.shuffle(destination_codes)
            changed_rows = np.asarray(changed_rows, dtype=np.int64)
            if len(changed_rows) != len(destination_codes):
                raise RuntimeError("Composition transport is not mass preserving")
            observed_code[changed_rows] = destination_codes

            # Unchanged cells retain their exact expression. Only relabelled cells
            # draw a real background donor of the destination type and same region.
            for code in np.unique(destination_codes):
                destinations = changed_rows[destination_codes == code]
                pool = donor_pools[int(code)]
                if not len(pool):
                    raise RuntimeError(
                        f"No {categories[int(code)]!r} donors in subregion {region!r}"
                    )
                donor_index[destinations] = rng.choice(
                    pool, size=len(destinations), replace=True
                )

            realized = np.bincount(
                observed_code[rows], minlength=len(categories)
            ) / len(rows)
            shift_records.append(
                {
                    "subregion": str(region),
                    "domain": f"domain_{int(salient_domain)}",
                    "promoted_cell_type": str(categories[promoted]),
                    "n_cells": int(len(rows)),
                    "n_cells_changed": int(len(changed_rows)),
                    "baseline": baseline.tolist(),
                    "target": target.tolist(),
                    "realized": realized.tolist(),
                    "l1_shift": float(np.abs(realized - baseline).sum()),
                }
            )

    condition = counts[donor_index].copy()
    observed_cell_type = categories[observed_code]
    return condition, observed_cell_type, donor_index, shift_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="/home/yancui/virtualorganoid/notebook/aging_coronal_spatial_mouse.h5ad",
    )
    parser.add_argument("--out_dir", default="data_composition_pairs")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--n_domains", type=int, default=3)
    parser.add_argument(
        "--shift_fraction",
        type=float,
        default=0.5,
        help="Fraction mixed into a domain-specific promoted cell type.",
    )
    parser.add_argument(
        "--sections",
        nargs="*",
        help="Optional slide IDs; by default every section is generated.",
    )
    args = parser.parse_args()

    source = ad.read_h5ad(args.source)
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    available = source.obs["slide_id"].astype(str).unique()
    sections = available if not args.sections else np.asarray(args.sections)
    unknown = sorted(set(sections).difference(available))
    if unknown:
        raise KeyError(f"Unknown sections: {unknown}")

    for section_i, section in enumerate(sections):
        one = source[source.obs["slide_id"].astype(str) == section].copy()
        log_x = one.X.toarray() if sp.issparse(one.X) else np.asarray(one.X)
        counts = reconstruct_counts(log_x, one.obs["barcodeCount"].to_numpy(float))
        subregion = one.obs["subregion"].astype(str).to_numpy()
        original_cell_type = one.obs["celltype"].astype(str).to_numpy()
        coords = np.asarray(one.obsm["spatial"], dtype=np.float64)
        domain, angle = balanced_spatial_domains(
            coords, subregion, args.seed + section_i, args.n_domains
        )

        categories, category_counts, promoted_codes, minimum_support = (
            choose_domain_promotions(
                original_cell_type, subregion, args.n_domains
            )
        )
        rng = np.random.default_rng(args.seed + 1000 + section_i)
        condition, observed_cell_type, donor_index, shift_records = simulate_condition(
            counts,
            subregion,
            original_cell_type,
            domain,
            categories,
            promoted_codes,
            args.shift_fraction,
            rng,
        )

        n = one.n_obs
        obs_control = one.obs.copy()
        obs_condition = one.obs.copy()
        obs_control.index = pd.Index([f"{section}_control_{x}" for x in one.obs_names])
        obs_condition.index = pd.Index([f"{section}_condition_{x}" for x in one.obs_names])
        for obs, arm, slice_id in (
            (obs_control, "control", f"{section}_control"),
            (obs_condition, "disease", f"{section}_condition"),
        ):
            obs["condition"] = arm
            obs["timepoint"] = section
            obs["slice_id"] = slice_id
            obs["gt_background_subregion"] = subregion
            obs["gt_background_celltype"] = original_cell_type
        obs_control["gt_sal_domain"] = "control"
        obs_condition["gt_sal_domain"] = np.asarray([f"domain_{x}" for x in domain])
        obs_control["observed_celltype"] = original_cell_type
        obs_condition["observed_celltype"] = observed_cell_type
        obs_control["expression_donor_index"] = np.arange(n, dtype=np.int64)
        obs_condition["expression_donor_index"] = donor_index
        if "barcodeCount" in obs_condition:
            obs_condition["barcodeCount"] = condition.sum(1)
        if "transcript_count" in obs_condition:
            obs_condition["transcript_count"] = condition.sum(1)
        obs = pd.concat([obs_control, obs_condition], axis=0)

        pair_counts = sp.csr_matrix(np.vstack([counts, condition]))
        pair = ad.AnnData(X=pair_counts, obs=obs, var=one.var.copy())
        pair.layers["counts"] = pair_counts.copy()
        pair.obsm["spatial"] = np.vstack([coords, coords]).astype(np.float32)

        nmi = float(normalized_mutual_info_score(subregion, domain))
        ari = float(adjusted_rand_score(subregion, domain))
        if nmi > 0.01 or abs(ari) > 0.01:
            raise RuntimeError(
                f"{section}: domain is not independent enough: NMI={nmi}, ARI={ari}"
            )
        effect_by_domain = {
            f"domain_{d}": {
                "promoted_cell_type": str(categories[promoted_codes[d]]),
                "shift_formula": "q = (1 - alpha) * p + alpha * one_hot(promoted)",
            }
            for d in range(args.n_domains)
        }
        mean_l1 = float(np.mean([record["l1_shift"] for record in shift_records]))
        pair.uns["paired_salient_simulation"] = {
            "source": args.source,
            "section": section,
            "control_is_exact_original_raw_counts": True,
            "condition_is_control_plus_poisson_effect": False,
            "condition_expression_model": (
                "empirical control donor sampled within background subregion and "
                "simulated observed cell type"
            ),
            "raw_count_inverse": "round(expm1(X) * barcodeCount / 10000)",
            "n_domains": args.n_domains,
            "projection_angle_radians": angle,
            "domain_balanced_within_subregion": True,
            "domain_subregion_nmi": nmi,
            "domain_subregion_ari": ari,
            "cell_type_categories": categories.tolist(),
            "cell_type_counts": category_counts.tolist(),
            "shift_fraction": args.shift_fraction,
            "domain_promoted_cell_types": categories[promoted_codes].tolist(),
            "promoted_cell_type_minimum_subregion_support": minimum_support[
                promoted_codes
            ].tolist(),
            "effects_by_domain": effect_by_domain,
            "stratum_composition_shifts_json": json.dumps(shift_records),
            "mean_realized_l1_shift": mean_l1,
            "seed": args.seed + section_i,
        }
        path = os.path.join(args.out_dir, f"mouse_brain_{section}_paired.h5ad")
        pair.write_h5ad(path, compression="gzip")
        row = {
            "section": section,
            "n_cells_original": n,
            "n_cells_pair": pair.n_obs,
            "n_subregions": int(np.unique(subregion).size),
            "n_cell_types": int(len(categories)),
            "domain_subregion_nmi": nmi,
            "domain_subregion_ari": ari,
            "shift_fraction": args.shift_fraction,
            "fraction_cells_changed": float(
                np.mean(observed_cell_type != original_cell_type)
            ),
            "mean_realized_l1_shift": mean_l1,
            "path": os.path.abspath(path),
        }
        manifest.append(row)
        print(json.dumps(row), flush=True)

    pd.DataFrame(manifest).to_csv(os.path.join(args.out_dir, "manifest.csv"), index=False)


if __name__ == "__main__":
    main()
