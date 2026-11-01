"""Build exact same-section PEtracer control/condition pairs.

The control arm is the observed MERFISH section.  The condition arm has the
same cells, coordinates, annotations, and per-cell library sizes.  Only
high-quality lineage-resolved malignant cells receive a nonlinear expression
program keyed by their PEtracer clade.  Programs are shared across sections and
derived from consensus lineage-mark signatures, so related clades receive
related effects.  Clade labels are never written into the expression model's
training covariates; they are retained only as ground truth for evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import softmax
from sklearn.cluster import AgglomerativeClustering


DEFAULT_SOURCE = Path(
    "/home/yancui/Jill/fourteen/petracer_data/M1_tumor_tracing.h5td"
)


def read_treedata_parts(path: Path):
    """Read only the AnnData-compatible blocks needed by this benchmark."""
    with h5py.File(path, "r") as handle:
        obs = ad.io.read_elem(handle["obs"])
        var = ad.io.read_elem(handle["var"])
        counts = ad.io.read_elem(handle["layers/counts"]).tocsr()
        spatial = np.asarray(handle["obsm/spatial"], dtype=np.float32)
        characters = ad.io.read_elem(handle["obsm/characters"])
    if len(obs) != counts.shape[0] or len(obs) != len(spatial):
        raise ValueError("PEtracer matrices and annotations are not row aligned")
    return obs, var, counts, spatial, characters


def clean_clade(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    out = np.full(len(values), "NA", dtype=object)
    valid = np.isfinite(numeric)
    out[valid] = np.asarray([str(int(x)) for x in numeric[valid]], dtype=object)
    return out.astype(str)


def lineage_mask(obs: pd.DataFrame, clade: np.ndarray, detection: float) -> np.ndarray:
    return (
        (obs["cell_subtype"].astype(str).to_numpy() == "Malignant")
        & (obs["type"].astype(str).to_numpy() == "tracing")
        & (pd.to_numeric(obs["detection_rate"], errors="coerce").to_numpy() >= detection)
        & (clade != "NA")
    )


def consensus_signatures(
    characters: pd.DataFrame, clade: np.ndarray, eligible: np.ndarray
) -> tuple[list[str], np.ndarray]:
    """One-hot consensus lineage-mark signature for each clade."""
    values = characters.to_numpy()
    clades = sorted(np.unique(clade[eligible]), key=int)
    consensus = np.zeros((len(clades), values.shape[1]), dtype=np.int16)
    for row, value in enumerate(clades):
        take = values[(clade == value) & eligible]
        for column in range(values.shape[1]):
            detected = take[:, column]
            detected = detected[detected >= 0].astype(int)
            if len(detected):
                consensus[row, column] = np.bincount(detected, minlength=9).argmax()
            else:
                consensus[row, column] = -1
    one_hot = np.zeros((len(clades), values.shape[1] * 9), dtype=np.float32)
    for row in range(len(clades)):
        for column, state in enumerate(consensus[row]):
            if state >= 0:
                one_hot[row, column * 9 + int(state)] = 1.0
    return clades, one_hot


def choose_effect_genes(
    counts: sp.csr_matrix, eligible: np.ndarray, n_effect: int, n_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = counts[eligible].astype(np.float64)
    total = np.asarray(rows.sum(1)).ravel()
    normalized = rows.multiply(1e4 / np.maximum(total, 1.0)[:, None])
    log = normalized.copy()
    log.data = np.log1p(log.data)
    mean = np.asarray(log.mean(0)).ravel()
    second = np.asarray(log.power(2).mean(0)).ravel()
    variance = np.maximum(second - mean**2, 0.0)
    prevalence = np.asarray((rows > 0).mean(0)).ravel()
    score = variance * np.sqrt(np.maximum(prevalence, 1e-6))
    supported = np.flatnonzero(prevalence >= 0.03)
    order = supported[np.argsort(-score[supported], kind="mergesort")]
    needed = min(len(order), n_effect + n_state)
    if needed < 8:
        raise ValueError("Too few expressed MERFISH genes for a lineage simulation")
    n_effect = min(n_effect, max(4, needed - min(n_state, needed // 4)))
    effect = order[:n_effect]
    state = order[n_effect:needed]
    if not len(state):
        state = order[: min(4, len(order))]
    malignant_mean = np.asarray(rows.mean(0)).ravel() + 0.05
    malignant_mean /= malignant_mean.sum()
    return effect, state, malignant_mean


def make_effects(
    signatures: np.ndarray,
    n_genes: int,
    effect_genes: np.ndarray,
    n_families: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_clades = len(signatures)
    n_families = min(n_families, n_clades)
    families = AgglomerativeClustering(
        n_clusters=n_families, metric="cosine", linkage="average"
    ).fit_predict(signatures)
    projection = rng.normal(0, 1, (signatures.shape[1], len(effect_genes)))
    inherited = signatures @ projection / np.sqrt(max(signatures.shape[1], 1))
    inherited -= inherited.mean(1, keepdims=True)
    inherited /= np.maximum(inherited.std(1, keepdims=True), 1e-6)
    # Explicit family and clade marker blocks keep the truth clusterable.
    compact = 0.20 * inherited
    family_slots = 2
    family_width = min(n_families * family_slots, len(effect_genes))
    for row, family in enumerate(families):
        left = int(family) * family_slots
        right = min(left + family_slots, family_width)
        compact[row, left:right] += 1.8
    clade_start = family_width
    if len(effect_genes) - clade_start < n_clades:
        raise ValueError("Need family markers plus one marker per clade")
    for row in range(n_clades):
        compact[row, clade_start + row] += 1.45
        compact[row, clade_start + ((row + n_clades // 2) % n_clades)] -= 0.75
    effects = np.zeros((n_clades, n_genes), dtype=np.float32)
    effects[:, effect_genes] = compact.astype(np.float32)
    return effects, families.astype(np.int16)


def cell_state_score(
    counts: sp.csr_matrix, eligible: np.ndarray, state_genes: np.ndarray
) -> np.ndarray:
    score = np.zeros(counts.shape[0], dtype=np.float32)
    block = counts[eligible][:, state_genes].toarray().astype(np.float32)
    block = np.log1p(block)
    block -= block.mean(0, keepdims=True)
    block /= np.maximum(block.std(0, keepdims=True), 1e-4)
    raw = block.mean(1)
    raw = (raw - np.median(raw)) / max(float(np.std(raw)), 1e-4)
    score[eligible] = np.tanh(raw).astype(np.float32)
    return score


def simulate_condition(
    counts: sp.csr_matrix,
    eligible: np.ndarray,
    clade: np.ndarray,
    clade_order: list[str],
    effects: np.ndarray,
    state_score: np.ndarray,
    malignant_mean: np.ndarray,
    strength: float,
    smoothing: float,
    rng: np.random.Generator,
) -> sp.csr_matrix:
    """Non-additive multinomial resampling with exact library-size preservation."""
    condition = counts.tolil(copy=True)
    clade_index = {value: index for index, value in enumerate(clade_order)}
    for row in np.flatnonzero(eligible):
        observed = counts.getrow(row).toarray().ravel().astype(np.float64)
        library = int(np.rint(observed.sum()))
        if library <= 0:
            continue
        base = observed + smoothing * library * malignant_mean
        log_probability = np.log(np.maximum(base, 1e-12))
        program = effects[clade_index[clade[row]]]
        # The effect changes with the cell's pre-existing transcriptional state,
        # making the condition neither additive nor a clade-wide fixed fold change.
        amplitude = strength * (0.85 + 0.35 * state_score[row])
        interaction = 0.30 * strength * state_score[row] * np.sign(program) * program**2
        probability = softmax(log_probability + amplitude * program + interaction)
        expected = library * probability
        allocation = np.floor(expected).astype(np.int32)
        remainder = int(library - allocation.sum())
        if remainder:
            priority = expected - allocation + rng.uniform(0, 1e-10, len(expected))
            allocation[np.argsort(-priority)[:remainder]] += 1
        condition[row] = allocation
    return condition.tocsr().astype(np.int32)


def boundary_bins(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = np.isfinite(numeric)
    labels = np.full(len(values), "unknown", dtype=object)
    if finite.sum() >= 4:
        edges = np.unique(np.quantile(numeric[finite], [0, 0.25, 0.5, 0.75, 1]))
        if len(edges) >= 3:
            labels[finite] = pd.cut(
                numeric[finite], bins=edges, include_lowest=True, duplicates="drop"
            ).astype(str)
    return labels.astype(str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=Path("data_m1t1_pairs"))
    parser.add_argument("--tumor", default="1")
    parser.add_argument("--sections", nargs="*", default=["M1-1", "M1-2", "M1-3", "M1-4"])
    parser.add_argument("--detection-threshold", type=float, default=0.6)
    parser.add_argument("--n-effect-genes", type=int, default=54)
    parser.add_argument("--n-state-genes", type=int, default=10)
    parser.add_argument("--n-lineage-families", type=int, default=6)
    parser.add_argument("--effect-strength", type=float, default=1.40)
    parser.add_argument("--profile-smoothing", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    obs, var, counts, spatial, characters = read_treedata_parts(args.source)
    tumor = obs["tumor"].astype(str).to_numpy() == str(args.tumor)
    requested = np.isin(obs["sample"].astype(str).to_numpy(), args.sections)
    keep = tumor & requested
    if not keep.any():
        raise ValueError("No cells match the requested tumor and sections")

    cohort_obs = obs.loc[keep].copy()
    cohort_counts = counts[keep]
    cohort_characters = characters.loc[keep].copy()
    cohort_clade = clean_clade(cohort_obs["clade"])
    cohort_eligible = lineage_mask(
        cohort_obs, cohort_clade, args.detection_threshold
    )
    clade_order, signatures = consensus_signatures(
        cohort_characters, cohort_clade, cohort_eligible
    )
    effect_genes, state_genes, malignant_mean = choose_effect_genes(
        cohort_counts,
        cohort_eligible,
        args.n_effect_genes,
        args.n_state_genes,
    )
    rng = np.random.default_rng(args.seed)
    effects, families = make_effects(
        signatures,
        counts.shape[1],
        effect_genes,
        args.n_lineage_families,
        rng,
    )
    family_by_clade = {
        value: f"family_{int(families[index])}"
        for index, value in enumerate(clade_order)
    }
    gene_names = var.index.astype(str).to_numpy()
    manifest = []

    global_positions = np.flatnonzero(keep)
    for section_i, section in enumerate(args.sections):
        local = cohort_obs["sample"].astype(str).to_numpy() == section
        if not local.any():
            continue
        one_obs = cohort_obs.loc[local].copy()
        one_counts = cohort_counts[local].astype(np.int32)
        one_spatial = spatial[global_positions[local]]
        one_clade = cohort_clade[local]
        one_eligible = cohort_eligible[local]
        state = cell_state_score(one_counts, one_eligible, state_genes)
        section_rng = np.random.default_rng(args.seed + 1000 + section_i)
        condition_counts = simulate_condition(
            one_counts,
            one_eligible,
            one_clade,
            clade_order,
            effects,
            state,
            malignant_mean,
            args.effect_strength,
            args.profile_smoothing,
            section_rng,
        )
        before_library = np.asarray(one_counts.sum(1)).ravel()
        after_library = np.asarray(condition_counts.sum(1)).ravel()
        if not np.array_equal(before_library, after_library):
            raise RuntimeError(f"{section}: condition changed per-cell library sizes")

        n = len(one_obs)
        control_obs = one_obs.copy()
        condition_obs = one_obs.copy()
        pair_ids = one_obs.index.astype(str).to_numpy()
        control_obs.index = pd.Index([f"{section}_control_{x}" for x in pair_ids])
        condition_obs.index = pd.Index([f"{section}_condition_{x}" for x in pair_ids])
        niche = one_obs["cell_neighborhood"].astype(str).to_numpy()
        subtype = one_obs["cell_subtype"].astype(str).to_numpy()
        boundary = boundary_bins(one_obs["tumor_boundary_dist"])
        clade_truth = np.asarray(
            [f"clade_{x}" if ok else "not_lineage_target" for x, ok in zip(one_clade, one_eligible)]
        )
        family_truth = np.asarray(
            [family_by_clade[x] if ok else "not_lineage_target" for x, ok in zip(one_clade, one_eligible)]
        )
        for frame, arm, graph in (
            (control_obs, "control", f"{section}_control"),
            (condition_obs, "disease", f"{section}_condition"),
        ):
            frame["condition"] = arm
            frame["section_id"] = section
            frame["graph_id"] = graph
            frame["pair_id"] = pair_ids
            frame["gt_background_niche"] = niche
            frame["gt_background_celltype"] = subtype
            frame["gt_background_boundary_bin"] = boundary
            frame["lineage_effect_eligible"] = one_eligible
            frame["gt_lineage_clade"] = clade_truth
            frame["gt_lineage_family"] = family_truth
        control_obs["lineage_effect_applied"] = False
        condition_obs["lineage_effect_applied"] = one_eligible
        combined_obs = pd.concat([control_obs, condition_obs], axis=0)
        paired_counts = sp.vstack([one_counts, condition_counts], format="csr")
        paired = ad.AnnData(X=paired_counts.copy(), obs=combined_obs, var=var.copy())
        paired.layers["counts"] = paired_counts.copy()
        paired.obsm["spatial"] = np.vstack([one_spatial, one_spatial]).astype(np.float32)
        paired.uns["petracer_lineage_simulation"] = {
            "source": str(args.source.resolve()),
            "tumor": str(args.tumor),
            "section": section,
            "paired_cells_are_exact": True,
            "coordinates_are_exact": True,
            "cell_annotations_are_exact": True,
            "per_cell_library_size_is_exact": True,
            "effect_model": "state-dependent lineage-mark program + multinomial resampling",
            "effect_strength": args.effect_strength,
            "profile_smoothing": args.profile_smoothing,
            "detection_threshold": args.detection_threshold,
            "clades": clade_order,
            "lineage_families": family_by_clade,
            "effect_genes": gene_names[effect_genes].tolist(),
            "state_interaction_genes": gene_names[state_genes].tolist(),
            "training_covariates": ["condition"],
            "evaluation_only": ["gt_lineage_clade", "gt_lineage_family"],
            "seed": args.seed + 1000 + section_i,
        }
        output = args.out_dir / f"petracer_M1T1_{section}_paired.h5ad"
        paired.write_h5ad(output, compression="gzip")
        changed = np.asarray((one_counts != condition_counts).sum(1)).ravel() > 0
        row = {
            "section": section,
            "n_original_cells": n,
            "n_paired_cells": 2 * n,
            "n_lineage_cells": int(one_eligible.sum()),
            "n_clades": int(np.unique(one_clade[one_eligible]).size),
            "n_lineage_families": int(len(np.unique(family_truth[one_eligible]))),
            "fraction_all_cells_changed": float(changed.mean()),
            "fraction_lineage_cells_changed": float(changed[one_eligible].mean()),
            "library_size_exact": bool(np.array_equal(before_library, after_library)),
            "path": str(output.resolve()),
        }
        manifest.append(row)
        print(json.dumps(row), flush=True)

    with open(args.out_dir / "shared_lineage_programs.json", "w") as handle:
        json.dump(
            {
                "clade_order": clade_order,
                "family_by_clade": family_by_clade,
                "effect_genes": gene_names[effect_genes].tolist(),
                "state_interaction_genes": gene_names[state_genes].tolist(),
                "effect_strength": args.effect_strength,
                "seed": args.seed,
            },
            handle,
            indent=2,
        )
    pd.DataFrame(manifest).to_csv(args.out_dir / "manifest.csv", index=False)


if __name__ == "__main__":
    main()
