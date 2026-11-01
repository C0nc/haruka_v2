#!/usr/bin/env python3
"""Package a control/condition identifiability audit for the real PEtracer data."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
DATA = Path("/home/yancui/Jill/fourteen/petracer_data")
BENCH = Path("/home/yancui/haruka_v2/bench_petracer_lineage")
RUNNER = Path("/home/yancui/haruka_v2/bench_kras_xenium/run_haruka_separate_batch_heads.py")


def decode(values):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in values]


def categorical(handle: h5py.File, key: str) -> tuple[list[str], np.ndarray]:
    group = handle[f"obs/{key}"]
    categories = decode(group["categories"][:])
    codes = np.asarray(group["codes"][:], dtype=int)
    return categories, codes


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for directory in ["tables", "figures", "logs", "commands", "hashes"]:
        (OUT / directory).mkdir(parents=True, exist_ok=True)

    source_rows = []
    sample_rows = []
    sample_tumor_rows = []
    sample_celltype_rows = []
    column_rows = []
    feature_sets: dict[str, set[str]] = {}
    audit_lines = []

    for path in sorted(DATA.glob("*.h5td")):
        with h5py.File(path, "r") as handle:
            mouse = path.name.split("_")[0]
            n_obs, n_vars = map(int, handle["X"].shape)
            features = decode(handle["var/_index"][:])
            feature_sets[mouse] = set(features)
            columns = decode(handle["obs"].attrs["column-order"])
            for column in columns:
                column_rows.append({"source": path.name, "column": column})

            samples, sample_codes = categorical(handle, "sample")
            tumors, tumor_codes = categorical(handle, "tumor")
            types, type_codes = categorical(handle, "type")
            celltypes, celltype_codes = categorical(handle, "cell_subtype")
            spatial = handle["obsm/spatial"]
            spatial_probe = spatial[::1000]
            counts_data = handle["layers/counts/data"][:]
            source_rows.append({
                "source": path.name,
                "path": str(path),
                "mouse": mouse,
                "n_cells": n_obs,
                "n_features": n_vars,
                "assay": "targeted MERFISH RNA",
                "raw_counts_location": "layers/counts",
                "raw_counts_dtype": str(counts_data.dtype),
                "raw_nonzero_min": int(counts_data.min()),
                "raw_nonzero_max": int(counts_data.max()),
                "raw_counts_integer": bool(np.issubdtype(counts_data.dtype, np.integer)),
                "X_dtype": str(handle["X"].dtype),
                "X_is_raw_counts": False,
                "spatial_key": "obsm/spatial",
                "spatial_finite_probe": bool(np.isfinite(spatial_probe).all()),
                "sample_key": "sample",
                "cell_type_key": "cell_subtype",
                "condition_key_present": False,
                "n_sections": len(samples),
                "sections": ";".join(samples),
                "n_tumor_labels": len(tumors),
            })

            for sample_index, sample in enumerate(samples):
                selected = sample_codes == sample_index
                sample_rows.append({
                    "source": path.name,
                    "mouse": mouse,
                    "section": sample,
                    "n_cells": int(selected.sum()),
                    "injection_day_after_PEmax": {"M1": 12, "M2": 6, "M3": 6}[mouse],
                    "days_post_injection_at_sacrifice": {"M1": 39, "M2": 28, "M3": 35}[mouse],
                    "recipient": {
                        "M1": "Balbc/J x SELECTIV",
                        "M2": "Balbc/J x [SELECTIV x CMV-Cre]",
                        "M3": "Balbc/J x [SELECTIV x CMV-Cre]",
                    }[mouse],
                    "feature_panel_size": n_vars,
                })
                for tumor_index, tumor in enumerate(tumors):
                    count = int(np.sum(selected & (tumor_codes == tumor_index)))
                    if count:
                        sample_tumor_rows.append({"source": path.name, "section": sample, "tumor": tumor, "n_cells": count})
                for celltype_index, celltype in enumerate(celltypes):
                    count = int(np.sum(selected & (celltype_codes == celltype_index)))
                    if count:
                        sample_celltype_rows.append({"source": path.name, "section": sample, "cell_type": celltype, "n_cells": count})

            audit_lines.extend([
                f"{path.name}: shape={n_obs}x{n_vars}",
                f"  sections={dict(zip(samples, np.bincount(sample_codes, minlength=len(samples)).astype(int).tolist()))}",
                f"  type={dict(zip(types, np.bincount(type_codes, minlength=len(types)).astype(int).tolist()))}",
                f"  raw counts: layers/counts, dtype={counts_data.dtype}, nonzero range={counts_data.min()}..{counts_data.max()}",
                f"  X: dense {handle['X'].dtype} scaled/non-count matrix; spatial={spatial.shape}, finite probe={np.isfinite(spatial_probe).all()}",
                "  explicit condition/control column: absent",
            ])

    sources = pd.DataFrame(source_rows)
    samples = pd.DataFrame(sample_rows)
    sample_tumor = pd.DataFrame(sample_tumor_rows)
    sample_celltype = pd.DataFrame(sample_celltype_rows)
    columns = pd.DataFrame(column_rows)
    sources.to_csv(OUT / "tables/source_files.csv", index=False)
    samples.to_csv(OUT / "tables/sample_inventory.csv", index=False)
    sample_tumor.to_csv(OUT / "tables/sample_by_tumor.csv", index=False)
    sample_celltype.to_csv(OUT / "tables/sample_by_cell_type.csv", index=False)
    columns.to_csv(OUT / "tables/observation_columns.csv", index=False)

    all_union = set.union(*feature_sets.values())
    all_intersection = set.intersection(*feature_sets.values())
    pair_rows = []
    mice = sorted(feature_sets)
    for i, left in enumerate(mice):
        for right in mice[i + 1:]:
            pair_rows.append({
                "left": left,
                "right": right,
                "left_features": len(feature_sets[left]),
                "right_features": len(feature_sets[right]),
                "intersection": len(feature_sets[left] & feature_sets[right]),
                "left_only": len(feature_sets[left] - feature_sets[right]),
                "right_only": len(feature_sets[right] - feature_sets[left]),
            })
    pd.DataFrame(pair_rows).to_csv(OUT / "tables/feature_panel_overlap.csv", index=False)
    pd.DataFrame({"feature": sorted(all_intersection)}).to_csv(OUT / "tables/common_features_all_mice.csv", index=False)
    pd.DataFrame({"feature": sorted(all_union)}).to_csv(OUT / "tables/union_features_all_mice.csv", index=False)

    candidates = pd.DataFrame([
        {
            "candidate": "untreated_or_sham_vs_tumor",
            "available": False,
            "defensible": False,
            "reason": "No untreated, vehicle, sham, or non-tumor-bearing experimental arm is present in any source file.",
        },
        {
            "candidate": "M1_vs_M2_M3",
            "available": True,
            "defensible": False,
            "reason": "Mouse identity, injection preparation day (12 vs 6), recipient genotype, post-injection endpoint (39 vs 28/35 d), section, and for M3 the feature panel are confounded.",
        },
        {
            "candidate": "M2_28d_vs_M3_35d",
            "available": True,
            "defensible": False,
            "reason": "One mouse per time point makes time perfectly confounded with animal; M2 has one section and M3 four; feature panels differ (124 vs 175).",
        },
        {
            "candidate": "within_tumor_vs_outside_tumor",
            "available": True,
            "defensible": False,
            "reason": "Observational spatial location with strong cell-type/support differences, not a randomized or matched control-condition intervention.",
        },
        {
            "candidate": "malignant_vs_nonmalignant",
            "available": True,
            "defensible": False,
            "reason": "Cell identity contrast with non-overlapping biological support, not treatment response; state-matched counterfactual anchors are not credible.",
        },
        {
            "candidate": "lineage_family_or_clade",
            "available": True,
            "defensible": False,
            "reason": "Post hoc lineage evaluation labels; using them as the model condition would violate the requested unsupervised contract.",
        },
        {
            "candidate": "tracing_vs_nontracing_or_detection_class",
            "available": True,
            "defensible": False,
            "reason": "Lineage-readout/QC status rather than a biological intervention; highly coupled to detection quality and malignant identity.",
        },
    ])
    candidates.to_csv(OUT / "tables/condition_candidate_audit.csv", index=False)

    code_audit = pd.DataFrame([
        {
            "path": str(BENCH / "build_paired_lineage.py"),
            "classification": "simulated benchmark generator",
            "evidence": "Docstring states observed sections are copied into control and condition; condition expression is synthesized by lineage-keyed programs; writes petracer_lineage_simulation and gt_* fields.",
            "usable_for_real_biological_claims": False,
        },
        {
            "path": str(BENCH / "build_joint_four_replicates.py"),
            "classification": "combiner of simulated paired files",
            "evidence": "Loads data_m1t1_pairs_22k, retains one copied control and four simulated disease arms.",
            "usable_for_real_biological_claims": False,
        },
        {
            "path": str(BENCH / "score_lineage_pair.py"),
            "classification": "simulation evaluator",
            "evidence": "Evaluates gt_lineage_family/gt_lineage_clade and paired benchmark outputs.",
            "usable_for_real_biological_claims": False,
        },
        {
            "path": str(BENCH / "analyze_salient_spatial_clusters.py"),
            "classification": "simulation-specific post-analysis",
            "evidence": "Requires exact pair_id controls, simulation JSON, response_* programs, and gt_* labels.",
            "usable_for_real_biological_claims": False,
        },
        {
            "path": "/home/yancui/haruka_v2/bench_real_discovery/score_petracer_lineage_distance_cc3.py",
            "classification": "lineage-distance scorer on simulated-pair embeddings",
            "evidence": "Reads original character matrix, but representations and condition eligibility come from paired_22k simulated disease data.",
            "usable_for_real_biological_claims": False,
        },
        {
            "path": str(RUNNER),
            "classification": "skill-designated Haruka model runner",
            "evidence": "Appropriate only after an explicit identifiable binary condition and valid sample-bounded input are prepared.",
            "usable_for_real_biological_claims": False,
        },
    ])
    code_audit.to_csv(OUT / "tables/code_assumption_audit.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), constrained_layout=True)
    palette = {"M1": "#4C78A8", "M2": "#F58518", "M3": "#54A24B"}
    axes[0].bar(samples["section"], samples["n_cells"], color=[palette[x] for x in samples["mouse"]])
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Cells")
    axes[0].set_title("Original PEtracer physical sections")
    for mouse, color in palette.items():
        axes[0].bar([], [], color=color, label=mouse)
    axes[0].legend(frameon=False)

    timeline = samples.drop_duplicates("mouse").sort_values("mouse")
    y = np.arange(len(timeline))[::-1]
    axes[1].scatter(timeline["injection_day_after_PEmax"], y, s=90, color=[palette[x] for x in timeline["mouse"]], label="injection")
    endpoint = timeline["injection_day_after_PEmax"] + timeline["days_post_injection_at_sacrifice"]
    for yi, (_, row), end in zip(y, timeline.iterrows(), endpoint):
        color = palette[row["mouse"]]
        axes[1].plot([row["injection_day_after_PEmax"], end], [yi, yi], color=color, lw=4)
        axes[1].scatter([end], [yi], marker="X", s=100, color=color)
        axes[1].text(end + 1, yi, f"sacrifice: +{row['days_post_injection_at_sacrifice']} d", va="center", fontsize=9)
    axes[1].set_yticks(y, timeline["mouse"])
    axes[1].set_xlabel("Day after PEmax induction")
    axes[1].set_title("Mouse-specific preparation and endpoints\n(no replicated control-condition contrast)")
    axes[1].set_xlim(0, max(endpoint) + 15)
    fig.savefig(OUT / "figures/design_audit.png", dpi=180)
    plt.close(fig)

    git_commit = subprocess.run(
        ["git", "-C", "/home/yancui/haruka_v2", "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True,
    ).stdout.strip()
    tracked = sorted(BENCH.glob("*.py")) + [RUNNER, Path("/home/yancui/haruka_v2/bench_real_discovery/score_petracer_lineage_distance_cc3.py")]
    with (OUT / "hashes/code_sha256.txt").open("w") as handle:
        handle.write(f"haruka_v2_git_commit  {git_commit or 'unavailable'}\n")
        for path in tracked:
            handle.write(f"{sha256(path)}  {path}\n")
    with (OUT / "hashes/source_sha256.txt").open("w") as handle:
        for path in sorted(DATA.glob("*.h5td")):
            handle.write(f"{sha256(path)}  {path}\n")

    design = {
        "status": "blocked_no_real_control",
        "decision": "No Haruka training or responder analysis launched.",
        "reason": "The original real PEtracer source has no true control/condition arm, and all available proxies fail biological identifiability.",
        "source_data": source_rows,
        "verified_contract": {
            "modality": "targeted MERFISH RNA",
            "counts": "raw integer counts in layers/counts; X is scaled and not used for NB",
            "sample_key": "sample",
            "graph_boundary": "physical tissue section in obs['sample']",
            "batch_key_if_valid": "sample",
            "cell_type_key": "cell_subtype",
            "spatial_key": "obsm['spatial']",
            "condition_key": None,
            "control_value": None,
            "condition_value": None,
            "feature_policy_if_valid": "retain all targeted-panel features (n_vars <= 2000); cross-mouse run would first require an explicitly audited common panel",
            "identifiability": False,
        },
        "requested_model_configuration_not_run": {
            "model": "one-stage composition-gated Haruka",
            "z_dim": 32,
            "s_dim": 32,
            "hidden": 256,
            "encoder_layers": 1,
            "decoder_layers": 2,
            "epochs": 150,
            "niche_k": 20,
            "spatial_weight": 100,
            "composition_loss_weight": 1,
            "neighbor_expression_loss_weight": 1,
            "neighbor_expression_mode": "composition_gated",
            "zs_independence_penalty": 10,
            "likelihood": "NB",
            "decoder_batch_dim": 8,
            "batch_injection_mode": "decoder_only",
            "separate_decoder_projectors": ["assay", "composition", "niche_expression"],
            "seed": 0,
            "condition_only_cc": "symmetric 3-hop, section-bounded, own-cell block excluded",
        },
        "outputs_not_created_because_design_invalid": [
            "trained model", "z/s embeddings", "condition-only z-CC/s-CC", "responder scores",
            "cell programs", "domain enrichments", "leave-section-out metrics", "sample leakage metrics",
        ],
        "closest_valid_options": [
            {
                "option": "obtain matched control data",
                "requirements": "Untreated/vehicle or explicit perturbation control, replicated animals in each arm, compatible MERFISH panel, comparable endpoints, multiple sections per animal.",
                "supports_requested_claims": True,
            },
            {
                "option": "single-condition lineage/spatial descriptive analysis",
                "requirements": "Use original counts and lineage labels only after unsupervised embedding; bound graphs by section; make no response/responder or treatment-effect claims.",
                "supports_requested_claims": False,
            },
            {
                "option": "M2-vs-M3 28-vs-35-day exploratory pilot",
                "requirements": "Restrict to common genes, disclose n=1 mouse/time and complete animal-time confounding; descriptive sensitivity analysis only.",
                "supports_requested_claims": False,
            },
        ],
        "paper_context": {
            "citation": "Koblan et al., High-resolution spatial mapping of cell state and lineage dynamics in vivo with PEtracer, Science (2025)",
            "doi": "https://doi.org/10.1126/science.adx3800",
            "pmc": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12766569/",
            "code": "https://github.com/jweissmanlab/PEtracer-2025",
            "methods_facts": "M1 injected on day 12 after PEmax induction and sacrificed 39 days post-injection; M2/M3 injected day 6 and sacrificed 28/35 days post-injection; M1 recipient differs from M2/M3.",
        },
        "haruka_repository_git_commit": git_commit or None,
        "random_seed": 0,
    }
    with (OUT / "experiment_design.json").open("w") as handle:
        json.dump(design, handle, indent=2)

    (OUT / "logs/metadata_audit.log").write_text("\n".join(audit_lines) + "\n")


if __name__ == "__main__":
    main()
