#!/usr/bin/env python3
"""Write the auditable design manifest for the six final real-data analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import pandas as pd


ROOT = Path("/home/yancui/haruka_v2")
RESULTS = ROOT / "results_real_compgated_nb_final"
MODEL = "haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0"
CC = "haruka_z_s_cellcharter_cc3_condition_only"

DATASETS = {
    "kras": {
        "title": "KRAS-inhibitor mouse-lung Xenium",
        "data": Path("/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad"),
        "tag": "mouselung_kras_xenium_clean_noC3",
        "condition_key": "condition_label", "control": "Control", "condition": "KRASi_treated",
        "replicate_key": "slice_id", "graph_key": "slice_id", "cell_type_key": "celltype",
        "pairing": "No one-to-one pairing assumed; treated sections are analyzed replicate-aware.",
        "primary_question": "Treatment-response heterogeneity in treated tumor and macrophage niches.",
    },
    "starmap_ad": {
        "title": "STARmap Alzheimer disease",
        "data": ROOT / "bench_real_discovery/data/ad.h5ad", "tag": "ad",
        "results_root": ROOT / "results_real_compgated_lambda10_samplebatch_sepheads",
        "condition_key": "condition", "control": "control", "condition": "disease",
        "replicate_key": "slice_id", "graph_key": "slice_id", "cell_type_key": "cell_type",
        "pairing": "No longitudinal pairing assumed; disease slices are evaluated separately.",
        "primary_question": "Plaque/tau-associated disease niches and glial trajectories.",
    },
    "codex_ad": {
        "title": "CODEX Alzheimer disease",
        "data": ROOT / "bench_codex_cns_2026/data/processed/codex_cns_myeloid_primary_no_aml_no_combat.h5ad",
        "tag": "codex_cns_myeloid_primary_no_aml_no_combat",
        "results_root": ROOT / "results_real_compgated_lambda10_samplebatch_sepheads",
        "condition_key": "condition", "control": "control", "condition": "disease",
        "replicate_key": "donor_id", "graph_key": "donor_id", "cell_type_key": "paper_initial_leiden",
        "pairing": "No longitudinal pairing assumed; plaque readout is donor-held-out.",
        "primary_question": "Protein-defined plaque proximity and donor-robust disease niches.",
    },
    "sparc_seq": {
        "title": "SPARC-seq lung perturbation",
        "data": ROOT / "bench_spacseq/data/spac_lung_functional_guides_k8_hvg3000.h5ad",
        "tag": "spac_lung_functional_guides_k8_hvg3000",
        "condition_key": "condition", "control": "control", "condition": "disease",
        "replicate_key": "slice_id", "graph_key": "slice_id", "cell_type_key": "cell_type",
        "pairing": "NT controls and targeting guides are compared within each slice; no cell pairing.",
        "primary_question": "sgRNA functional class and state-matched responder heterogeneity.",
    },
    "oskm": {
        "title": "Old-control versus old-OSKM MERFISH (full cohort)",
        "data": ROOT / "bench_oskm_reprogramming/data/oskm_old_full_684253.h5ad",
        "tag": "oskm_old_full_684253",
        "results_root": ROOT / "results_real_compgated_nb_full_oskm",
        "condition_key": "condition", "control": "control", "condition": "oskm",
        "replicate_key": "slice_id", "graph_key": "slice_id", "cell_type_key": "cell_type",
        "pairing": "Old control and old OSKM mice are matched by paired_batch (B9/B10/B13/B14).",
        "primary_question": "OSKM-associated rejuvenation relative to a young reference.",
    },
    "lung_fibrosis": {
        "title": "GSE250346 human lung fibrosis",
        "data": ROOT / "bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad",
        "tag": "gse250346_unaffected_vs_all_fibrosis",
        "condition_key": "condition", "control": "control", "condition": "fibrosis",
        "replicate_key": "sample", "graph_key": "slice_id", "cell_type_key": "cell_type",
        "pairing": "Training contrasts unaffected versus fibrosis; Less/More affected regions are paired within donor when available.",
        "primary_question": "Fibrosis severity, within-donor niche shifts, and initial-state/fate coupling.",
    },
}


def stable_json_hash(path: Path) -> str:
    payload = json.dumps(json.loads(path.read_text()), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def serial_counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.astype(str).value_counts(dropna=False).sort_index().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "final_results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, cfg0 in DATASETS.items():
        cfg = dict(cfg0)
        results_root = cfg.get("results_root", RESULTS)
        model_dir = results_root / cfg["tag"] / MODEL
        cc_dir = results_root / cfg["tag"] / CC
        meta_path = cc_dir / "meta.json"
        meta = json.loads(meta_path.read_text())
        data = ad.read_h5ad(cfg["data"], backed="r")
        obs = data.obs
        condition_counts = serial_counts(obs[cfg["condition_key"]])
        replicate_counts = serial_counts(obs[cfg["replicate_key"]])
        cell_type_counts = serial_counts(obs[cfg["cell_type_key"]])
        record = {
            "dataset": name, "title": cfg["title"], "data_path": str(cfg["data"]),
            "data_n_obs": int(data.n_obs), "data_n_vars": int(data.n_vars),
            "condition_key": cfg["condition_key"], "control_label": cfg["control"],
            "condition_label": cfg["condition"], "condition_counts": condition_counts,
            "replicate_key": cfg["replicate_key"], "replicate_counts": replicate_counts,
            "graph_key": cfg["graph_key"], "decoder_batch_key": meta["batch_key"],
            "cell_type_key": cfg["cell_type_key"], "n_cell_types": len(cell_type_counts),
            "cell_type_counts": cell_type_counts, "pairing": cfg["pairing"],
            "primary_question": cfg["primary_question"], "model_dir": str(model_dir),
            "cellcharter_dir": str(cc_dir), "training_seed": int(meta["seed"]),
            "analysis_seed": args.seed, "model_meta_sha256": stable_json_hash(meta_path),
            "model": {
                "one_stage": meta["one_stage"], "n_latent_z": meta["n_latent"],
                "n_latent_s": meta["n_latent"], "n_hidden": meta["n_hidden"],
                "encoder_layers": meta["n_layers"], "decoder_layers": meta["decoder_layers"],
                "epochs": meta["max_epochs"], "batch_size": meta["batch_size"],
                "k_niche": meta["k_niche"], "neighbor_embedding_dim": meta["neighbor_embedding_dim"],
                "spatial_weight": meta["spatial_weight"],
                "zs_independence_penalty": meta["zs_independence_penalty"],
                "neighbor_expression_mode": meta["neighbor_expression_mode"],
                "batch_injection_mode": meta["batch_injection_mode"],
                "decoder_batch_embedding_dim": meta["decoder_batch_embedding_dim"],
                "separate_decoder_batch_projectors": meta["separate_decoder_batch_projectors"],
                "assay_type": meta["assay_type"], "assay_reconstruction": meta["assay_reconstruction"],
                "n_top_genes": meta["n_top_genes"], "selected_features": meta["selected_assay_feature_count"],
                "composition_loss_weight": meta["composition_loss_weight"],
                "neighbor_mean_loss_weight": meta["neighbor_mean_loss_weight"],
            },
            "cellcharter": {
                "condition_only": True, "sample_key": meta["sample_key"],
                "n_hops": meta["n_hops"], "includes_self_block": False,
                "representation": meta["spatial_readout"],
            },
            "determinism": {
                "PYTHONHASHSEED": args.seed, "UMAP_random_state": args.seed,
                "KMeans_random_state": args.seed, "PCA_random_state": args.seed,
                "single_thread_BLAS": True,
            },
        }
        out_dir = args.out / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "experiment_design.json").write_text(json.dumps(record, indent=2) + "\n")
        rows.append({
            "dataset": name, "title": cfg["title"], "n_cells": data.n_obs, "n_features": data.n_vars,
            "condition_key": cfg["condition_key"], "control_label": cfg["control"],
            "condition_label": cfg["condition"], "control_n": condition_counts.get(cfg["control"], 0),
            "condition_n": condition_counts.get(cfg["condition"], 0),
            "replicate_key": cfg["replicate_key"], "n_replicates": len(replicate_counts),
            "graph_key": cfg["graph_key"], "decoder_batch_key": meta["batch_key"],
            "cell_type_key": cfg["cell_type_key"], "n_cell_types": len(cell_type_counts),
            "assay": meta["assay_type"], "likelihood": meta["assay_reconstruction"],
            "hvg_n": meta["n_top_genes"], "seed": args.seed, "pairing": cfg["pairing"],
        })
    pd.DataFrame(rows).to_csv(args.out / "experiment_design.csv", index=False)
    (args.out / "README.md").write_text(
        "# Final reproducible Haruka analyses\n\n"
        "All results use the final one-stage composition-gated Haruka checkpoints with "
        "decoder-only sample injection, separate batch projectors for all three decoder heads, "
        "z/s independence weight 10, and fixed seed 0. Spatial z-CC/s-CC uses condition cells "
        "only, is bounded by the recorded graph/sample key, averages exactly three graph hops, "
        "and excludes the cell's own latent block. See `experiment_design.csv` and each dataset's "
        "`experiment_design.json` for exact labels, counts, paths, and preprocessing.\n"
    )


if __name__ == "__main__":
    main()
