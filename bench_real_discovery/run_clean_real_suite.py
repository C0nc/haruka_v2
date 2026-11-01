"""Run the benchmark-selected clean Haruka configuration on real studies.

This runner deliberately imports the benchmark variant instead of copying its
loss settings.  Dataset-specific overrides are limited to the likelihood and a
batch covariate when the prepared-data audit establishes that it is removable.
All cells are co-trained; no holdout, staged fit, supervision within the target
arm, or post-fit residualization is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "bench_merfish", ROOT / "src"):
    sys.path.insert(0, str(path))

from common import Timer, save_embeddings  # noqa: E402
from run_haruka import SHARED, VARIANTS, extract_all_cells, validate_batch_condition  # noqa: E402


STUDIES = {
    "spacseq": {
        "data": ROOT / "bench_spacseq/data/spac_lung_all_slices_all_sgrna_all_nt_k8_hvg3000.h5ad",
        "likelihood": "poisson",
        "batch_key": "slice_id",
        "library_key": "graph_id",
    },
    "aging": {
        "data": ROOT / "bench_aging/data/merfish_aging_100k.h5ad",
        "likelihood": "poisson",
        "batch_key": "technical_batch",
        "library_key": "graph_id",
    },
    "dkd": {
        "data": ROOT / "bench_real_discovery/data/dkd.h5ad",
        "likelihood": "poisson",
        "batch_key": None,
        "library_key": "graph_id",
    },
    "colitis": {
        "data": ROOT / "bench_real_discovery/data/colitis.h5ad",
        "likelihood": "poisson",
        "batch_key": None,
        "library_key": "graph_id",
    },
    "ad": {
        "data": ROOT / "bench_real_discovery/data/ad.h5ad",
        "likelihood": "normal",
        "batch_key": None,
        "library_key": "graph_id",
    },
}


def run_study(study: str, args: argparse.Namespace) -> None:
    from haruka_spatial import HarukaSpatial

    spec = STUDIES[study]
    data_path = Path(spec["data"])
    data = ad.read_h5ad(data_path)
    required = {"condition", "slice_id", "graph_id"}
    missing = sorted(required - set(data.obs))
    if missing:
        raise KeyError(f"{study}: missing columns {missing}")
    if set(data.obs.condition.astype(str)) != {"control", "disease"}:
        raise ValueError(f"{study}: condition must contain control and disease")
    if "counts" not in data.layers:
        data.layers["counts"] = data.X.copy()

    kwargs = {
        **SHARED,
        **VARIANTS["both_clean_cvi_symmetric"],
        "gene_likelihood": spec["likelihood"],
        "batch_key": spec["batch_key"],
    }
    validate_batch_condition(data.obs, kwargs["batch_key"])

    out = (
        Path(args.out_root) / study /
        f"haruka_both_clean_cvi_symmetric_no_postfit_seed{args.seed}"
    )
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n===== clean real-data Haruka: {study} ({data.n_obs} cells) =====", flush=True)
    with Timer() as timer:
        model = HarukaSpatial(
            data,
            cov_gene_num=0,
            seed=args.seed,
            k_spatial=args.k_spatial,
            library_key=spec["library_key"],
        )
        model.prepare_spatial_graph()
        model.setup_model(seed=args.seed, **kwargs)
        model.train(
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            plan_kwargs={"lr": args.lr},
        )
        reps = extract_all_cells(
            model,
            n_layer=args.n_layer,
            orthogonalize_paired_slots=False,
        )

    save_embeddings(
        str(out),
        background=reps["background"],
        spatial=reps["spatial"],
        salient=reps["salient"],
        salient_spatial=reps["salient_spatial"],
        obs_names=data.obs_names.to_numpy(),
        method="haruka_both_clean_cvi_symmetric_real_data",
        study=study,
        data=str(data_path),
        runtime_s=timer.seconds,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        k_spatial=args.k_spatial,
        n_layer=args.n_layer,
        graph_library_key=spec["library_key"],
        batch_key=spec["batch_key"],
        binary_arm_membership_only=True,
        within_target_label_supervision=False,
        perturbation_identity_supervision=False,
        stage_supervision=False,
        plaque_supervision=False,
        training_mode="one_stage",
        all_samples_joint=True,
        holdout_graph=None,
        control_only_pretraining=False,
        postfit_residualization=False,
        paired_slot_orthogonalization=False,
        benchmark_variant="both_clean_cvi_symmetric",
        benchmark_variant_kwargs=VARIANTS["both_clean_cvi_symmetric"],
        resolved_kwargs=kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", nargs="+", choices=sorted(STUDIES), default=sorted(STUDIES))
    parser.add_argument(
        "--out-root",
        default="bench_real_discovery/results_clean_cvi_symmetric_no_postfit",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k-spatial", type=int, default=8)
    parser.add_argument("--n-layer", type=int, default=0)
    args = parser.parse_args()

    print(json.dumps({"studies": args.studies, "contract": {
        "variant": "both_clean_cvi_symmetric",
        "all_samples_joint": True,
        "holdout": None,
        "one_stage": True,
        "postfit_residualization": False,
    }}, indent=2), flush=True)
    for study in args.studies:
        run_study(study, args)


if __name__ == "__main__":
    main()
