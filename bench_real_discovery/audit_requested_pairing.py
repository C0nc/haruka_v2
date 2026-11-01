#!/usr/bin/env python3
"""Print sample/condition structure for the six retained real-data studies."""

from pathlib import Path

import anndata as ad
import pandas as pd


DATASETS = [
    ("KRAS", Path("/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad"),
     ["condition_label", "slice_id"]),
    ("SPARC", Path("/home/yancui/haruka_v2/bench_spacseq/data/spac_lung_functional_guides_k8_hvg3000.h5ad"),
     ["condition", "slice_id", "sgRNA"]),
    ("STARmap AD", Path("/home/yancui/haruka_v2/bench_real_discovery/data/ad.h5ad"),
     ["condition", "slice_id", "graph_id"]),
    ("CODEX AD", Path("/home/yancui/haruka_v2/bench_codex_cns_2026/data/processed/codex_cns_myeloid_primary_no_aml_no_combat.h5ad"),
     ["condition", "donor_id"]),
    ("OSKM", Path("/home/yancui/haruka_v2/bench_oskm_reprogramming/data/oskm_old_matched_160k.h5ad"),
     ["condition", "slice_id", "paired_batch"]),
    ("Lung", Path("/home/yancui/haruka_v2/bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad"),
     ["condition", "slice_id", "sample", "donor", "sample_affect"]),
]


def main() -> None:
    for name, path, requested in DATASETS:
        data = ad.read_h5ad(path, backed="r")
        columns = [column for column in requested if column in data.obs]
        print(f"\n### {name}: n={data.n_obs:,}; columns={columns}")
        if columns:
            counts = data.obs[columns].astype(str).value_counts().sort_index()
            print(counts.to_string())
        data.file.close()


if __name__ == "__main__":
    main()
