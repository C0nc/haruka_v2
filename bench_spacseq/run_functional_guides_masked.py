"""Strict functional transfer: supervise one guide and mask its partner."""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "bench_merfish", ROOT / "src", ROOT / "bench_spacseq"):
    sys.path.insert(0, str(path))

import run_multislice_all_guides as runner  # noqa: E402
from batch_invariance import install_four_slot_batch_mmd as install_batch  # noqa: E402
from functional_supervision_compact import install_functional_supervision  # noqa: E402


def argument_value(flag):
    if flag not in sys.argv:
        raise ValueError(f"required argument {flag} is missing")
    return sys.argv[sys.argv.index(flag) + 1]


adata = ad.read_h5ad(argument_value("--data"))
design = adata.uns["functional_guide_design"]
functions = sorted(design["targets"])
function_index = {name: index for index, name in enumerate(functions)}
train_guide = {
    function: list(design["guides_by_target"][function])[0]
    for function in functions
}
all_guides = sorted(name for name in set(adata.obs.guide_identity.astype(str)) if name)
guide_index = {name: index for index, name in enumerate(all_guides)}
anchor = adata.obs.is_anchor.to_numpy(bool)
function_obs = adata.obs.function.astype(str).to_numpy()
guide_obs = adata.obs.guide_identity.astype(str).to_numpy()
labelled = np.asarray([
    use and function in train_guide and guide == train_guide[function]
    for use, function, guide in zip(anchor, function_obs, guide_obs)
])
function_by_row = np.asarray([
    function_index.get(function, -1) if use else -1
    for function, use in zip(function_obs, labelled)
], dtype=np.int64)
guide_by_row = np.asarray([
    guide_index.get(guide, -1) if use else -1
    for guide, use in zip(guide_obs, labelled)
], dtype=np.int64)


def install_combined(module, batch_weight):
    install_batch(module, batch_weight)
    install_functional_supervision(
        module, function_by_row, guide_by_row, len(functions),
        function_weight=5.0, guide_invariance_weight=0.0,
        compactness_weight=5.0,
    )
    module.masked_function_train_guides = train_guide
    return module


runner.install_four_slot_batch_mmd = install_combined


if __name__ == "__main__":
    print(f"[masked function supervision] {train_guide}", flush=True)
    runner.main()
