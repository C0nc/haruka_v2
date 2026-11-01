"""Run guide-supervised multi-slice Haruka after the unsupervised audit."""

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
from perturbation_supervision import install_perturbation_supervision  # noqa: E402


def argument_value(flag):
    if flag not in sys.argv:
        raise ValueError(f"required argument {flag} is missing")
    return sys.argv[sys.argv.index(flag) + 1]


data_path = argument_value("--data")
adata = ad.read_h5ad(data_path)
classes = sorted(set(adata.obs.perturbation.astype(str)) - {"NT"})
class_index = {label: index for index, label in enumerate(classes)}
labels_by_row = np.asarray([
    class_index.get(label, -1) for label in adata.obs.perturbation.astype(str)
], dtype=np.int64)


def install_combined(module, batch_weight):
    install_batch(module, batch_weight)
    install_perturbation_supervision(
        module, labels_by_row, len(classes), weight=10.0
    )
    return module


runner.install_four_slot_batch_mmd = install_combined


if __name__ == "__main__":
    runner.main()
