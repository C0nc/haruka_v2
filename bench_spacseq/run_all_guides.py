"""Stable entry point for multi-slice all-guide Haruka training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "bench_merfish", ROOT / "src", ROOT / "bench_spacseq"):
    sys.path.insert(0, str(path))

from run_multislice_all_guides import main  # noqa: E402


if __name__ == "__main__":
    main()
