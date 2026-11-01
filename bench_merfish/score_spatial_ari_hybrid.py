"""Hybrid-simulation entry point for the spatial-only PCA+GMM scorer."""

from __future__ import annotations

import sys

from score_spatial_ari import main


if __name__ == "__main__":
    if "--background_truth" not in sys.argv:
        sys.argv.extend(["--background_truth", "gt_niche"])
    main()
