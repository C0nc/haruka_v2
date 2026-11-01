# Final reproducible Haruka analyses

All results use the final one-stage composition-gated Haruka checkpoints with decoder-only sample injection, separate batch projectors for all three decoder heads, z/s independence weight 10, and fixed seed 0. Spatial z-CC/s-CC uses condition cells only, is bounded by the recorded graph/sample key, averages exactly three graph hops, and excludes the cell's own latent block. See `experiment_design.csv` and each dataset's `experiment_design.json` for exact labels, counts, paths, and preprocessing.
