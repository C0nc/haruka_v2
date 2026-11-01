# Paired-identifiable, density-stable Haruka

The corrected two-domain model addresses two distinct failures.

1. `run_haruka_paired_adaptive.py` preserves the standard protein MSE but adds:
   - matched background-latent invariance;
   - counterfactual condition-minus-control reconstruction through the salient
     branch;
   - lower Wasserstein (`0.05`) and niche reconstruction (`3.0`) weights.
2. `apply_graph_residual.py` replaces fixed-radius concatenation with a
   density-normalized multiscale kNN residual (`k=12,36,72`, `alpha=0.25`).

No niche or simulated-domain labels are used during fitting or graph
construction. Labels are read only by `score_2domain.py` after embeddings have
been saved.

The paired objective does use known cell correspondence between the copied
control and condition cells. It is therefore appropriate for this paired
simulation and for genuinely matched observations. It must not be presented as
an unpaired-data result. For unpaired biological samples, exact correspondence
must be replaced by an OT or mutual-nearest-neighbor soft matching objective.

Across the five BE sections, the final model obtains background-domain ARI
`0.2037 ± 0.0994` and salient-domain ARI `0.9944 ± 0.0039`.
