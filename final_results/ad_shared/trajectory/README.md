# Haruka s-CC DPT + CellRank 2 AD analysis

Disease cells were analyzed separately for every replicate/donor and broad biological cluster.
The KNN graph uses only `salient_spatial.npy` (Haruka salient CellCharter, s-CC).
DPT is rooted in the s-CC medoid of the plaque-distant decile. CellRank's PseudotimeKernel
then directs transitions along increasing DPT. GPCCA terminal states are inferred without
plaque labels; the terminal state with the smallest median plaque distance is annotated post hoc
as the plaque-associated fate.

These are cross-sectional state-transition potentials, not lineage-traced temporal fates.

Successful replicate × cluster analyses: 66
Skipped/failed groups: 4

## Files

- `cellrank_cluster_summary.csv`: per-analysis effect sizes, AUCs, and tests.
- `cellrank_cell_scores.csv.gz`: per-cell DPT, plaque-fate probability, and transition score.
- `cellrank_dpt_quintile_transition_probabilities.csv`: coarse directed transition matrices.
- `*/transition_matrices/*.npz`: full sparse CellRank transition matrices.
- PNG figures: cluster heatmaps, cross-dataset boxplots, spatial maps, and diffusion examples.
