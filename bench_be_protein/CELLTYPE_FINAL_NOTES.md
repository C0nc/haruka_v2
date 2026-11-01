# Final BE two-domain benchmark with correct cell annotation

- Background truth: `neigh_name` (six most frequent niches per section).
- Cell-composition annotation: fine `Cell Type` (30–36 types per section).
- `neigh_sub` is retained only as source metadata and is not treated as cell type.
- Salient truth: three simulated condition-specific spatial domains.

The salient domain is balanced within `neigh_name × Cell Type`; its NMI with
fine cell type is at most 0.0036. The paired-difference oracle remains ARI 1.0.

The final background representation contains 24 PCA dimensions from the
multiscale background protein latent and eight PCA dimensions from observed
multiscale fine-cell-type composition. The 24:8 split was selected as one
global configuration across all five sections. `neigh_name` is never used in
training or representation construction.

The salient graph residual is accepted only when the unsupervised clustering
is stable relative to the cell latent (AMI at least 0.8). This selected alpha
0.25 for E08, E11, E12, and E19, and alpha 0 for E17. No truth label is used by
this gate.

Five-section results are background ARI `0.2770 ± 0.1295`, salient ARI
`0.9873 ± 0.0053`, and mean ARI `0.6321 ± 0.0627`.
