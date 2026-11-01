# Haruka-MENDER on real SPARC-seq and Jill AD

## Training contract

- All samples were jointly trained; no holdout and no post-fit regression.
- Training used only binary control/condition membership.
- SPARC sgRNA identity, cell type, and niche were evaluation-only.
- AD cell type, tissue domain, p-tau, and amyloid-beta were evaluation-only.
- Haruka-MENDER uses ContrastiveVI intrinsic encoders, dual niche-composition
  decoders, and independent MENDER ring aggregation; it has no SIMVI or GAT.

## SPARC-seq

The background intrinsic cell-type LOSO score is
0.299;
background spatial niche LOSO is
0.637.

The salient-spatial representation separates sgRNA contexts within a slice
(0.492
BA; chance 0.071), but does not transfer to an unseen slice
(0.094).
Salient intrinsic is also near chance across slices
(0.077).

Exploratory cross-slice direction candidates are intrinsic
sgArf6_1, sgAbcc3_1, sgPhgr1_2 and spatial sgUtrn_1, sgGdf15_2, sgLrig3_2. However,
none of the 14 named targets shows significant raw-RNA knockdown after BH
correction (minimum q=0.649). These are therefore
response-associated hypotheses, not validated on-target discoveries.

## Jill AD

The decomposition assigns cell identity to background (0.707),
tissue domain to background spatial (0.632),
and the intrinsic AD program to salient intrinsic
(0.296).

The strongest reproducible disease discovery is amyloid-beta adjacency in
salient intrinsic: BA=0.833, AUROC=0.900
under reciprocal disease-replicate transfer. p-tau is strongest in background
spatial: BA=0.711, AUROC=0.781.

The new salient-spatial MENDER block is not a successful plaque representation:
p-tau BA=0.521, versus
0.722 for the previous GAT-based clean model.
Thus the supported new biological conclusion is an intrinsic A-beta-associated
AD state plus a shared spatial p-tau tissue context, not a distinct AD-specific
spatial plaque field.
