# BE CODEX protein benchmark

This benchmark uses `EsophagusAtlas_CODEX.h5mu` and one section from each of
five patients: `E08_reg006`, `E11_reg002`, `E12_reg003`, `E17_reg006`, and
`E19_reg001`.

## Four-factor truth

| Representation | Ground truth |
|---|---|
| background intrinsic | observed composition label (`neigh_sub`) |
| background spatial | observed niche name (`neigh_name`) |
| salient intrinsic | simulated three-class IID protein program |
| salient spatial | simulated three-domain spatial protein program |

For each section, the condition sample is a paired copy of its control cells.
The condition receives disjoint salient-intrinsic and salient-spatial protein
programs. The two simulated labels are balanced within the natural background
labels, and the intrinsic label is additionally balanced within the simulated
spatial label. This prevents label correlation from creating artificial ARI.

The source CODEX matrix is feature-standardized rather than a count matrix.
Every reconstruction-based model therefore uses exact per-cell summed squared
error. PCA and CINEMA-OT do not have a decoder likelihood; they receive the
same clipped and shifted continuous-protein input as the trainable methods.

All four representations are evaluated by the same unsupervised protocol:
`StandardScaler -> PCA (at most 50 components) -> full-covariance GMM`.
Salient truths are evaluated on condition cells only.

## Outputs

- `data_pairs/manifest.csv`: paired simulation manifest
- `data_pairs/self_check.csv`: balance, independence, spatial-coherence checks
- `results/`: embeddings and per-section `be_four_factor_ari.json` files
- `summary/be_protein_four_factor_by_section.csv`: all 25 section-method rows
- `summary/be_protein_four_factor_mean_std.csv`: five-section mean and SD
- `summary/be_protein_four_factor_comparison.png`: four-panel comparison

The benchmark methods are Haruka-MENDER, protein-MSE ContrastiveVI plus
CellCharter, CINEMA-OT plus CellCharter, a protein-MSE autoencoder plus
CellCharter, and PCA plus CellCharter.
