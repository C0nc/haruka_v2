# Claims and limitations

## Supported claims

1. The three original H5TD files are tumor-bearing targeted MERFISH RNA datasets with raw integer counts, spatial coordinates, physical-section labels, cell-type annotations, and lineage metadata.
2. None contains an explicit experimental control/condition column or an untreated/sham arm.
3. Mouse/section labels cannot be promoted to a defensible treatment contrast: experimental timing, animal identity, endpoint, recipient background, section structure, and/or feature panel are confounded.
4. The existing paired PEtracer benchmark is explicitly simulated and must not support real responder-biology claims.
5. Consequently, state-matched control-null responders, response programs, and condition-associated s/s-CC biology are not identifiable from the available real data.

## Not claimed

- No model was trained.
- No z, s, z-CC, or s-CC was constructed.
- No cells were labeled responders or non-responders.
- No lineage family/clade association, barcode-distance preservation, cell program, cell-type/neighborhood enrichment, leave-section-out result, or sample-leakage result is reported for a real condition effect.
- Absence of a valid contrast is not evidence of absence of biological heterogeneity.

## Design limitation

Multiple sections from one animal are technical/spatial subsamples, not independent animal replicates. Decoder-only sample conditioning cannot repair condition-animal confounding or create a missing counterfactual. An M2-versus-M3 fit would be able to encode observed differences, but it could not distinguish time from animal or panel effects.

## Required data to unblock

Provide an explicit binary perturbation with replicated animals per arm, matched tissue and endpoints, compatible features, a physical-section key, and cell-state overlap sufficient for control-anchor matching. Then train all selected real samples jointly with section-bounded graphs and evaluate generalization by holding out animals/sections.
