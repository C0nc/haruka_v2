# Claims and limitations

## Primary claim

The full-cohort model contains spatially organized OSKM-response heterogeneity, but the evidence supports a pair-sensitive, heterogeneous response rather than uniform rejuvenation.  Salient s-CC improves extreme-responder classification beyond z-CC, whereas z-CC is stronger for strict pair-adjusted continuous response.  Residual sample structure sharply limits domain interpretation.

## Safe biological claims

- Three pairs are youngward and B14 is oldward under the matched-control primary definition.
- Primary q95 responders comprise 10.34% of OSKM cells; q90–q99 and scaled/unsmoothed/leave-pair-out sensitivities preserve the broad heterogeneity but alter prevalence.
- Domain 5 is responder-rich after pair×cell-type×subregion standardization and passes the sample-dominance flag; domain 0 is responder-poor and also passes.
- The responder-rich contrast is associated with neuronal/developmental genes and reduced astrocytic/reactive genes, but these are descriptive panel-internal associations.
- The frozen clock has modest adjusted concordance with Haruka response but weak cell-level responder overlap and no FDR-supported replicated rejuvenation by cell type.

## Required caveats

- Four matched pairs provide little replicate-level power; cell-level P values do not create more biological replicates.
- Condition and slice are perfectly confounded. Decoder-only slice embeddings do not prove batch removal.
- s-CC is sample-dominated (cluster NMI 0.397; UMAP NMI 0.350); six of eight domains are flagged, and domain 7 is entirely B10.
- All continuous pair-adjusted held-out R² values are non-positive for the expression-derived score, despite positive rank correlations.
- No cell type or subregion is primary-youngward in every pair; B14 is negative across all reported groups.
- Response, genes, and programs come from the same 300-gene targeted panel and are not statistically independent.
- Cell-type composition is a Haruka reconstruction target; domain composition and cell-cell enrichments are interpretations, not independent validation.
- The clock cache covers a deterministic 160k cohort subset, not all 684k cells. Serialized models warn about sklearn 1.2.2→1.7.2 loading; current arithmetic was verified exactly, but cross-version support is not guaranteed.
- UMAP and static spatial organization are descriptive. No measured temporal lineage or causal neighborhood effect is claimed.
