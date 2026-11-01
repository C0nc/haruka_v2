# Haruka real-study discovery evaluation

## Contract

All runs use the latest large, one-stage, unsupervised four-slot Haruka: a 64-d
background latent (48-d private background factor reported here), 16-d salient,
16-d spatial, 16-d salient-spatial, hidden width 512, three layers, four heads,
k=8 and 180 epochs. The binary control/target arm is
the only biological label used to define contrastive training. Cell type,
subtype, exact stage, tissue-domain, marker programs, and AD plaque annotations
are evaluation-only. Specimen/section is condition-confounded in all three
studies, so it is a graph boundary, never a removable batch.

DKD and colitis use raw counts with a Poisson likelihood. Jill AD contains
continuous preprocessed expression rather than counts and therefore uses the
model's normal likelihood. The final Jill AD model jointly fits control-r1, disease-r1 and disease-r2
(28,040 cells); no section is held out from unsupervised representation learning.

## Cross-study result

| study | strongest supported discovery | limitation |
|---|---|---|
| DKD | Background recovers renal identity (0.812 cross-section BA); salient genes include MMP7, SPP1, VIM, LGALS3 and CLU; salient-spatial is best for the local injury program (rho 0.250). | Exact 30-60 versus <30 GFR is not reliably separated across held-out sections (best BA 0.451, chance 0.5); severity is strongly specimen/diagnosis-confounded. |
| DSS colitis | All four intended roles are approximately recovered. Background identity BA=0.761; spatial tissue-domain BA=0.315; salient intrinsic injury rho=0.206; salient-spatial local injury rho=0.192 and DSS-stage BA=0.589 (chance 0.333). | Intrinsic injury remains almost as readable from background (rho=0.233), so the intrinsic split is not fully identifiable. |
| Jill AD | Joint fitting improves background cell identity to BA=0.577 and makes salient intrinsic the best slot for the AD response program (rho=0.274). Shared spatial remains best for tissue domain (BA=0.692), while salient genes recover GFAP/CST7/CD68/APOE/TREM2 reactive glial biology. | Salient-spatial is still not the local-program-best slot (rho=0.004 versus 0.140 in shared spatial). One control section versus two disease sections cannot fully identify shared versus AD-specific spatial structure. |

## Biological discoveries

### DKD

- Background intrinsic: tubular and glomerular identity, led by ALDOB, ATP1B1,
  SLC22A8, GATM, HPD and PAH.
- Salient intrinsic: failed-repair/injury and matrix-remodeling program including
  MMP7, SPP1, VIM, LGALS3, CLU, SOD2 and MGP.
- Salient spatial: spatially localized MMP7/LGALS3/CLU/VIM/THBS1 remodeling.
- The pooled diseased arm is easy to distinguish, but exact GFR bin is not. This
  supports disease injury discovery, not a clean continuous DKD severity axis.

### DSS colitis

- Background intrinsic: stable stromal/smooth-muscle identity (SPARC, TAGLN,
  COL1A2, COL6A1, DPT, GREM1).
- Shared spatial: epithelial architecture (VIL1, SDC1, FZD5, LGR4, IL22RA1,
  CCL28) and local stress signaling.
- Salient intrinsic: acute response within a cell (EGR1, MMP9, NR4A1, IL1B,
  ACOD1, DUSP1, SOCS3, CXCL10).
- Salient spatial: inflammatory niche program (IL1B, CXCL10, CD14, TREM1,
  NOS2, CXCL2, MMP3/MMP9/MMP13). It is the best slot for held-out DSS day and
  local injury, making colitis the cleanest dataset for the intended design.

### Jill AD

- Background intrinsic is the best cell-identity slot across specimens
  (BA=0.577; assignment margin 0.202).
- Salient intrinsic is the best slot for the cross-specimen reactive AD program
  (rho=0.274; margin 0.065) and recovers GFAP, CST7, TREM2, CD68, APOE, CLEC7A,
  CD9 and P2RY12 biology.
- Shared spatial is the best tissue-domain slot (BA=0.692; margin 0.358) and
  carries most plaque geography. Disease-section cross-prediction reaches
  BA=0.709/AUC=0.799 for p-tau and BA=0.708/AUC=0.780 for any plaque.
- A-beta is most strongly readable from salient intrinsic across disease
  sections (BA=0.826, AUC=0.883), consistent with a non-local reactive response.
- Salient-spatial contains glial/plaque-neighborhood genes (CST7, C1QA, TREM2,
  GFAP, HEXB, CLEC7A, CD68 and GPNMB), but its local-program score is weak.
  The biology is present, but the shared-versus-salient spatial assignment is
  not identifiable from the available control design.

## Interpretation rules

Probe accuracy demonstrates linearly readable information, not discrete
clusters. ARIs remain modest in all three studies, so these factors are mostly
continuous programs. All final models, including AD, jointly fit all available
sections for decomposition. Grouped probes hold out specimens only at the
evaluation stage; they are not model-training holdouts.
