# Default Haruka method for the mouse-brain decomposition benchmark

The canonical method in this benchmark is **Haruka-MENDER**:

1. fit the unsupervised ContrastiveVI background and salient encoders;
2. learn the dual niche decoder targets;
3. keep only the background and salient composition blocks;
4. aggregate each block independently in six non-overlapping 15-micron rings;
5. apply MENDER-style normalize-total and `log1p` transformation;
6. evaluate each spatial slot with the shared PCA + full-covariance GMM protocol.

It does not use SIMVI, GAT, CellCharter, decoder activation blocks, labels during
training, post-fitting, or regression of the other slots.

Run it with:

```bash
/home/yancui/.conda/envs/haruka2/bin/python \
  bench_mouse_brain_scviva/run_haruka.py \
  --data bench_mouse_brain_scviva/data_pairs/mouse_brain_A8_paired.h5ad
```

The output is written to
`results_haruka/<dataset>/haruka_mender_seed<seed>/`. The legacy decoder and
CellCharter scripts remain available as explicit ablations and do not define the
default method.
