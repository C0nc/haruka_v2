#!/usr/bin/env python3
"""Panel m: within each vascular type, share of the exercise-cell rejuvenation readout explained by
mouse (section), brain region (within mouse) and local cell-type proportions (held-out-mouse R^2, floored at 0);
observed clock vs held-out prediction from z and from s. Source: intra_vascular_partition.csv."""
import sys; import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/home/yancui/haruka_v2/figures_paper'); import paperstyle as ps
ps.setup()
T = pd.read_csv('intra_vascular_partition.csv'); T = T[T.cell_type != 'all vascular']
types = ['Endothelial', 'Pericyte', 'VLMC', 'VSMC']; reads = ['observed', 'z', 's']
col = {'observed': ps.LATENT['space'], 'z': ps.LATENT['z'], 's': ps.LATENT['s']}
fig, axes = plt.subplots(1, 3, figsize=(138 * ps.MM, 42 * ps.MM), sharey=False)
for ax, (key, title) in zip(axes, [('mouse_eta2_exercise', 'mouse / section'), ('region_eta2_exercise', 'brain region (within mouse)'),
                                   ('niche_r2_exercise', 'local cell-type proportions')]):
    for i, r in enumerate(reads):
        v = [max(T[(T.cell_type == t) & (T.readout == r)][key].iloc[0], 0) for t in types]
        ax.bar(np.arange(4) + (i - 1) * 0.26, v, 0.26, color=col[r], label='observed clock' if r == 'observed' else f'from {r}')
    ax.set_xticks(range(4)); ax.set_xticklabels(types, rotation=30, ha='right'); ax.set_title(title)
    ax.set_ylabel('variance explained')
axes[2].legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=6); ps.panel_label(axes[0], 'm', x=-0.35)
fig.suptitle('Exercise vascular cells: where does the within-type variation in clock age come from?', fontsize=6.5, y=1.04)
fig.tight_layout(); ps.save(fig, 'exb_m_intra_vascular_partition'); print(ps.check_fonts('exb_m_intra_vascular_partition.svg'))
