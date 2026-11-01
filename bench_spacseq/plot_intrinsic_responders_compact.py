"""Compact four-panel intrinsic responder visualization."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


parser = argparse.ArgumentParser()
parser.add_argument("--prefix", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
prefix = Path(args.prefix)
cells = pd.read_csv(prefix.with_suffix(".cells.csv"))
summary = json.loads(prefix.with_suffix(".summary.json").read_text())
functions = pd.DataFrame(summary["by_function"]).sort_values(
    "responder_fraction", ascending=False
)
guides = pd.DataFrame(summary["by_guide"])
order = functions.function.tolist()
guides["guide"] = guides.guide_identity.str.rsplit("_", n=1).str[-1]
target = cells[~cells.is_control]

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
sns.barplot(data=functions, x="function", y="responder_fraction", order=order,
            color="#4C78A8", ax=ax[0, 0])
ax[0, 0].axhline(summary["control_false_positive_fraction"], color="black",
                 ls="--", label="NT null")
ax[0, 0].set_title("Responder fraction by target function")
ax[0, 0].legend(frameon=False)

sns.barplot(data=guides, x="function", y="responder_fraction", hue="guide",
            order=order, palette="Set2", ax=ax[0, 1])
ax[0, 1].axhline(summary["control_false_positive_fraction"], color="black", ls="--")
ax[0, 1].set_title("Independent sgRNAs within each function")
ax[0, 1].legend(title="guide", frameon=False)

sns.violinplot(data=target, x="function", y="intrinsic_response_percentile",
               order=order, inner="quart", cut=0, color="#72B7B2", ax=ax[1, 0])
ax[1, 0].axhline(0.95, color="black", ls="--")
ax[1, 0].set_title("Response percentile against state-matched NT")

sns.scatterplot(data=target, x="ecm_minus_epithelial_score",
                y="intrinsic_response_percentile", hue="responder",
                palette={False: "#B7B7B7", True: "#D62728"}, alpha=.65,
                s=22, linewidth=0, ax=ax[1, 1])
rho = summary["response_state_spearman"]["rho"]
ax[1, 1].axhline(.95, color="black", ls="--")
ax[1, 1].set_title(f"Response vs ECM–epithelial state (ρ={rho:.3f})")
ax[1, 1].legend(title="responder", frameon=False)
for axis in (ax[0, 0], ax[0, 1], ax[1, 0]):
    axis.tick_params(axis="x", rotation=35)
output = Path(args.out)
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=220, bbox_inches="tight")
plt.close(fig)
print(f"[save] {output}")
