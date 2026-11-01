"""Direct visual audit of SPARC-seq responder/non-responder analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from compare_response_heterogeneity import expression_pcs, external_response_percentile


def umap_xy(x: np.ndarray, seed: int) -> np.ndarray:
    return UMAP(
        n_neighbors=25, min_dist=0.25, metric="cosine",
        random_state=seed, n_jobs=1,
    ).fit_transform(StandardScaler().fit_transform(x))


def response_scatter(ax, xy, response, title):
    point = ax.scatter(
        xy[:, 0], xy[:, 1], c=response, cmap="viridis", vmin=0, vmax=1,
        s=8, alpha=0.82, linewidths=0, rasterized=True,
    )
    ax.set(title=title, xticks=[], yticks=[], xlabel="UMAP 1", ylabel="UMAP 2")
    plt.colorbar(point, ax=ax, label="Expression-response percentile", fraction=0.046)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--emb", required=True, type=Path)
    parser.add_argument("--prediction-prefix", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--k-nt", type=int, default=25)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(args.data)
    names = np.load(args.emb / "obs_names.npy", allow_pickle=True).astype(str)
    a = a[names].copy()
    anchor = a.obs.is_anchor.to_numpy(bool)
    aa = a[anchor].copy()
    obs = aa.obs
    function = obs.function.astype(str).to_numpy()
    guide = obs.guide_identity.astype(str).to_numpy()
    slices = obs.slice_id.astype(str).to_numpy()
    control = function == "NT"
    target = ~control
    coordinates = obs[["array_row", "array_col"]].to_numpy(float)

    pcs = expression_pcs(aa, args.n_pcs, args.seed)
    distance, percentile = external_response_percentile(
        pcs, coordinates, control, slices, args.k_nt
    )
    responder = target & (percentile >= args.threshold)
    scores = pd.DataFrame({
        "obs_name": aa.obs_names.astype(str), "slice_id": slices,
        "function": function, "guide_identity": guide,
        "is_control": control, "expression_response_distance": distance,
        "response_percentile": percentile, "responder": responder,
    })
    scores.to_csv(args.out_dir / "response_scores_all_anchors.csv", index=False)

    zcc = np.load(args.emb / "spatial.npy")[anchor][target]
    scc = np.load(args.emb / "salient_spatial.npy")[anchor][target]
    zxy, sxy = umap_xy(zcc, args.seed), umap_xy(scc, args.seed + 1)
    target_response = percentile[target]
    target_responder = responder[target]

    sns.set_theme(style="white")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), constrained_layout=True)
    response_scatter(axes[0, 0], zxy, target_response, "Background z-CC")
    response_scatter(axes[0, 1], sxy, target_response, "Salient s-CC")
    axes[0, 2].scatter(sxy[~target_responder, 0], sxy[~target_responder, 1],
                       s=6, c="#cfcfcf", alpha=.45, linewidths=0, rasterized=True)
    axes[0, 2].scatter(sxy[target_responder, 0], sxy[target_responder, 1],
                       s=12, c="#d62728", alpha=.9, linewidths=0, rasterized=True,
                       label=f"Responder (n={target_responder.sum():,})")
    axes[0, 2].set(title="Salient s-CC — responder status", xticks=[], yticks=[],
                   xlabel="UMAP 1", ylabel="UMAP 2")
    axes[0, 2].legend(frameon=False)

    for ax, sample in zip(axes[1], sorted(np.unique(slices))):
        use = slices == sample
        ctl = use & control
        non = use & target & ~responder
        rsp = use & responder
        ax.scatter(coordinates[ctl, 0], coordinates[ctl, 1], s=5, c="#bdbdbd",
                   alpha=.35, linewidths=0, rasterized=True, label="NT control")
        ax.scatter(coordinates[non, 0], coordinates[non, 1], s=7, c="#4c78a8",
                   alpha=.5, linewidths=0, rasterized=True, label="Target non-responder")
        ax.scatter(coordinates[rsp, 0], coordinates[rsp, 1], s=13, c="#e45756",
                   alpha=.9, linewidths=0, rasterized=True, label="Target responder")
        ax.set(title=f"{sample}: physical anchor locations", xticks=[], yticks=[],
               xlabel="array row", ylabel="array column", aspect="equal")
    axes[1, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"SPARC-seq response: local-NT expression percentile ≥ {args.threshold:.2f}",
        fontsize=15,
    )
    fig.savefig(args.out_dir / "response_umap_and_spatial_maps.png", dpi=240,
                bbox_inches="tight")
    plt.close(fig)

    guide_prediction = pd.read_csv(
        Path(f"{args.prediction_prefix}.leave_one_guide_out.predictions.csv")
    )
    slice_prediction = pd.read_csv(
        Path(f"{args.prediction_prefix}.leave_one_slice_out.predictions.csv")
    )
    summary = (
        scores[~scores.is_control]
        .groupby(["function", "guide_identity"], as_index=False)
        .agg(n=("responder", "size"), responder_fraction=("responder", "mean"),
             median_response=("response_percentile", "median"))
    )
    summary.to_csv(args.out_dir / "responder_fraction_by_gene_guide.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), constrained_layout=True)
    plot_frame = scores.copy()
    plot_frame["arm"] = np.where(plot_frame.is_control, "NT control", "Targeting")
    sns.violinplot(data=plot_frame, x="arm", y="response_percentile", inner="quart",
                   cut=0, palette=["#9e9e9e", "#4c78a8"], ax=axes[0, 0])
    axes[0, 0].axhline(args.threshold, color="#d62728", ls="--")
    axes[0, 0].set(title="Independent expression-response score", xlabel="",
                   ylabel="Response percentile")

    sns.pointplot(data=summary, x="function", y="responder_fraction",
                  hue="guide_identity", dodge=.35, errorbar=None, markers="o",
                  linestyles="none", ax=axes[0, 1])
    axes[0, 1].axhline((percentile[control] >= args.threshold).mean(),
                       color="#777777", ls="--", label="NT false-positive rate")
    axes[0, 1].set(title="Responder rate by target and independent guide",
                   xlabel="Target gene", ylabel="Responder fraction")
    axes[0, 1].tick_params(axis="x", rotation=35)
    axes[0, 1].legend(frameon=False, fontsize=6, ncol=2)

    for rep, color in (("background", "#4c78a8"), ("salient", "#e45756")):
        y = guide_prediction.responder.to_numpy(bool)
        probability = guide_prediction[f"{rep}_probability"].to_numpy(float)
        fpr, tpr, _ = roc_curve(y, probability)
        auc = roc_auc_score(y, probability)
        axes[0, 2].plot(fpr, tpr, color=color, lw=2, label=f"{rep} AUC={auc:.3f}")
    axes[0, 2].plot([0, 1], [0, 1], "--", color="#777777")
    axes[0, 2].set(title="Leave-one-guide-out responder prediction",
                   xlabel="False-positive rate", ylabel="True-positive rate")
    axes[0, 2].legend(frameon=False)

    for ax, rep, color in zip(axes[1, :2], ("background", "salient"),
                              ("#4c78a8", "#e45756")):
        observed = guide_prediction.response_percentile.to_numpy(float)
        predicted = guide_prediction[f"{rep}_continuous"].to_numpy(float)
        rho = spearmanr(observed, predicted).statistic
        ax.hexbin(observed, predicted, gridsize=35, mincnt=1, cmap="viridis")
        ax.set(title=f"Held-guide {rep}: Spearman={rho:.3f}",
               xlabel="Observed response percentile", ylabel="Held-guide prediction")

    for rep, color in (("background", "#4c78a8"), ("salient", "#e45756")):
        y = slice_prediction.responder.to_numpy(bool)
        probability = slice_prediction[f"{rep}_probability"].to_numpy(float)
        fpr, tpr, _ = roc_curve(y, probability)
        auc = roc_auc_score(y, probability)
        axes[1, 2].plot(fpr, tpr, color=color, lw=2, label=f"{rep} AUC={auc:.3f}")
    axes[1, 2].plot([0, 1], [0, 1], "--", color="#777777")
    axes[1, 2].set(title="Leave-one-slice-out responder prediction",
                   xlabel="False-positive rate", ylabel="True-positive rate")
    axes[1, 2].legend(frameon=False)
    fig.suptitle("SPARC-seq direct responder/non-responder evidence", fontsize=15)
    fig.savefig(args.out_dir / "response_definition_and_heldout_validation.png",
                dpi=240, bbox_inches="tight")
    plt.close(fig)

    definition = pd.DataFrame([{
        "population": "is_anchor=True cells",
        "expression_features": f"log-normalized expression, {args.n_pcs} standardized PCs",
        "local_control_reference": f"{args.k_nt} physically nearest same-slice NT anchors",
        "score": "Euclidean PC distance to local NT mean, empirical percentile in same-slice NT null",
        "responder_threshold": args.threshold,
        "target_responder_fraction": float(responder[target].mean()),
        "nt_false_positive_fraction": float((percentile[control] >= args.threshold).mean()),
        "haruka_used_to_define_label": False,
    }])
    definition.to_csv(args.out_dir / "responder_definition.csv", index=False)
    print(definition.to_string(index=False), flush=True)
    print(f"[save] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
