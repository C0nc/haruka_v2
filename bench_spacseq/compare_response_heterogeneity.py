"""Compare Haruka background and salient latents on an external response score.

The evaluation target is intentionally not defined from either Haruka latent.
For every sgRNA anchor, it measures the expression-PC distance from physically
nearby, same-slice NT anchors and calibrates that distance against the same-slice
NT null.  Models are evaluated with complete sgRNAs or slices held out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def expression_pcs(adata: ad.AnnData, n_components: int, seed: int) -> np.ndarray:
    counts = adata.layers["counts"]
    totals = np.asarray(counts.sum(1)).ravel().clip(min=1)
    expression = counts.multiply(1e4 / totals[:, None])
    if sparse.issparse(expression):
        expression.data = np.log1p(expression.data)
        expression = expression.toarray()
    else:
        expression = np.log1p(expression)
    expression = StandardScaler().fit_transform(expression)
    return PCA(n_components, random_state=seed).fit_transform(expression)


def external_response_percentile(
    pcs: np.ndarray,
    coordinates: np.ndarray,
    control: np.ndarray,
    slices: np.ndarray,
    k_nt: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Expression distance to spatially local NTs, calibrated by slice NT null."""
    distance = np.full(len(pcs), np.nan)
    for slice_id in np.unique(slices):
        pool = np.flatnonzero(control & (slices == slice_id))
        members = np.flatnonzero(slices == slice_id)
        if len(pool) < 3:
            raise RuntimeError(f"slice {slice_id} has only {len(pool)} NT anchors")
        model = NearestNeighbors(n_neighbors=min(k_nt + 1, len(pool))).fit(
            coordinates[pool]
        )
        neighbors = model.kneighbors(coordinates[members], return_distance=False)
        for row, local in zip(members, neighbors):
            use = pool[local]
            use = use[use != row][:k_nt]
            distance[row] = np.linalg.norm(pcs[row] - pcs[use].mean(0))

    percentile = np.full(len(pcs), np.nan)
    for slice_id in np.unique(slices):
        null = distance[control & (slices == slice_id)]
        for row in np.flatnonzero(slices == slice_id):
            percentile[row] = (np.sum(null <= distance[row]) + 1) / (len(null) + 1)
    return distance, percentile


def held_group_predictions(
    x: np.ndarray,
    response: np.ndarray,
    responder: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    continuous = np.full(len(response), np.nan)
    probability = np.full(len(response), np.nan)
    classification = np.full(len(response), -1, dtype=int)
    for held in np.unique(groups):
        test = groups == held
        train = ~test
        continuous[test] = make_pipeline(
            StandardScaler(), Ridge(alpha=10)
        ).fit(x[train], response[train]).predict(x[test])
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0),
        ).fit(x[train], responder[train])
        probability[test] = classifier.predict_proba(x[test])[:, 1]
        classification[test] = classifier.predict(x[test])
    return continuous, probability, classification


def metric_values(
    response: np.ndarray,
    responder: np.ndarray,
    continuous: np.ndarray,
    probability: np.ndarray,
    classification: np.ndarray,
) -> dict[str, float]:
    return {
        "spearman": float(spearmanr(response, continuous).statistic),
        "r2": float(r2_score(response, continuous)),
        "auroc": float(roc_auc_score(responder, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(responder, classification)),
    }


def cluster_bootstrap(
    frame: pd.DataFrame,
    group_key: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    rng = np.random.default_rng(seed)
    groups = frame[group_key].unique()
    rows = []
    for iteration in range(n_bootstrap):
        sampled = rng.choice(groups, len(groups), replace=True)
        take = np.concatenate([
            np.flatnonzero(frame[group_key].to_numpy() == group) for group in sampled
        ])
        part = frame.iloc[take]
        if part.responder.nunique() < 2:
            continue
        for representation in ("background", "salient"):
            values = metric_values(
                part.response_percentile.to_numpy(),
                part.responder.to_numpy(bool),
                part[f"{representation}_continuous"].to_numpy(),
                part[f"{representation}_probability"].to_numpy(),
                part[f"{representation}_classification"].to_numpy(),
            )
            rows.extend(
                {
                    "iteration": iteration,
                    "representation": representation,
                    "metric": metric,
                    "value": value,
                }
                for metric, value in values.items()
            )
    boot = pd.DataFrame(rows)
    delta = {}
    for metric in sorted(boot.metric.unique()):
        pivot = boot[boot.metric == metric].pivot(
            index="iteration", columns="representation", values="value"
        ).dropna()
        difference = pivot.salient - pivot.background
        delta[metric] = {
            "salient_minus_background": float(difference.mean()),
            "ci_low": float(np.quantile(difference, 0.025)),
            "ci_high": float(np.quantile(difference, 0.975)),
            "bootstrap_probability_salient_better": float(np.mean(difference > 0)),
        }
    return boot, delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb-dir", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--k-nt", type=int, default=25)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.emb_dir)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(args.data)
    order = np.load(root / "obs_names.npy", allow_pickle=True).astype(str)
    adata = adata[order].copy()
    anchor = adata.obs.is_anchor.to_numpy(bool)
    anchored = adata[anchor].copy()
    obs = anchored.obs
    function = obs.function.astype(str).to_numpy()
    control = function == "NT"
    target = ~control
    slices = obs.slice_id.astype(str).to_numpy()
    guides = obs.guide_identity.astype(str).to_numpy()
    coordinates = obs[["array_row", "array_col"]].to_numpy(float)

    pcs = expression_pcs(anchored, args.n_pcs, args.seed)
    expression_distance, response = external_response_percentile(
        pcs, coordinates, control, slices, args.k_nt
    )
    responder = response >= 0.95
    representations = {
        name: np.load(root / f"{name}.npy")[anchor]
        for name in ("background", "salient")
    }

    target_frame = pd.DataFrame({
        "obs_name": anchored.obs_names.to_numpy()[target],
        "slice_id": slices[target],
        "function": function[target],
        "guide_identity": guides[target],
        "expression_response_distance": expression_distance[target],
        "response_percentile": response[target],
        "responder": responder[target],
    })
    result = {
        "response_definition": (
            "log-normalized expression-PC distance from physically nearest "
            "same-slice NT anchors; percentile calibrated against same-slice NT"
        ),
        "responder_definition": "external expression-response percentile >= 0.95",
        "n_target_anchors": int(target.sum()),
        "target_responder_fraction": float(responder[target].mean()),
        "nt_false_positive_fraction": float(responder[control].mean()),
        "cross_validation": {},
    }
    bootstrap_frames = []
    for scheme, groups in {
        "leave_one_guide_out": guides[target],
        "leave_one_slice_out": slices[target],
    }.items():
        frame = target_frame.copy()
        result["cross_validation"][scheme] = {"representations": {}}
        for name, x in representations.items():
            continuous, probability, classification = held_group_predictions(
                x[target], response[target], responder[target], groups
            )
            frame[f"{name}_continuous"] = continuous
            frame[f"{name}_probability"] = probability
            frame[f"{name}_classification"] = classification
            result["cross_validation"][scheme]["representations"][name] = metric_values(
                response[target], responder[target], continuous, probability, classification
            )
        group_key = "guide_identity" if scheme == "leave_one_guide_out" else "slice_id"
        boot, delta = cluster_bootstrap(
            frame, group_key, args.n_bootstrap, args.seed + len(bootstrap_frames)
        )
        boot["scheme"] = scheme
        bootstrap_frames.append(boot)
        result["cross_validation"][scheme]["paired_cluster_bootstrap_delta"] = delta
        frame.to_csv(Path(f"{args.out_prefix}.{scheme}.predictions.csv"), index=False)

    Path(f"{args.out_prefix}.metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    bootstrap.to_csv(Path(f"{args.out_prefix}.bootstrap.csv"), index=False)

    sns_colors = {"background": "#4C78A8", "salient": "#E45756"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, scheme in zip(axes, ("leave_one_guide_out", "leave_one_slice_out")):
        subset = bootstrap[
            (bootstrap.scheme == scheme) & bootstrap.metric.isin(["spearman", "auroc"])
        ]
        positions = {"spearman": 0, "auroc": 1}
        for offset, representation in [(-0.12, "background"), (0.12, "salient")]:
            for metric, position in positions.items():
                values = subset[
                    (subset.representation == representation) & (subset.metric == metric)
                ].value.to_numpy()
                point = result["cross_validation"][scheme]["representations"][representation][metric]
                low, high = np.quantile(values, [0.025, 0.975])
                ax.errorbar(
                    position + offset, point,
                    yerr=[[point - low], [high - point]], fmt="o", capsize=4,
                    color=sns_colors[representation], label=representation if metric == "spearman" else None,
                )
        ax.set_xticks([0, 1], ["Continuous\nSpearman", "Responder\nAUROC"])
        ax.axhline(0.5, color="#888888", lw=1, ls="--")
        ax.set_ylim(0, 0.8)
        ax.set_title(scheme.replace("_", " ").title())
        ax.set_ylabel("Held-out performance")
        ax.legend(frameon=False)
    fig.suptitle("SPARC-seq: external expression-defined response heterogeneity")
    fig.savefig(Path(f"{args.out_prefix}.comparison.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
