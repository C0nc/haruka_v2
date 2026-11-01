"""Summarize multi-section recovery and leakage for the composition-gated sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def cluster_ari(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    pred = KMeans(
        n_clusters=len(np.unique(y)), random_state=seed, n_init=20
    ).fit_predict(np.asarray(x, dtype=np.float32))
    return float(adjusted_rand_score(y, pred))


def cv_ridge_r2(
    x: np.ndarray, y: np.ndarray, seed: int, max_cells: int
) -> float:
    rng = np.random.default_rng(seed)
    if len(x) > max_cells:
        keep = np.sort(rng.choice(len(x), max_cells, replace=False))
        x, y = x[keep], y[keep]
    predictions = np.zeros_like(y, dtype=np.float64)
    folds = KFold(n_splits=3, shuffle=True, random_state=seed)
    for train, test in folds.split(x):
        xs = StandardScaler().fit(x[train])
        ys = StandardScaler().fit(y[train])
        model = Ridge(alpha=len(train) * 0.01).fit(
            xs.transform(x[train]), ys.transform(y[train])
        )
        predictions[test] = ys.inverse_transform(model.predict(xs.transform(x[test])))
    return float(r2_score(y, predictions, multioutput="uniform_average"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("."))
    parser.add_argument("--lambdas", nargs="+", type=float, required=True)
    parser.add_argument("--sections", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_cv_cells", type=int, default=10000)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, float | str]] = []
    for lam in args.lambdas:
        tag = f"{lam:g}"
        for section in args.sections:
            data_path = args.base / "data_composition_pairs_all" / f"mouse_brain_{section}_paired.h5ad"
            emb_dir = (
                args.base
                / f"results_shared_full_niche_compgated_lambda{tag}_cc"
                / f"mouse_brain_{section}_paired"
                / "haruka_z_s_cellcharter_seed0"
            )
            data = ad.read_h5ad(data_path, backed="r")
            names = np.load(emb_dir / "obs_names.npy", allow_pickle=True).astype(str)
            positions = pd.Index(data.obs_names.astype(str)).get_indexer(names)
            if (positions < 0).any():
                raise ValueError(f"Rows do not align for lambda={tag}, section={section}")
            obs = data.obs.iloc[positions]
            target = obs["condition"].astype(str).to_numpy() == "disease"
            fine = obs["gt_background_subregion"].astype(str).to_numpy()
            broad = obs["region"].astype(str).to_numpy()
            domain = obs["gt_sal_domain"].astype(str).to_numpy()
            z = np.load(emb_dir / "background.npy", mmap_mode="r")
            s = np.load(emb_dir / "salient.npy", mmap_mode="r")
            z_cc = np.load(emb_dir / "spatial.npy", mmap_mode="r")
            s_cc = np.load(emb_dir / "salient_spatial.npy", mmap_mode="r")
            with open(emb_dir / "direct_kmeans_ari.json") as handle:
                scores = json.load(handle)["scores"]
            bg = float(scores["background_subregion"]["ari"])
            sal = float(scores["salient_domain"]["ari"])
            rows.append(
                {
                    "lambda": lam,
                    "section": section,
                    "background_ari": bg,
                    "salient_ari": sal,
                    "balanced_recovery": (bg + sal) / 2,
                    "recovery_floor": min(bg, sal),
                    "z_salient_leakage_ari": cluster_ari(z_cc[target], domain[target], args.seed),
                    "s_background_fine_leakage_ari": cluster_ari(s_cc[target], fine[target], args.seed),
                    "s_background_broad_leakage_ari": cluster_ari(s_cc[target], broad[target], args.seed),
                    "cv_r2_z_from_s": cv_ridge_r2(s[target], z[target], args.seed, args.max_cv_cells),
                    "cv_r2_s_from_z": cv_ridge_r2(z[target], s[target], args.seed, args.max_cv_cells),
                }
            )
            print(f"lambda={tag} section={section} done", flush=True)

    out = args.out_dir
    (out / "figures").mkdir(parents=True, exist_ok=True)
    per = pd.DataFrame(rows)
    per.to_csv(out / "per_section_metrics.csv", index=False)
    metrics = [c for c in per.columns if c not in {"lambda", "section"}]
    summary = per.groupby("lambda")[metrics].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    # Selection is deliberately lexicographic: protect the weaker factor first,
    # then prefer lower wrong-slot leakage.  It does not mix arbitrary units.
    summary["mean_wrong_slot_leakage"] = (
        summary["z_salient_leakage_ari_mean"]
        + summary["s_background_fine_leakage_ari_mean"]
    ) / 2
    summary = summary.sort_values(
        ["recovery_floor_mean", "mean_wrong_slot_leakage"],
        ascending=[False, True],
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary.to_csv(out / "summary_and_ranking.csv", index=False)

    ordered = summary.sort_values("lambda")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for metric, label, color in [
        ("background_ari", "Background z-CC ARI", "#377eb8"),
        ("salient_ari", "Salient s-CC ARI", "#e41a1c"),
    ]:
        axes[0].errorbar(
            ordered["lambda"], ordered[f"{metric}_mean"],
            yerr=ordered[f"{metric}_std"], marker="o", capsize=3,
            color=color, label=label,
        )
    axes[0].set(title="Intended recovery", xlabel="Disentanglement lambda", ylabel="ARI")
    axes[0].legend(frameon=False)
    for metric, label, color in [
        ("z_salient_leakage_ari", "Domain from z-CC", "#984ea3"),
        ("s_background_fine_leakage_ari", "Background from s-CC", "#ff7f00"),
    ]:
        axes[1].errorbar(
            ordered["lambda"], ordered[f"{metric}_mean"],
            yerr=ordered[f"{metric}_std"], marker="o", capsize=3,
            color=color, label=label,
        )
    axes[1].set(title="Wrong-slot label leakage", xlabel="Disentanglement lambda", ylabel="ARI (lower is better)")
    axes[1].legend(frameon=False)
    for metric, label, color in [
        ("cv_r2_z_from_s", "z predicted from s", "#4daf4a"),
        ("cv_r2_s_from_z", "s predicted from z", "#a65628"),
    ]:
        axes[2].errorbar(
            ordered["lambda"], ordered[f"{metric}_mean"],
            yerr=ordered[f"{metric}_std"], marker="o", capsize=3,
            color=color, label=label,
        )
    axes[2].set(title="Cross-latent predictability", xlabel="Disentanglement lambda", ylabel="Held-out R2 (lower is better)")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "figures" / "lambda_recovery_and_leakage.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(ordered["mean_wrong_slot_leakage"], ordered["recovery_floor_mean"], s=65)
    for _, row in ordered.iterrows():
        ax.annotate(f"lambda={row['lambda']:g}", (row["mean_wrong_slot_leakage"], row["recovery_floor_mean"]), xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="Mean wrong-slot ARI (lower is better)", ylabel="Mean weaker-factor ARI (higher is better)", title="Recovery-disentanglement tradeoff")
    fig.tight_layout()
    fig.savefig(out / "figures" / "lambda_tradeoff.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
