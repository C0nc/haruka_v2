"""Multi-seed evaluation of the 2x2 spatial decomposition (spatial_attach='both').

WHY MULTI-SEED. The synthetic benchmark outcome is BIMODAL: background ARI vs the
`layer` truth lands near either ~0.45 (the background latent absorbed the lesion) or
~0.96 (clean). A mean over that is meaningless, so the headline metric here is the MODE
FRACTION -- the fraction of seeds that land in the clean mode -- and every other number
is reported as a median with an IQR rather than a mean with a standard deviation.

VARIANTS (all four share identical epochs, lr, latent widths, k, cluster method and
data; the only differences are the three fields named below):

    A  n_spatial_latent=0                                      vanilla two-latent
    B  n_spatial_latent=16, spatial_attach="background"         existing single spatial
    C  n_spatial_latent=16, spatial_attach="both"               the new 2x2
    D  same as C but spatial_invariance_penalty=0.0             ablation of the breaker

Usage::

    python run_2x2_eval.py --seeds 6 --out_dir outputs/eval_2x2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_haruka_spatial_example import _probe, make_synthetic  # noqa: E402

# ---------------------------------------------------------------------------
# HELD IDENTICAL ACROSS ALL FOUR VARIANTS.
# ---------------------------------------------------------------------------
SHARED = dict(
    n_x=30,
    n_y=20,
    n_genes=60,
    n_peaks=80,
    cov_num=8,
    latent_dim=32,
    n_layer=2,
    k_spatial=8,
    max_epochs=400,
    lr=1e-3,
    batch_size=512,
    cluster_method="gmm",
    reg_to_use="mi",
    lam_mi=5.0,
    kl_spatial_weight=0.01,  # SIMVI's 100x down-weight; kept, not tuned
    spatial_encoder="gat",
    background_cluster_number=3,
    salient_cluster_num=2,
    spatial_cluster_num=2,
)

VARIANTS = {
    "A_vanilla": dict(n_spatial_latent=0),
    "A_me_bg": dict(n_spatial_latent=0, me_reduction="sum", me_from_background_only=True),
    "A_me_bg_w3": dict(n_spatial_latent=0, me_reduction="sum", me_weight=3.0, me_from_background_only=True),

    "A_covet_sum": dict(n_spatial_latent=0, me_reduction="sum"),
    "A_covet_sum_w3": dict(n_spatial_latent=0, me_reduction="sum", me_weight=3.0),

    "B_background": dict(n_spatial_latent=16, spatial_attach="background"),
    "E_salient": dict(n_spatial_latent=16, spatial_attach="salient"),
    "B_w1": dict(n_spatial_latent=16, spatial_attach="background", wasserstein_penalty=1.0),
    "B_w5": dict(n_spatial_latent=16, spatial_attach="background", wasserstein_penalty=5.0),
    "B_ep800": dict(n_spatial_latent=16, spatial_attach="background", max_epochs=800),
    "B_lr3e4": dict(n_spatial_latent=16, spatial_attach="background", lr=3e-4),

    "B16_h4": dict(n_spatial_latent=16, spatial_attach="background", n_attention_heads=4),
    "B32": dict(n_spatial_latent=32, spatial_attach="background"),
    "B32_h4": dict(n_spatial_latent=32, spatial_attach="background", n_attention_heads=4),

    "C_both": dict(n_spatial_latent=16, spatial_attach="both",
                   spatial_invariance_penalty=0.5),
    "D_both_noinv": dict(n_spatial_latent=16, spatial_attach="both",
                         spatial_invariance_penalty=0.0),
}


def run_one(variant: str, seed: int) -> dict:
    """Train one RNA model and score it. Returns a flat dict of metrics."""
    from haruka_spatial import HarukaSpatial, seed_everything

    # Matches HarukaMultiomicsSpatial._run_modality, which seeds immediately before
    # constructing the model. Without it the variants would not share an init stream.
    seed_everything(seed)

    cfg = VARIANTS[variant]
    rna, _ = make_synthetic(
        n_x=SHARED["n_x"], n_y=SHARED["n_y"], n_genes=SHARED["n_genes"],
        n_peaks=SHARED["n_peaks"], seed=seed,
    )

    model = HarukaSpatial(
        rna, cov_gene_num=SHARED["cov_num"], seed=seed, k_spatial=SHARED["k_spatial"]
    )
    model.construct_cov()

    setup = dict(
        latent_dim=SHARED["latent_dim"],
        gene_likelihood="poisson",
        batch_key="slice_id",
        condition_key="condition",
        condition_label="disease",
        n_spatial_latent=cfg["n_spatial_latent"],
        reg_to_use=SHARED["reg_to_use"],
        lam_mi=SHARED["lam_mi"],
        kl_spatial_weight=SHARED["kl_spatial_weight"],
    )
    if cfg["n_spatial_latent"] > 0:
        setup["spatial_attach"] = cfg["spatial_attach"]
        setup["spatial_encoder"] = SHARED["spatial_encoder"]
        if cfg["spatial_attach"] == "both":
            setup["spatial_invariance_penalty"] = cfg["spatial_invariance_penalty"]
    model.setup_model(**setup)

    model.train(
        max_epochs=SHARED["max_epochs"],
        batch_size=SHARED["batch_size"],
        plan_kwargs=dict(lr=SHARED["lr"]),
        enable_progress_bar=False,
    )
    model.extract_rep(n_layer=SHARED["n_layer"])
    model.cluster(
        salient_cluster_num=SHARED["salient_cluster_num"],
        background_cluster_number=SHARED["background_cluster_number"],
        spatial_cluster_num=SHARED["spatial_cluster_num"],
        seed=seed,
        cluster_method=SHARED["cluster_method"],
    )

    out = model.output_adata  # target (disease) arm only, as in the example script
    truth = {t: np.asarray(out.obs[t]).astype(str) for t in ("layer", "lesion", "niche")}

    def ari(kind, t):
        col = f"cluster_haruka_{kind}"
        if col not in out.obs.columns:
            return float("nan")
        return float(adjusted_rand_score(truth[t], np.asarray(out.obs[col]).astype(int)))

    res = {
        "variant": variant,
        "seed": seed,
        "bg_ari_layer": ari("background", "layer"),
        "bg_ari_lesion": ari("background", "lesion"),
        "sal_ari_lesion": ari("salient", "lesion"),
    }

    if cfg["n_spatial_latent"] > 0:
        sp = np.asarray(out.obsm["spatial_rep"])
        res["sp_probe_niche"] = _probe(sp, truth["niche"], seed)
        res["sp_probe_layer"] = _probe(sp, truth["layer"], seed)

    # s_sp on CONTROL cells. No mask forces it to zero -- the claim is that it falls
    # out of s_int being ~0 there -- so it has to be measured on the full object,
    # since output_adata carries the target arm only.
    if cfg.get("spatial_attach") == "both":
        full = model.input_adata
        ctrl = np.asarray(full.obs["condition"] != "disease")
        get = model.model.get_latent_representation
        for name, kind in (("s_sp", "salient_spatial"), ("z_sp", "spatial"),
                           ("s_int", "salient")):
            arr = np.asarray(get(full, representation_kind=kind))
            c = float(np.abs(arr[ctrl]).mean())
            t = float(np.abs(arr[~ctrl]).mean())
            res[f"{name}_absmean_ctrl"] = c
            res[f"{name}_absmean_tgt"] = t
            res[f"{name}_ctrl_over_tgt"] = c / max(t, 1e-12)

    return res


def summarize(rows: list[dict]) -> str:
    """Mode fraction plus medians and IQRs, per variant."""
    def q(vals, p):
        vals = [v for v in vals if v == v]
        return float(np.percentile(vals, p)) if vals else float("nan")

    def med_iqr(vals):
        vals = [v for v in vals if v == v]
        if not vals:
            return "        n/a"
        return f"{q(vals, 50):.3f} [{q(vals, 25):.3f},{q(vals, 75):.3f}]"

    metrics = [
        ("bg_ari_layer", "bg ARI vs layer"),
        ("bg_ari_lesion", "bg ARI vs lesion"),
        ("sal_ari_lesion", "sal ARI vs lesion"),
        ("sp_probe_niche", "sp probe vs niche"),
        ("sp_probe_layer", "sp probe vs layer"),
    ]
    lines = []
    hdr = f"{'variant':<15}{'n':>3}{'MODE FRAC':>11}  " + "".join(
        f"{lab:<26}" for _, lab in metrics
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for variant in VARIANTS:
        sub = [r for r in rows if r["variant"] == variant and "error" not in r]
        if not sub:
            lines.append(f"{variant:<15}{0:>3}   (no successful runs)")
            continue
        # THE HEADLINE: clean mode = background recovered the layers and did NOT
        # absorb the lesion.
        clean = [
            r for r in sub
            if r["bg_ari_layer"] > 0.8 and r["bg_ari_lesion"] < 0.15
        ]
        frac = len(clean) / len(sub)
        cells = "".join(
            f"{med_iqr([r.get(k, float('nan')) for r in sub]):<26}" for k, _ in metrics
        )
        lines.append(
            f"{variant:<15}{len(sub):>3}{frac:>8.2f} ({len(clean)}/{len(sub)})  {cells}"
        )

    # s_sp-on-controls check, "both" variants only.
    lines.append("")
    lines.append("Is s_sp empirically ~0 on control cells? (mean |value|, full object)")
    lines.append(
        f"{'variant':<15}{'s_int c/t':>12}{'s_sp c/t':>12}{'z_sp c/t':>12}"
        f"{'s_sp |ctrl|':>14}{'z_sp |ctrl|':>14}"
    )
    for variant in VARIANTS:
        sub = [r for r in rows if r["variant"] == variant and "s_sp_ctrl_over_tgt" in r]
        if not sub:
            continue
        g = lambda k: q([r[k] for r in sub], 50)  # noqa: E731
        lines.append(
            f"{variant:<15}{g('s_int_ctrl_over_tgt'):>12.3f}"
            f"{g('s_sp_ctrl_over_tgt'):>12.3f}{g('z_sp_ctrl_over_tgt'):>12.3f}"
            f"{g('s_sp_absmean_ctrl'):>14.4f}{g('z_sp_absmean_ctrl'):>14.4f}"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--out_dir", default="outputs/eval_salient")
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "results.jsonl")

    done = set()
    rows = []
    if os.path.exists(results_path):
        with open(results_path) as fh:
            for line in fh:
                r = json.loads(line)
                rows.append(r)
                done.add((r["variant"], r["seed"]))
        print(f"Resuming: {len(done)} runs already on disk.")

    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    variants = [v for v in args.variants.split(",") if v]
    todo = [(v, s) for v in variants for s in seeds if (v, s) not in done]
    print(f"{len(todo)} runs to do ({len(variants)} variants x {len(seeds)} seeds).")

    for i, (variant, seed) in enumerate(todo, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(todo)}] {variant} seed={seed} ...", flush=True)
        try:
            res = run_one(variant, seed)
        except Exception as exc:  # keep the sweep alive; a dead cell is not a result
            traceback.print_exc()
            res = {"variant": variant, "seed": seed, "error": repr(exc)}
        res["secs"] = round(time.time() - t0, 1)
        rows.append(res)
        with open(results_path, "a") as fh:
            fh.write(json.dumps(res) + "\n")
        print(f"    {json.dumps({k: v for k, v in res.items() if k != 'variant'})}",
              flush=True)

    print("\n" + "=" * 100)
    print("2x2 SPATIAL DECOMPOSITION -- MULTI-SEED RESULT")
    print("epochs/lr/latent widths/k/cluster method IDENTICAL across variants:")
    print(f"  {SHARED}")
    print("MODE FRAC = fraction of seeds with bg-vs-layer ARI > 0.8 AND "
          "bg-vs-lesion ARI < 0.15")
    print("other cells are median [Q1,Q3] over seeds")
    print("=" * 100 + "\n")
    table = summarize(rows)
    print(table)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as fh:
        fh.write(table + "\n")


if __name__ == "__main__":
    main()
