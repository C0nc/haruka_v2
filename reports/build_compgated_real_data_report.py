#!/usr/bin/env python3
"""Build a portable audit report for the composition-gated real-data reruns."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path("/home/yancui/haruka_v2")
OUT = ROOT / "reports" / "haruka_compgated_lambda10_real_data"


DATASETS = [
    {
        "name": "SPARC-seq perturbation",
        "status": "complete",
        "batch": "slice_id decoder injection; salient readouts use condition cells only",
        "finding": "s-CC carries more conditional sgRNA and perturbation-neighborhood information; responder discrimination improves modestly for leave-guide-out classification.",
        "metrics": "direct sgRNA CMI s=0.422 vs z=0.311; perturbation-neighborhood CMI s=0.490 vs z=0.345; responder AUROC s=0.742 vs z=0.720",
        "root": ROOT / "bench_spacseq/results_compgated_lambda10/analysis",
        "figures": [
            "figures/spacseq_condition_only_zcc_scc_umap.png",
            "direct_response/response_definition_and_heldout_validation.png",
            "unique_salient_spatial_biology/unique_salient_spatial_biology.png",
        ],
    },
    {
        "name": "STARmap Alzheimer disease",
        "status": "complete",
        "batch": "slice_id decoder injection using independent projections for gene, composition, and niche-expression decoders; slice is nested in condition, so technical removal is not identifiable from disease removal",
        "finding": "s-CC is more tightly related to plaque geography than z-CC and supports replicate-wise glial trajectories.",
        "metrics": "Aβ-distance within-replicate |rho|: s=0.405 vs z=0.078; p-tau: s=0.191 vs z=0.130",
        "root": ROOT / "ad_analysis_compgated_lambda10_batchremoved",
        "figures": [
            "figures/starmap_condition_only_zcc_scc_umap.png",
            "zcc_scc_pc1_plaque_distance_comparison.png",
            "scc_dpt_cellrank2/starmap_replicate_spatial_dpt_fate.png",
            "scc_dpt_cellrank2/glia_terminal_analysis/enrichment_and_composition/starmap_shared_scc_dpt_cell_subtype_stacked.png",
        ],
    },
    {
        "name": "CODEX Alzheimer disease",
        "status": "complete",
        "batch": "donor_id decoder injection using independent projections for all three decoders; protein MSE likelihood; donor is nested in condition, so technical removal is not identifiable from disease removal",
        "finding": "s-CC has substantially stronger within-donor association with dense and diffuse Aβ distance than z-CC.",
        "metrics": "dense-Aβ within-donor rho: s=0.630 vs z=-0.188; high-plaque AUROC s=0.834 vs z=0.825",
        "root": ROOT / "ad_analysis_compgated_lambda10_batchremoved",
        "figures": [
            "codex_readouts/plaque_four_slot_umap.png",
            "codex_spatial_distance_to_dense_diffuse_abeta.png",
            "scc_dpt_cellrank2/codex_replicate_spatial_dpt_fate.png",
            "scc_dpt_cellrank2/glia_terminal_analysis/enrichment_and_composition/codex_shared_scc_dpt_myeloid_state_stacked.png",
        ],
    },
    {
        "name": "KRAS-inhibitor Xenium",
        "status": "complete",
        "batch": "slice_id decoder injection with independent gene/composition/niche-expression projections; treated-only salient analysis; slice is nested in treatment arm, so this cannot guarantee a batch-free latent space",
        "finding": "background predicts the global Itgb6 residual-tumor axis slightly better, whereas s-CC resolves within-treated-section response niches and interpretable TME programs. Inferred CNV burden is shared rather than salient-specific.",
        "metrics": "relative-responder AUROC s=0.663 vs z=0.689; Itgb6 tumor AUROC s=0.987 vs z=0.994; held-out-section CNV-burden Spearman s=0.397 vs z=0.424",
        "root": ROOT / "bench_kras_xenium/results_compgated_lambda10_separate_batchheads/analysis",
        "figures": [
            "global/figures/treated_only_zcc_scc_umap.png",
            "responder_niche_interpretation/figures/treated_only_annotated_zcc_scc_niche_umap.png",
            "responder_niche_interpretation/figures/treated_only_niche_response_by_replicate.png",
            "macrophage/figures/treated_only_macrophage_zcc_scc_paga.png",
            "tumor_dpt/figures/tumor_scc_per_sample_diffmap_pseudotime_state.png",
            "tumor_infercnv/figures/tumor_scc_umap_pseudotime_cnv.png",
            "tumor_infercnv/figures/treated_tumor_spatial_pseudotime_cnv.png",
        ],
    },
    {
        "name": "PEtracer lineage",
        "status": "complete (four independent section pairs)",
        "batch": "each paired section fit independently; no cross-section nuisance key",
        "finding": "s improves lineage-family clustering over z, but z remains slightly better for lineage clade.",
        "metrics": "family ARI s=0.274±0.162 vs z=0.214±0.114; clade ARI s=0.167 vs z=0.201",
        "root": ROOT / "bench_petracer_lineage/results_compgated_lambda10/summary",
        "figures": ["ari_boxplot.png", "salient_spatial_enrichment/continuous_effect_sizes.png"],
    },
    {
        "name": "OSKM spatial aging",
        "status": "complete",
        "batch": "decoder-only slide injection; old control vs old OSKM; young used only for clock evaluation",
        "finding": "s-CC better captures the frozen-clock rejuvenation shift, but also retains substantial slide structure and is worse for the raw matched-response endpoint.",
        "metrics": "clock R2 s=0.155 vs z=-0.145; clock rho s=0.404 vs z=0.098; slide NMI s=0.536 vs z=0.014",
        "root": ROOT / "bench_oskm_reprogramming/results_compgated_lambda10/analysis",
        "figures": [
            "salient/figures/oskm_condition_only_zcc_scc_umap.png",
            "spatial_aging_clock/figures/zcc_scc_clock_prediction_comparison.png",
            "spatial_aging_clock/figures/spatial_clock_celltype_rejuvenation.png",
        ],
    },
    {
        "name": "GSE250346 lung fibrosis",
        "status": "complete",
        "batch": "spatial graphs are core-bounded; paired-donor transport is evaluated separately",
        "finding": "initial-state fate prediction is weak; s-CC is slightly above chance and reveals fibrosis-response niches, while z/background composition remains important for anatomy.",
        "metrics": "fate AUROC s-CC=0.505, z-CC=0.473; background composition=0.529; salient composition shift=0.524",
        "root": ROOT / "bench_lung_fibrosis_gse250346/results_compgated_lambda10/analysis",
        "figures": [
            "figures/condition_only_zcc_scc_umap.png",
            "figures/s_cc_cluster_biology_summary.png",
            "moscot_initial_state_fate/initial_state_fate_prediction.png",
            "moscot_spatiotemporal_salient_marginal/salient_target_marginals.png",
        ],
    },
    {
        "name": "Diabetic kidney disease",
        "status": "complete",
        "batch": "no specimen correction because specimen is condition-confounded",
        "finding": "s-CC improves stage and tissue-domain prediction, while cell identity remains primarily background.",
        "metrics": "stage balanced accuracy s=0.571 vs z=0.411; tissue-domain s=0.625 vs z=0.607",
        "root": ROOT / "bench_real_discovery/results_compgated_lambda10/dkd",
        "figures": ["analysis/four_slot_factor_matrix.png", "target_cc_analysis/figures/target_only_zcc_scc_umap.png"],
    },
    {
        "name": "Colitis",
        "status": "complete",
        "batch": "no specimen correction because specimen is condition-confounded; continuous normalized input uses Normal likelihood",
        "finding": "s-CC improves disease-stage clustering, but specimen NMI also increases, so part of the salient separation is study/sample structure.",
        "metrics": "stage ARI s=0.257 vs z=0.046; stage balanced accuracy s=0.775 vs z=0.730; sample NMI s=0.508 vs z=0.414",
        "root": ROOT / "bench_real_discovery/results_compgated_lambda10/colitis",
        "figures": ["analysis/four_slot_factor_matrix.png", "target_cc_analysis/figures/target_only_zcc_scc_umap.png"],
    },
    {
        "name": "MERFISH brain aging",
        "status": "complete",
        "batch": "technical_batch decoder injection",
        "finding": "z-CC retains anatomical region, while s-CC improves age-stage recovery; s nevertheless retains more sample identity.",
        "metrics": "stage balanced accuracy s=0.704 vs z=0.581; region ARI z=0.395 vs s=0.172; sample NMI s=0.290 vs z=0.103",
        "root": ROOT / "bench_aging/results_compgated_lambda10",
        "figures": ["analysis/four_slot_umap.png", "target_cc_analysis/figures/target_only_zcc_scc_umap.png"],
    },
]


def copy_assets() -> tuple[list[dict], list[str]]:
    assets = OUT / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    rendered = []
    for dataset_index, dataset in enumerate(DATASETS):
        copied = []
        for figure_index, relative in enumerate(dataset["figures"]):
            source = dataset["root"] / relative
            if not source.exists():
                missing.append(str(source))
                continue
            target = assets / f"d{dataset_index:02d}_f{figure_index:02d}{source.suffix.lower()}"
            shutil.copy2(source, target)
            copied.append((target.relative_to(OUT).as_posix(), source.name))
        rendered.append({**dataset, "copied": copied})
    return rendered, missing


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rendered, missing = copy_assets()
    cards = []
    sections = []
    for idx, dataset in enumerate(rendered):
        cards.append(
            f'<a class="card" href="#d{idx}"><strong>{html.escape(dataset["name"])}</strong>'
            f'<span>{html.escape(dataset["metrics"])}</span></a>'
        )
        figures = "".join(
            f'<figure><a href="{html.escape(path)}"><img loading="lazy" src="{html.escape(path)}"></a>'
            f'<figcaption>{html.escape(name)}</figcaption></figure>' for path, name in dataset["copied"]
        )
        sections.append(f'''<section id="d{idx}">
<h2>{html.escape(dataset["name"])}</h2>
<div class="status">{html.escape(dataset["status"])}</div>
<p><b>Outcome:</b> {html.escape(dataset["finding"])}</p>
<p><b>Key metrics:</b> {html.escape(dataset["metrics"])}</p>
<p><b>Batch/design:</b> {html.escape(dataset["batch"])}</p>
<p class="path"><b>Source:</b> {html.escape(str(dataset["root"]))}</p>
<div class="gallery">{figures}</div>
</section>''')

    css = """
    :root{--ink:#162033;--muted:#5f6b7a;--blue:#315efb;--paper:#f5f7fb;--line:#dfe5ee}
    *{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,sans-serif}
    header{padding:56px 6vw;color:white;background:linear-gradient(125deg,#172554,#315efb,#0f766e)}
    header h1{font-size:clamp(2rem,5vw,4rem);line-height:1;margin:0 0 16px}header p{max-width:1000px;color:#e6edff}
    main{width:min(1500px,92vw);margin:34px auto 80px}.overview{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
    .card{display:flex;flex-direction:column;gap:7px;padding:17px;background:white;border:1px solid var(--line);border-radius:13px;text-decoration:none;color:inherit}.card span{color:var(--muted);font-size:.9rem}
    section{margin-top:52px;padding-top:12px;border-top:1px solid var(--line)}h2{font-size:1.8rem;margin-bottom:8px}.status{display:inline-block;background:#dcfce7;color:#166534;border-radius:999px;padding:4px 10px;font-size:.82rem}
    .path{font:13px ui-monospace,monospace;color:var(--muted);overflow-wrap:anywhere}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}
    figure{margin:0;background:white;border:1px solid var(--line);border-radius:12px;overflow:hidden}figure img{width:100%;display:block}figcaption{padding:10px 13px;color:var(--muted);font-size:.85rem}
    .warning{padding:18px;border-left:5px solid #d97706;background:#fffbeb;margin:24px 0}footer{padding:30px 6vw;color:var(--muted);border-top:1px solid var(--line)}
    @media(max-width:600px){.gallery{grid-template-columns:1fr}}
    """
    report = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Haruka composition-gated real-data reruns</title><style>{css}</style></head><body>
<header><h1>Haruka composition-gated real-data reruns</h1><p>One-stage shared decoder, composition-gated niche expression, z/s dimension 32/32, and default independence λ=10. AD and KRAS use the requested decoder-only batch injection with separate projections for the three reconstruction heads; because their sample keys are nested in condition, this adjustment is not statistically identifiable as purely technical correction.</p></header>
<main><div class="warning"><b>Interpretation:</b> salient does not have to outperform background for every endpoint. Background-defined anatomy or pre-existing state should remain in z; s is supported when it adds condition-specific signal after held-out-sample validation. Decoder batch injection removes modeled nuisance at reconstruction time but is not adversarial latent invariance.</div>
<div class="overview">{"".join(cards)}</div>{"".join(sections)}
<section><h2>Audit notes</h2><p>Missing optional figures: {html.escape('; '.join(missing)) if missing else 'none'}.</p></section></main>
<footer>Generated from saved outputs in /home/yancui/haruka_v2. All metrics are observational unless their local source table states otherwise.</footer></body></html>'''
    (OUT / "index.html").write_text(report, encoding="utf-8")
    manifest = {
        "report": str(OUT / "index.html"),
        "protocol": {
            "neighbor_expression_mode": "composition_gated",
            "z_dim": 32,
            "s_dim": 32,
            "independence_lambda": 10,
            "ad_batch_mode": "decoder_only with distinct batch projections for gene, composition, and niche-expression heads; unidentifiable because sample/donor is nested in condition",
        },
        "datasets": [
            {k: str(v) if isinstance(v, Path) else v for k, v in dataset.items() if k != "figures"}
            for dataset in rendered
        ],
        "missing": missing,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame([
        {k: str(v) if isinstance(v, Path) else v for k, v in dataset.items() if k not in {"figures", "copied"}}
        for dataset in rendered
    ]).to_csv(OUT / "dataset_summary.csv", index=False)
    print(OUT / "index.html")


if __name__ == "__main__":
    build()
