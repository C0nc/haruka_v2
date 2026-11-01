#!/usr/bin/env python3
"""Build an offline HTML summary of Haruka AD and lung-fibrosis analyses."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path("/home/yancui/haruka_v2")
AD = ROOT / "ad_analysis"
DPT = AD / "scc_dpt_cellrank2"
GLIA = DPT / "glia_terminal_analysis"
COMP = GLIA / "enrichment_and_composition"
LUNG = ROOT / "bench_lung_fibrosis_gse250346/results/analysis"
DEFAULT_OUT = ROOT / "reports/haruka_codex_starmap_lung_2026-08-31"


FIGURES = {
    "starmap_overview": (
        AD / "starmap_latest_overview.png",
        "STARmap overview",
        "Final shared-decoder Haruka result for control and AD sections, including background z-CC and salient s-CC structure.",
    ),
    "starmap_plaque": (
        AD / "starmap_spatial_distance_to_abeta_ptau.png",
        "STARmap plaque-distance maps",
        "Disease-section distances to Aβ-associated cells, p-tau-associated cells, and either plaque-associated population.",
    ),
    "ad_pc1": (
        AD / "zcc_scc_pc1_plaque_distance_comparison.png",
        "Background versus salient plaque association",
        "Matched z-CC and s-CC PC1 associations with plaque distance. PC1 signs are arbitrary; effect magnitude and replicate consistency are the interpretable quantities.",
    ),
    "starmap_terminal": (
        GLIA / "starmap_micro_astro_oligo_spatial_terminal_stages.png",
        "STARmap glial terminal-state geography",
        "Spatial DPT and inferred terminal states for microglia, astrocytes, and oligodendrocytes, analyzed separately within each disease replicate.",
    ),
    "starmap_ctsub": (
        COMP / "starmap_shared_scc_dpt_cell_subtype_stacked.png",
        "STARmap ct_sub composition along shared s-CC DPT",
        "Fine-grained ct_sub proportions only (Astro1–3, Micro1–3, Oligo1–3), binned into replicate-specific DPT quintiles.",
    ),
    "starmap_ctsub_within": (
        COMP / "starmap_within_lineage_dpt_cell_subtype_stacked.png",
        "Within-lineage STARmap ct_sub shifts",
        "Subtype proportions along lineage-specific s-CC DPT; the replicated Micro1-to-Micro3 shift is the strongest effect.",
    ),
    "starmap_enrichment": (
        COMP / "starmap_glial_terminal_functional_enrichment.png",
        "STARmap glial functional enrichment",
        "Local over-representation analysis of direction-consistent early molecular predictors of inferred terminal fate.",
    ),
    "starmap_predictors": (
        GLIA / "starmap_glial_initial_molecular_terminal_predictors.png",
        "STARmap initial molecular predictors",
        "Genes measured in the early trajectory region that associate with later inferred plaque-proximal versus alternative terminal states.",
    ),
    "codex_overview": (
        AD / "codex_plaque_four_slot_umap_cc3.png",
        "CODEX four-slot plaque overview",
        "Background and salient representations for the AD CODEX myeloid dataset using three-hop, donor-bounded CellCharter.",
    ),
    "codex_plaque": (
        AD / "codex_spatial_distance_to_dense_diffuse_abeta.png",
        "CODEX dense and diffuse Aβ distances",
        "Distances to official dense- and diffuse-Aβ plaque objects in AD donors. CODEX has no tau marker, so tau distance is not evaluated.",
    ),
    "codex_terminal": (
        GLIA / "codex_myeloid_macrophage_spatial_terminal_stages.png",
        "CODEX myeloid terminal-state geography",
        "Per-donor spatial DPT, transition potential, and inferred plaque-associated terminal stage.",
    ),
    "codex_stack": (
        COMP / "codex_shared_scc_dpt_myeloid_state_stacked.png",
        "CODEX myeloid states along shared s-CC DPT",
        "Published myeloid Leiden-state proportions across shared salient-CC pseudotime in four AD donors.",
    ),
    "codex_predictors": (
        DPT / "codex_initial_molecular_terminal_predictors.png",
        "CODEX initial protein predictors",
        "Early-state protein associations with inferred plaque-proximal versus alternative terminal outcomes.",
    ),
    "lung_umap": (
        LUNG / "figures/condition_only_zcc_scc_umap.png",
        "Lung fibrosis z-CC and s-CC",
        "More-affected-core UMAPs. z-CC preferentially retains tissue organization; s-CC highlights heterogeneous fibrosis-associated response niches.",
    ),
    "lung_pathology": (
        LUNG / "figures/pathology_vs_representation_magnitude.png",
        "Lung severity versus representation magnitude",
        "Core-level pathology relationships for background, salient, and decoded composition-shift magnitudes.",
    ),
    "lung_biology": (
        LUNG / "figures/s_cc_cluster_biology_summary.png",
        "Lung s-CC response-niche biology",
        "Response-cluster composition shifts, deposited annotations, and pathology relationships.",
    ),
    "lung_hotspot": (
        LUNG / "hotspot_cc_knn/hotspot_module_heatmap_background_salient.png",
        "Background and salient spatial gene programs",
        "Hotspot modules learned on z-CC and condition-cell s-CC neighborhood graphs.",
    ),
    "lung_program_spatial": (
        LUNG / "hotspot_cc_knn/key_background_salient_programs_all_slices_original_spatial.png",
        "Spatial distribution of key lung programs",
        "Selected background- and salient-derived gene programs projected back to original tissue coordinates.",
    ),
    "lung_alignment": (
        LUNG / "moscot_fullrank/moscot_alignment_agreement_summary.png",
        "Paired-donor moscot alignment audit",
        "Full-rank fused-GW alignment agreement for the six genuinely paired less-to-more affected core comparisons.",
    ),
    "lung_fate_umap": (
        LUNG / "moscot_initial_state_fate/initial_state_fate_umap_spatial.png",
        "Initial alveolar-fibroblast fate structure",
        "Less-affected alveolar fibroblasts colored by inferred probability of reaching activated fibrotic fibroblast states.",
    ),
    "lung_fate_prediction": (
        LUNG / "moscot_initial_state_fate/initial_state_fate_prediction.png",
        "Initial-state fate prediction",
        "Leave-donor-out prediction of primed versus stable initial alveolar fibroblasts from expression and background z-CC.",
    ),
    "lung_de": (
        LUNG / "moscot_initial_state_fate/initial_state_DE_volcano.png",
        "Primed versus stable initial-state genes",
        "Differential expression between initial cells with high versus low inferred fibrotic-fate probability.",
    ),
    "lung_go": (
        LUNG / "moscot_initial_state_fate/initial_primed_GO_local_strict.png",
        "Initial primed-state enrichment",
        "Local GO enrichment of the strict initial primed signature; ECM organization is driven by COL1A1 and POSTN.",
    ),
    "lung_salient_marginal": (
        LUNG / "moscot_spatiotemporal_salient_marginal/salient_target_marginals.png",
        "Salient-weighted target marginals",
        "Sensitivity analysis showing how s-CC-derived target weighting changes transport mass over more-affected states.",
    ),
}


TABLES = {
    "ad_plaque_pc1.csv": AD / "cc_pc1_plaque_distance_association.csv",
    "ad_trajectory_cluster_summary.csv": DPT / "cellrank_cluster_summary.csv",
    "ad_composition_shift_statistics.csv": COMP / "pseudotime_composition_shift_statistics.csv",
    "starmap_ct_sub_composition.csv": COMP / "starmap_shared_ct_sub_composition.csv",
    "starmap_within_lineage_ct_sub_composition.csv": COMP / "starmap_within_lineage_ct_sub_composition.csv",
    "starmap_glial_functional_enrichment.csv": COMP / "starmap_glial_terminal_functional_enrichment.csv",
    "starmap_glial_initial_predictors.csv": GLIA / "starmap_glial_top_initial_molecular_terminal_predictors.csv",
    "codex_initial_protein_associations.csv": GLIA / "codex_myeloid_initial_protein_terminal_associations.csv",
    "lung_clustering_metrics.csv": LUNG / "tables/zcc_scc_clustering_metrics.csv",
    "lung_pathology_associations.csv": LUNG / "tables/salient_response_magnitude_pathology.csv",
    "lung_moscot_alignment_metrics.csv": LUNG / "moscot_fullrank/moscot_alignment_metrics.csv",
    "lung_initial_fate_prediction.csv": LUNG / "moscot_initial_state_fate/initial_state_fate_prediction.csv",
    "lung_initial_state_de.csv": LUNG / "moscot_initial_state_fate/initial_state_DE_primed_vs_stable.csv",
    "lung_initial_state_go.csv": LUNG / "moscot_initial_state_fate/initial_primed_GO_local_strict.csv",
    "lung_hotspot_module_annotations.csv": LUNG / "hotspot_cc_knn/module_annotations.csv",
    "lung_salient_marginal_routes.csv": LUNG / "moscot_spatiotemporal_salient_marginal/pooled/top_salient_marginal_route_changes.csv",
}


def dataframe_html(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    view = frame.loc[:, columns].copy()
    for column, fmt in formats.items():
        if column in view:
            view[column] = view[column].map(lambda x: fmt.format(x) if pd.notna(x) else "—")
    return view.to_html(index=False, border=0, classes="data-table", escape=True)


def copy_inputs(out: Path) -> tuple[dict[str, tuple[str, str, str]], list[str]]:
    assets = out / "assets"
    tables = out / "tables"
    assets.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    copied_figures: dict[str, tuple[str, str, str]] = {}
    missing: list[str] = []
    for key, (source, title, caption) in FIGURES.items():
        if not source.exists():
            missing.append(str(source))
            continue
        target = assets / f"{key}{source.suffix.lower()}"
        shutil.copy2(source, target)
        copied_figures[key] = (f"assets/{target.name}", title, caption)
    for target_name, source in TABLES.items():
        if not source.exists():
            missing.append(str(source))
            continue
        shutil.copy2(source, tables / target_name)
    return copied_figures, missing


def figure(figures: dict[str, tuple[str, str, str]], key: str, wide: bool = False) -> str:
    if key not in figures:
        return ""
    path, title, caption = figures[key]
    cls = "figure wide" if wide else "figure"
    return (
        f'<figure class="{cls}"><a href="{html.escape(path)}" target="_blank">'
        f'<img loading="lazy" src="{html.escape(path)}" alt="{html.escape(title)}"></a>'
        f"<figcaption><strong>{html.escape(title)}</strong><span>{html.escape(caption)}</span></figcaption></figure>"
    )


def download_list() -> str:
    items = []
    for target_name, source in TABLES.items():
        if source.exists():
            items.append(f'<li><a href="tables/{html.escape(target_name)}">{html.escape(target_name)}</a></li>')
    return "<ul class=\"downloads\">" + "".join(items) + "</ul>"


def build_report(out: Path) -> None:
    figures, missing = copy_inputs(out)

    plaque = pd.read_csv(AD / "cc_pc1_plaque_distance_association.csv")
    composition = pd.read_csv(COMP / "pseudotime_composition_shift_statistics.csv")
    codex_proteins = pd.read_csv(GLIA / "codex_myeloid_initial_protein_terminal_associations.csv")
    starmap_predictors = pd.read_csv(GLIA / "starmap_glial_top_initial_molecular_terminal_predictors.csv")
    enrichment = pd.read_csv(COMP / "starmap_glial_terminal_functional_enrichment.csv")
    lung_pathology = pd.read_csv(LUNG / "tables/salient_response_magnitude_pathology.csv")
    lung_metrics = pd.read_csv(LUNG / "tables/zcc_scc_clustering_metrics.csv")
    lung_alignment = pd.read_csv(LUNG / "moscot_fullrank/moscot_alignment_metrics.csv")
    lung_fate = pd.read_csv(LUNG / "moscot_initial_state_fate/initial_state_fate_prediction.csv")
    lung_de = pd.read_csv(LUNG / "moscot_initial_state_fate/initial_state_DE_primed_vs_stable.csv")
    lung_go = pd.read_csv(LUNG / "moscot_initial_state_fate/initial_primed_GO_local_strict.csv")
    with open(LUNG / "hotspot_cc_knn/biological_summary.json") as handle:
        hotspot_summary = json.load(handle)

    plaque_view = plaque[plaque.endpoint.isin([
        "either-plaque distance", "native any-plaque distance"
    ])].copy()
    plaque_view["representation"] = plaque_view.dataset + " " + plaque_view.representation

    star_comp = composition[composition.analysis.str.contains("starmap")].copy()
    star_comp["panel"] = star_comp.main_cluster.fillna("Shared glia")
    codex_comp = composition[composition.analysis == "codex_shared_state"].copy()

    sig_enrichment = enrichment[enrichment["Adjusted P-value"] < 0.10].copy()
    sig_enrichment = sig_enrichment.sort_values("Adjusted P-value").head(8)
    de_sig = lung_de[lung_de.padj_bh < 0.05].sort_values("padj_bh")
    go_sig = lung_go[lung_go["Adjusted P-value"] < 0.05].sort_values("Adjusted P-value").head(8)

    original = lung_alignment[lung_alignment.geometry == "original_shape_standardized"].set_index(["donor", "target_slice"])
    aligned = lung_alignment[lung_alignment.geometry == "moscot_zcc_aligned"].set_index(["donor", "target_slice"])
    improvements = (aligned[["cell_type_agreement", "lineage_agreement", "CNiche_agreement", "TNiche_agreement"]] -
                    original[["cell_type_agreement", "lineage_agreement", "CNiche_agreement", "TNiche_agreement"]]).mean()

    starmap_q = star_comp[star_comp.panel != "Shared glia"]
    micro_v = starmap_q[starmap_q.panel == "Micro"].cramers_v.mean()
    astro_v = starmap_q[starmap_q.panel == "Astro"].cramers_v.mean()
    oligo_v = starmap_q[starmap_q.panel == "Oligo"].cramers_v.mean()
    codex_v = codex_comp.cramers_v

    top_star = starmap_predictors[
        (starmap_predictors.lineage == "Micro") &
        (starmap_predictors.terminal_association == "Plaque-proximal terminal")
    ].sort_values("meta_q_value").head(6)
    top_codex = codex_proteins.sort_values("meta_q_value").head(8)

    css = """
    :root { --ink:#172033; --muted:#5f6b7a; --paper:#f5f7fb; --card:#fff;
      --blue:#315efb; --teal:#0b8f87; --amber:#d97706; --line:#dfe5ee; --soft:#eef3ff; }
    * { box-sizing:border-box; }
    html { scroll-behavior:smooth; }
    body { margin:0; color:var(--ink); background:var(--paper);
      font:16px/1.58 Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }
    header { color:#fff; background:linear-gradient(135deg,#101d3a,#2444a5 58%,#087c78);
      padding:64px max(5vw,24px) 54px; }
    header h1 { margin:0 0 12px; font-size:clamp(2rem,4.5vw,4rem); line-height:1.06; letter-spacing:-.035em; }
    header p { max-width:900px; margin:0; color:#dce7ff; font-size:1.12rem; }
    .meta { display:flex; flex-wrap:wrap; gap:10px; margin-top:24px; }
    .pill { padding:7px 12px; border:1px solid #ffffff44; border-radius:999px; background:#ffffff12; }
    nav { position:sticky; top:0; z-index:10; display:flex; gap:18px; overflow:auto;
      padding:13px max(5vw,24px); background:#ffffffee; backdrop-filter:blur(10px); border-bottom:1px solid var(--line); }
    nav a { color:#344054; text-decoration:none; white-space:nowrap; font-weight:650; }
    main { width:min(1450px,92vw); margin:38px auto 80px; }
    section { scroll-margin-top:72px; margin:0 0 56px; }
    h2 { font-size:clamp(1.6rem,3vw,2.35rem); letter-spacing:-.025em; margin:0 0 16px; }
    h3 { margin:26px 0 12px; font-size:1.2rem; }
    .lead { max-width:1000px; color:var(--muted); font-size:1.06rem; }
    .callout { padding:20px 22px; border-left:5px solid var(--blue); background:var(--soft); border-radius:0 12px 12px 0; }
    .grid { display:grid; grid-template-columns:repeat(12,1fr); gap:18px; margin:20px 0; }
    .card { grid-column:span 4; background:var(--card); border:1px solid var(--line);
      border-radius:16px; padding:20px; box-shadow:0 8px 28px #1e293b0a; }
    .card.wide { grid-column:span 6; }
    .card h3 { margin:0 0 7px; }
    .metric { font-size:2rem; line-height:1.1; font-weight:780; letter-spacing:-.035em; color:#183b9b; }
    .metric-label { color:var(--muted); font-size:.92rem; }
    .figure-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:22px; margin:22px 0; }
    figure { margin:0; background:#fff; border:1px solid var(--line); border-radius:16px; overflow:hidden; }
    figure.wide { grid-column:1/-1; }
    figure img { display:block; width:100%; height:auto; background:#fff; }
    figcaption { display:flex; flex-direction:column; gap:5px; padding:14px 16px 17px; }
    figcaption span { color:var(--muted); font-size:.92rem; }
    .table-wrap { overflow:auto; border:1px solid var(--line); border-radius:14px; background:#fff; margin:16px 0; }
    .data-table { width:100%; border-collapse:collapse; font-size:.9rem; }
    .data-table th { background:#eef2f8; text-align:left; position:sticky; top:0; }
    .data-table th,.data-table td { padding:9px 11px; border-bottom:1px solid #edf0f4; white-space:nowrap; }
    .key-findings li { margin-bottom:10px; }
    .note { color:var(--muted); font-size:.92rem; }
    .downloads { columns:2; padding-left:22px; }
    .downloads li { break-inside:avoid; margin:5px 0; }
    a { color:#214fc6; }
    footer { color:#6b7280; border-top:1px solid var(--line); padding:28px max(5vw,24px) 50px; }
    @media (max-width:900px) { .card,.card.wide{grid-column:1/-1}.figure-grid{grid-template-columns:1fr}.downloads{columns:1} }
    @media print { nav{display:none} body{background:#fff} main{width:96%} section{break-inside:avoid} }
    """

    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Haruka spatial contrastive analysis — AD and lung fibrosis</title>
  <style>{css}</style>
</head>
<body>
<header>
  <h1>Haruka spatial contrastive analysis</h1>
  <p>Integrated summary of CODEX Alzheimer’s disease, STARmap Alzheimer’s disease, and GSE250346 human lung fibrosis. The report emphasizes what background z-CC and salient s-CC uniquely recover, how the salient trajectories relate to pathology, and which findings replicate across samples.</p>
  <div class="meta"><span class="pill">Generated 31 August 2026</span><span class="pill">Final shared three-objective decoder</span><span class="pill">Three-hop CellCharter</span><span class="pill">λ = 25</span></div>
</header>
<nav><a href="#summary">Executive summary</a><a href="#method">Model</a><a href="#starmap">STARmap AD</a><a href="#codex">CODEX AD</a><a href="#lung">Lung fibrosis</a><a href="#limits">Interpretation</a><a href="#data">Downloads</a></nav>
<main>
<section id="summary">
  <h2>Executive summary</h2>
  <div class="callout"><strong>Overall result.</strong> Across both AD assays, s-CC is more tightly associated with plaque geography than z-CC and supports reproducible disease-state composition shifts. In lung fibrosis, the decoded salient composition-shift magnitude increases with pathology, while background z-CC is the more useful space for anatomical alignment and for predicting which initial alveolar fibroblasts are primed toward activated fibrotic states.</div>
  <div class="grid">
    <div class="card"><div class="metric">ρ = 0.61</div><div class="metric-label">CODEX s-CC PC1 versus dense-Aβ distance; sample-controlled ρ = 0.61</div></div>
    <div class="card"><div class="metric">V = {micro_v:.2f}</div><div class="metric-label">Mean STARmap Micro ct_sub shift across the two disease replicates</div></div>
    <div class="card"><div class="metric">V = {codex_v.min():.2f}–{codex_v.max():.2f}</div><div class="metric-label">CODEX myeloid-state redistribution over s-CC DPT across four donors</div></div>
    <div class="card"><div class="metric">ρ = 0.46</div><div class="metric-label">Lung decoded composition-shift magnitude versus pathology across 20 more-affected cores</div></div>
    <div class="card"><div class="metric">AUC = 0.617</div><div class="metric-label">Leave-donor-out initial fibroblast fate prediction from background z-CC; permutation p = 0.0099</div></div>
    <div class="card"><div class="metric">1.41M</div><div class="metric-label">Lung cells jointly fitted without holdout or downsampling</div></div>
  </div>
  <ul class="key-findings">
    <li><strong>STARmap:</strong> salient plaque association is consistent across Aβ, p-tau, and combined plaque distances. Microglia show a replicated Micro1 → Micro3 shift; astrocytes shift moderately toward Astro3; oligodendrocyte subtype redistribution is weak.</li>
    <li><strong>CODEX:</strong> s-CC plaque-distance association is substantially stronger than z-CC and all four donors show large myeloid-state redistribution along the shared s-CC trajectory.</li>
    <li><strong>Lung:</strong> z-CC retains cellular niche organization and supports paired-donor alignment, whereas s-CC highlights heterogeneous response niches and fibrosis-linked composition shifts. Initial background context predicts subsequent fibrotic fate better than initial expression alone.</li>
  </ul>
</section>

<section id="method">
  <h2>Analysis model and readouts</h2>
  <p class="lead">All three studies use the final one-stage Haruka model: a contrastive intrinsic decomposition combined with a shared spatial decoder. Control cells are decoded from <em>D</em><sub>niche</sub>(z,0), condition cells from <em>D</em><sub>niche</sub>(z,s), and salient spatial response is the counterfactual difference <em>D</em><sub>niche</sub>(z,s) − <em>D</em><sub>niche</sub>(z,0). There is no dual background/salient decoder and no post-fit residualization.</p>
  <div class="grid">
    <div class="card wide"><h3>Shared configuration</h3><p>z = 32, s = 32, hidden width 256, one encoder layer, two decoder layers, 150 epochs, spatial reconstruction weight 100, z–s independence penalty λ = 25, and decoder-only batch injection.</p></div>
    <div class="card wide"><h3>Spatial target</h3><p>Annotated neighborhood cell-type composition plus the cell-type-specific mean-neighbor molecular embedding. CellCharter aggregates symmetrically for three hops and never crosses a slice/core boundary.</p></div>
    <div class="card"><h3>STARmap</h3><p>Continuous Normal reconstruction; slice_id injected only into the decoder. Composition summaries use the fine-grained <code>ct_sub</code> annotation.</p></div>
    <div class="card"><h3>CODEX</h3><p>Protein MSE reconstruction; donor_id decoder injection. Analysis is restricted to the prepared primary myeloid population.</p></div>
    <div class="card"><h3>Lung</h3><p>Poisson sequencing reconstruction; run injected into the decoder; deposited <code>final_CT</code> defines the neighborhood target.</p></div>
  </div>
</section>

<section id="starmap">
  <h2>STARmap Alzheimer’s disease</h2>
  <p class="lead">Two AD sections were analyzed jointly with controls during training and separately by replicate for trajectory inference. The principal evidence is cross-replicate consistency: s-CC tracks plaque-associated geography, glial subtypes redistribute along shared DPT, and early molecular programs associate with later inferred terminal states.</p>
  <div class="table-wrap">{dataframe_html(plaque_view[plaque_view.dataset == "STARmap"], ["representation","endpoint","within_group_spearman_rho","median_group_rho","groups_same_sign"], {"within_group_spearman_rho":"{:.3f}","median_group_rho":"{:.3f}"})}</div>
  <div class="figure-grid">
    {figure(figures, "starmap_overview", True)}
    {figure(figures, "starmap_plaque")}
    {figure(figures, "ad_pc1")}
    {figure(figures, "starmap_terminal", True)}
  </div>
  <h3>Fine-grained ct_sub composition</h3>
  <p>The stacked proportions use <code>ct_sub</code>, not <code>ct_top</code>. Shared-glia shifts are substantial in both disease sections (Cramér’s V 0.30 and 0.21). Within lineage, Micro is strongest (mean V = {micro_v:.3f}), Astro is moderate (mean V = {astro_v:.3f}), and Oligo is weak (mean V = {oligo_v:.3f}). The dominant replicated pattern is depletion of Micro1 and enrichment of Micro3 from early to late DPT.</p>
  <div class="table-wrap">{dataframe_html(star_comp, ["replicate","panel","n_cells","n_categories","cramers_v","early_late_js_divergence","chi2_p_value"], {"cramers_v":"{:.3f}","early_late_js_divergence":"{:.3f}","chi2_p_value":"{:.2e}"})}</div>
  <div class="figure-grid">{figure(figures, "starmap_ctsub")}{figure(figures, "starmap_ctsub_within")}</div>
  <h3>Molecular interpretation</h3>
  <p>Microglial plaque-proximal predictors include CD9, CST7, CTSB, and SERPINE2. Local enrichment yields four microglial terms at FDR &lt; 0.10, centered on amino-acid catabolism, cell adhesion, and platelet activation/aggregation. Astrocyte amyloid-clearance terms involving APOE, CLU, CRYAB, and TREM2 are biologically suggestive but do not pass the 10% FDR threshold.</p>
  <div class="table-wrap">{dataframe_html(top_star, ["feature","meta_spearman_rho","meta_q_value","same_direction_both_replicates"], {"meta_spearman_rho":"{:.3f}","meta_q_value":"{:.3g}"})}</div>
  <div class="table-wrap">{dataframe_html(sig_enrichment, ["lineage","fate","Term","Adjusted P-value","Genes"], {"Adjusted P-value":"{:.3g}"})}</div>
  <div class="figure-grid">{figure(figures, "starmap_predictors")}{figure(figures, "starmap_enrichment")}</div>
</section>

<section id="codex">
  <h2>CODEX Alzheimer’s disease</h2>
  <p class="lead">The CODEX analysis provides an orthogonal protein-level validation in four AD donors. Salient s-CC shows much stronger plaque-distance structure than background z-CC, and the inferred trajectory is accompanied by large published myeloid-state composition changes.</p>
  <div class="table-wrap">{dataframe_html(plaque_view[plaque_view.dataset == "CODEX"], ["representation","endpoint","within_group_spearman_rho","median_group_rho","groups_same_sign"], {"within_group_spearman_rho":"{:.3f}","median_group_rho":"{:.3f}"})}</div>
  <p>s-CC PC1 has sample-controlled Spearman ρ = 0.606 for dense Aβ, 0.536 for diffuse Aβ, and 0.487 for native any-plaque distance. Corresponding z-CC associations have smaller magnitudes (−0.263, −0.214, and −0.213). The direction itself is not meaningful because PC1 sign is arbitrary.</p>
  <div class="figure-grid">{figure(figures, "codex_overview", True)}{figure(figures, "codex_plaque")}{figure(figures, "codex_terminal")}</div>
  <h3>Myeloid-state trajectory</h3>
  <p>All four donors show strong redistribution of published myeloid Leiden states over shared s-CC DPT: Cramér’s V ranges from {codex_v.min():.3f} to {codex_v.max():.3f}, with early-versus-late Jensen–Shannon divergence from {codex_comp.early_late_js_divergence.min():.3f} to {codex_comp.early_late_js_divergence.max():.3f}.</p>
  <div class="table-wrap">{dataframe_html(codex_comp, ["replicate","n_cells","n_categories","cramers_v","early_late_js_divergence"], {"cramers_v":"{:.3f}","early_late_js_divergence":"{:.3f}"})}</div>
  <div class="figure-grid">{figure(figures, "codex_stack", True)}{figure(figures, "codex_predictors", True)}</div>
  <p>Early-state proteins associated with the inferred plaque-proximal terminal include TMEM119, CD11c, CD163, and CD11b; CD68 and HLA-DR associate with the alternative terminal. Effect sizes are modest but estimated over 42 donor × initial-state groups.</p>
  <div class="table-wrap">{dataframe_html(top_codex, ["feature","meta_spearman_rho","meta_q_value","terminal_association"], {"meta_spearman_rho":"{:.3f}","meta_q_value":"{:.3g}"})}</div>
</section>

<section id="lung">
  <h2>GSE250346 human lung fibrosis</h2>
  <p class="lead">The fit contains all 1,405,182 cells: 531,567 cells from 15 less-affected cores as reference and 873,615 cells from 20 more-affected cores as condition. Spatial graphs are core-bounded. Five paired donors yield six valid less-to-more comparisons for moscot.</p>
  <h3>Background anatomy versus salient fibrosis response</h3>
  <p>Decoded salient composition-shift magnitude correlates with pathology (ρ = 0.459, p = 0.0417); s-CC magnitude is borderline (ρ = 0.443, p = 0.0506), while z-CC magnitude is not related to pathology. z-CC better recovers deposited cellular niches (CNiche NMI 0.314 versus 0.244 for s-CC), which is consistent with background retaining tissue organization.</p>
  <div class="table-wrap">{dataframe_html(lung_pathology, ["representation","spearman_rho","pvalue","n_cores"], {"spearman_rho":"{:.3f}","pvalue":"{:.3g}"})}</div>
  <div class="table-wrap">{dataframe_html(lung_metrics[lung_metrics.label.isin(["cell_type","CNiche","TNiche","run","donor"])], ["representation","label","ari","nmi"], {"ari":"{:.3f}","nmi":"{:.3f}"})}</div>
  <div class="figure-grid">{figure(figures, "lung_umap", True)}{figure(figures, "lung_pathology")}{figure(figures, "lung_biology")}</div>
  <h3>Spatial response programs</h3>
  <p>s-CC response niches are enriched for B/plasma cells, neutrophil–subpleural fibroblast–mesothelial structure, airway epithelial states, transitional AT2, and SPP1+/interstitial macrophage states. Salient-organized genes include {", ".join(hotspot_summary["top_salient_organized_genes"][:10])}; background-organized genes include {", ".join(hotspot_summary["top_background_organized_genes"][:10])}.</p>
  <div class="figure-grid">{figure(figures, "lung_hotspot")}{figure(figures, "lung_program_spatial")}</div>
  <h3>Paired-donor alignment and initial-state fate</h3>
  <p>Full-rank fused-GW alignment uses physical geometry plus background z-CC; s-CC never defines alignment. Averaged over six comparisons, alignment changes nearest cross-arm agreement by {improvements.cell_type_agreement:+.3f} for exact cell type, {improvements.lineage_agreement:+.3f} for lineage, {improvements.CNiche_agreement:+.3f} for CNiche, and {improvements.TNiche_agreement:+.3f} for TNiche.</p>
  <div class="figure-grid">{figure(figures, "lung_alignment", True)}{figure(figures, "lung_fate_umap", True)}</div>
  <p>Among 482 evidence-qualified initial alveolar fibroblasts, background z-CC predicts high versus low activated-fibrotic-fate probability with leave-donor-out AUC 0.617 (permutation p = 0.0099), whereas initial expression reaches AUC 0.531 (p = 0.188). This supports a spatially primed initial state.</p>
  <div class="table-wrap">{dataframe_html(lung_fate, ["feature","n","leave_donor_out_auc","balanced_accuracy","permutation_pvalue"], {"leave_donor_out_auc":"{:.3f}","balanced_accuracy":"{:.3f}","permutation_pvalue":"{:.4f}"})}</div>
  <p>The strict initial primed signature contains COL1A1 (log2FC {de_sig.iloc[0].log2FC_primed_vs_stable:.3f}, q = {de_sig.iloc[0].padj_bh:.2e}) and POSTN (log2FC {de_sig.iloc[1].log2FC_primed_vs_stable:.3f}, q = {de_sig.iloc[1].padj_bh:.2e}), both positive in all six pairs. Local GO enrichment identifies extracellular structure/ECM organization.</p>
  <div class="table-wrap">{dataframe_html(de_sig, ["gene","log2FC_primed_vs_stable","median_pair_log2FC","fraction_pairs_positive","padj_bh"], {"log2FC_primed_vs_stable":"{:.3f}","median_pair_log2FC":"{:.3f}","fraction_pairs_positive":"{:.2f}","padj_bh":"{:.2e}"})}</div>
  <div class="table-wrap">{dataframe_html(go_sig, ["Term","Adjusted P-value","Genes"], {"Adjusted P-value":"{:.3g}"})}</div>
  <div class="figure-grid">{figure(figures, "lung_fate_prediction")}{figure(figures, "lung_de")}{figure(figures, "lung_go")}{figure(figures, "lung_salient_marginal")}</div>
</section>

<section id="limits">
  <h2>Interpretation and limitations</h2>
  <ul class="key-findings">
    <li><strong>DPT and CellRank are cross-sectional state orderings.</strong> They support transition potential and terminal-state association, not direct lineage conversion. In particular, the shared STARmap trajectory must not be interpreted as oligodendrocytes converting into microglia.</li>
    <li><strong>PC1 sign is arbitrary.</strong> Compare association magnitude and consistency across samples rather than positive versus negative signs.</li>
    <li><strong>Composition-target supervision is not condition-label supervision.</strong> Haruka uses observed local cell-type composition as a reconstruction target, but no plaque, pathology, terminal-fate, or disease-class label supervises the latent decomposition.</li>
    <li><strong>Donor structure remains in s-CC.</strong> CellCharter can amplify real donor/core-specific spatial organization. Decoder-only batch injection limits technical leakage but cannot distinguish every biological donor difference from unmeasured technical structure.</li>
    <li><strong>Lung moscot evidence is paired and observational.</strong> Only genuine donor pairs are aligned. Salient-marginal transport is a sensitivity analysis, while the primary alignment is defined by physical coordinates and z-CC.</li>
    <li><strong>Enrichment depends on the measured panel.</strong> STARmap and especially CODEX have limited molecular panels; absent pathways cannot be interpreted as negative biological evidence.</li>
  </ul>
</section>

<section id="data">
  <h2>Result tables</h2>
  <p class="lead">The report folder is portable: figures and selected result tables are copied beside this HTML. Full-resolution source results remain in the original project analysis directories.</p>
  {download_list()}
  <p class="note">Missing optional inputs during build: {html.escape("; ".join(missing)) if missing else "none"}.</p>
</section>
</main>
<footer>Haruka integrated spatial contrastive report · generated from local saved outputs · 31 August 2026</footer>
</body>
</html>
"""
    (out / "index.html").write_text(report, encoding="utf-8")
    manifest = {
        "report": str(out / "index.html"),
        "figures": {key: value[0] for key, value in figures.items()},
        "tables": [name for name, source in TABLES.items() if source.exists()],
        "missing": missing,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out / "index.html")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    build_report(args.out)


if __name__ == "__main__":
    main()
