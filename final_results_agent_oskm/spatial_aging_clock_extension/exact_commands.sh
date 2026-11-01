#!/usr/bin/env bash
# Run from /home/yancui/haruka_v2/final_results_agent_oskm
# 1. Exact official-cache equivalence check (160,000 cells)
conda run --no-capture-output -n haruka2 python spatial_aging_clock_extension/scripts/run_official_clock_parallel.py --data /home/yancui/haruka_v2/bench_oskm_reprogramming/data/oskm_old_full_684253.h5ad --clock-repo /home/yancui/haruka_v2/bench_oskm_reprogramming/external/SpatialAgingClock --out /home/yancui/haruka_v2/final_results_agent_oskm/spatial_aging_clock_extension/validation_160k --workers 32 --restrict-cell-ids /home/yancui/haruka_v2/final_results_agent_oskm/clock/source_spatial_clock_cell_scores_160k.csv.gz --expected-cache /home/yancui/haruka_v2/final_results_agent_oskm/clock/source_spatial_clock_cell_scores_160k.csv.gz

# 2. Published frozen clock on all 684,253 old-control + old-OSKM cells
conda run --no-capture-output -n haruka2 python spatial_aging_clock_extension/scripts/run_official_clock_parallel.py --data /home/yancui/haruka_v2/bench_oskm_reprogramming/data/oskm_old_full_684253.h5ad --clock-repo /home/yancui/haruka_v2/bench_oskm_reprogramming/external/SpatialAgingClock --out /home/yancui/haruka_v2/final_results_agent_oskm/spatial_aging_clock_extension/clock_full --workers 32

# 3. Within-OSKM-slice s-CC diffusion pseudotime
conda run --no-capture-output -n haruka2 python spatial_aging_clock_extension/scripts/run_within_sample_dpt.py --cc-dir /home/yancui/haruka_v2/results_real_compgated_nb_full_oskm/oskm_old_full_684253/haruka_z_s_cellcharter_cc3_condition_only --umap-table /home/yancui/haruka_v2/final_results_agent_oskm/tables/condition_only_fixed_seed_umap_coordinates.csv.gz --out /home/yancui/haruka_v2/final_results_agent_oskm/spatial_aging_clock_extension/dpt --workers 4

# 4. Leakage-aware associations, bootstrap intervals, and figures
env OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 conda run --no-capture-output -n haruka2 python spatial_aging_clock_extension/scripts/analyze_clock_haruka_extension.py --clock /home/yancui/haruka_v2/final_results_agent_oskm/spatial_aging_clock_extension/clock_full/spatial_clock_predictions.csv.gz --data /home/yancui/haruka_v2/bench_oskm_reprogramming/data/oskm_old_full_684253.h5ad --cc-dir /home/yancui/haruka_v2/results_real_compgated_nb_full_oskm/oskm_old_full_684253/haruka_z_s_cellcharter_cc3_condition_only --umap /home/yancui/haruka_v2/final_results_agent_oskm/tables/condition_only_fixed_seed_umap_coordinates.csv.gz --dpt /home/yancui/haruka_v2/final_results_agent_oskm/spatial_aging_clock_extension/dpt/within_sample_scc_dpt.csv.gz --out /home/yancui/haruka_v2/final_results_agent_oskm/spatial_aging_clock_extension

# 5. Report and hashes
conda run -n haruka2 python spatial_aging_clock_extension/scripts/finalize_report.py
