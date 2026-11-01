#!/bin/bash
# MapMyCells (cell_type_mapper) flat correlation mapping, 100 bootstraps x 90% genes, as in Yuan et al. Methods.
set -euo pipefail
B=/home/yancui/haruka_v2/bench_sticr_birthdate
E=/home/yancui/.conda/envs/sticr_ctm/bin/python
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
mkdir -p $B/work/ctm_tmp
cd $B/work
$E -I -m cell_type_mapper.cli.precompute_stats_scrattch \
  --h5ad_path $B/work/ref_500.h5ad --hierarchy '["cls","cluster"]' \
  --output_path $B/work/ref_500_stats.h5 --normalization raw --n_processors 8 --tmp_dir $B/work/ctm_tmp --clobber True
$E -I -m cell_type_mapper.cli.from_specified_markers \
  --query_path $B/work/query_fb.h5ad \
  --precomputed_stats.path $B/work/ref_500_stats.h5 \
  --query_markers.serialized_lookup $B/work/markers_all500.json \
  --type_assignment.normalization raw --type_assignment.bootstrap_iteration 100 \
  --type_assignment.bootstrap_factor 0.9 --type_assignment.n_processors 8 \
  --type_assignment.rng_seed 11 --flatten True --map_to_ensembl False \
  --tmp_dir $B/work/ctm_tmp --extended_result_path $B/work/ctm_result.json \
  --csv_result_path $B/work/ctm_result.csv
echo MAPPING_DONE
