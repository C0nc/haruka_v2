#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(SeuratObject))
suppressPackageStartupMessages(library(Matrix))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: export_gse250346_unaffected_vs_fibrosis.R INPUT_RDS OUTPUT_DIRECTORY")
}

input_rds <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

message("Reading corrected GEO object: ", input_rds)
object <- readRDS(input_rds)
metadata <- object[[]]

required <- c(
  "sample", "patient", "sample_type", "sample_affect", "percent_pathology",
  "tma", "run", "final_CT", "final_lineage", "CNiche", "TNiche",
  "x_centroid", "y_centroid"
)
missing <- setdiff(required, colnames(metadata))
if (length(missing) > 0) {
  stop("Missing expected metadata: ", paste(missing, collapse = ", "))
}

keep <- metadata$sample_affect %in% c(
  "Unaffected", "Less Affected", "More Affected"
)
keep <- keep & !is.na(metadata$final_CT)
message(
  "Keeping ", sum(keep), " cells from ",
  length(unique(metadata$sample[keep])), " cores"
)

counts <- LayerData(object, assay = "RNA", layer = "counts")
counts <- counts[, keep, drop = FALSE]
if (!inherits(counts, "dgCMatrix")) {
  counts <- as(counts, "dgCMatrix")
}

selected <- metadata[keep, required, drop = FALSE]
selected$obs_name <- colnames(counts)
selected$condition <- ifelse(
  selected$sample_affect == "Unaffected", "control", "fibrosis"
)
selected$severity_group <- selected$sample_affect
selected$slice_id <- selected$sample
selected$batch <- selected$run
selected$donor <- selected$patient
selected$cell_type <- as.character(selected$final_CT)
selected$lineage <- as.character(selected$final_lineage)

selected <- selected[, c(
  "obs_name", "condition", "severity_group", "sample_affect", "sample_type",
  "slice_id", "sample", "donor", "patient", "batch", "run", "tma",
  "percent_pathology", "cell_type", "final_CT", "lineage", "final_lineage",
  "CNiche", "TNiche", "x_centroid", "y_centroid"
)]

obs_connection <- gzfile(file.path(output_dir, "obs.csv.gz"), open = "wt")
write.csv(selected, obs_connection, row.names = FALSE, quote = TRUE)
close(obs_connection)
var_connection <- gzfile(file.path(output_dir, "var_names.txt.gz"), open = "wt")
writeLines(rownames(counts), var_connection)
close(var_connection)
writeLines(
  paste(nrow(counts), ncol(counts), length(counts@x), sep = "\t"),
  file.path(output_dir, "counts_dims.tsv")
)

writeBin(counts@i, file.path(output_dir, "counts_i.int32.bin"), size = 4L)
writeBin(counts@p, file.path(output_dir, "counts_p.int32.bin"), size = 4L)
writeBin(counts@x, file.path(output_dir, "counts_x.float32.bin"), size = 4L)

sample_summary <- as.data.frame(
  table(selected$slice_id, selected$sample_affect)
)
colnames(sample_summary) <- c("slice_id", "sample_affect", "n_cells")
sample_summary <- sample_summary[sample_summary$n_cells > 0, ]
write.csv(
  sample_summary, file.path(output_dir, "sample_summary.csv"),
  row.names = FALSE
)

design_summary <- as.data.frame(
  table(selected$condition, selected$sample_affect, selected$run)
)
colnames(design_summary) <- c(
  "condition", "sample_affect", "run", "n_cells"
)
design_summary <- design_summary[design_summary$n_cells > 0, ]
write.csv(
  design_summary, file.path(output_dir, "design_summary.csv"),
  row.names = FALSE
)

message(
  "Exported ", nrow(counts), " genes x ", ncol(counts), " cells with ",
  length(counts@x), " nonzero counts"
)
