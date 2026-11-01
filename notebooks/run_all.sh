#!/usr/bin/env bash
# Execute every reproduction notebook in place. Default: end to end (Hugging Face download -> fit -> analyses -> figure).
# HARUKA_FIGURE_ONLY=1 notebooks/run_all.sh  redraws the figures from saved analysis outputs.
cd "$(dirname "$0")"; mkdir -p logs
NBC=${NBCONVERT:-jupyter}; KERNEL=${HARUKA_KERNEL:-haruka2}
for nb in 0*.ipynb; do
  "$NBC" nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name="$KERNEL" \
    --ExecutePreprocessor.timeout=-1 "$nb" > "logs/${nb%.ipynb}.log" 2>&1 &
done
wait; grep -L "Error\|Traceback" logs/*.log | sed 's/^/ok: /'; grep -l "Error\|Traceback" logs/*.log | sed 's/^/FAILED: /'
