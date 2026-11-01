#!/usr/bin/env bash
# Execute every reproduction notebook in place (figure stage by default; export RUN_DATA/RUN_FIT/RUN_ANALYSIS=1 for more).
cd "$(dirname "$0")"; mkdir -p logs
NBC=${NBCONVERT:-/home/yancui/.conda/envs/plot_pub/bin/jupyter}
for nb in 0*.ipynb; do
  "$NBC" nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=haruka2 \
    --ExecutePreprocessor.timeout=-1 "$nb" > "logs/${nb%.ipynb}.log" 2>&1 &
done
wait; grep -L "Error\|Traceback" logs/*.log | sed 's/^/ok: /'; grep -l "Error\|Traceback" logs/*.log | sed 's/^/FAILED: /'
