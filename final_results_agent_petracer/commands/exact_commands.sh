#!/usr/bin/env bash
set -euo pipefail

# Skill and required references read before acting.
sed -n '1,260p' /home/yancui/.codex/plugins/cache/personal/haruka-spatial/0.1.0/skills/haruka-spatial/SKILL.md
sed -n '1,320p' /home/yancui/.codex/plugins/cache/personal/haruka-spatial/0.1.0/skills/haruka-spatial/references/model-and-inputs.md
sed -n '1,320p' /home/yancui/.codex/plugins/cache/personal/haruka-spatial/0.1.0/skills/haruka-spatial/references/post-analysis.md
sed -n '1,320p' /home/yancui/.codex/plugins/cache/personal/haruka-spatial/0.1.0/skills/haruka-spatial/references/petracer.md

# Source and repository inventory.
find /home/yancui/Jill/fourteen/petracer_data -maxdepth 3 -type f -printf '%p\t%s bytes\n' | sort
find /home/yancui/haruka_v2/bench_petracer_lineage -maxdepth 3 -type f -printf '%p\t%s bytes\n' | sort
rg -n -S 'paired|simulation|gt_|condition|control|section|sample|target|lineage|character|response|responder|batch' /home/yancui/haruka_v2/bench_petracer_lineage/*.py /home/yancui/haruka_v2/bench_real_discovery/score_petracer_lineage_distance_cc3.py /home/yancui/haruka_v2/bench_real_discovery/analyze_domain_celltype_enrichment.py

# Rebuild all audit tables, figure, design JSON, logs, and provenance hashes.
conda run --no-capture-output -n haruka2 python -u /home/yancui/haruka_v2/final_results_agent_petracer/audit_petracer_sources.py

# Deliberately NOT RUN: Haruka training, CC construction, responder scoring,
# enrichment, held-out evaluation, or leakage metrics. The prerequisite real
# control/condition contrast is absent (status=blocked_no_real_control).
