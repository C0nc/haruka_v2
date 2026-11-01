#!/usr/bin/env bash
# Install the Haruka model: copy the patched ContrastiveVI module files over the installed scvi-tools 1.4.0.post1 copy.
set -euo pipefail
PY=${HARUKA_PY:-python}
DST=$("$PY" -c "import os, scvi; print(os.path.join(os.path.dirname(scvi.__file__), 'external', 'contrastivevi'))" 2>/dev/null | tail -1)
cp "$(dirname "${BASH_SOURCE[0]}")"/contrastivevi_patch/*.py "$DST"/
echo "patched $DST"
