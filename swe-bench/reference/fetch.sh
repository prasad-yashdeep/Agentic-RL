#!/usr/bin/env bash
# Fetch the Vizuara SWE-RL reference (abgoswam/swe_in_prod_vizuara_01) at the pinned commit into _upstream/
# (gitignored: the upstream repo has no license, so we fetch and verify it but never commit it), then build the
# cluster copy of the notebook (only the two `!pip` lines of cell 1 are removed; see make_cluster_nb.py).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SHA=8cddfb60c93cc6b2c7d9dc047939084858f30f76
BASE="https://raw.githubusercontent.com/abgoswam/swe_in_prod_vizuara_01/$SHA"
mkdir -p "$HERE/_upstream"
for f in README.md utils.py swe_grpo_one_step.py swe_grpo_one_step.ipynb swe_agent_env_interface.png; do
    [ -f "$HERE/_upstream/$f" ] || curl -fsSL "$BASE/$f" -o "$HERE/_upstream/$f"
done
(cd "$HERE/_upstream" && sha256sum -c ../UPSTREAM.sha256)
python3 "$HERE/make_cluster_nb.py"
echo "UPSTREAM_OK $SHA"
