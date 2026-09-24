#!/usr/bin/env bash
# S0: download the pinned model and dataset revisions into the HF cache (network + disk only; headnode is fine).
# Pins mirror sarl.paths (MODEL_REVISION, DATASETS). Refuses if /home has < 20 GB free (the NFS disk is shared).
set -euo pipefail
SWE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"; ARL="$(cd "$SWE/.." && pwd -P)"
export HF_HOME="${HF_HOME:-$HOME/hf-cache}"
free_gb=$(df --output=avail -BG /home | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 20 ] || { echo "only ${free_gb} GB free on /home; refusing"; exit 1; }
"$ARL/.venv/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
from sarl.paths import DATASETS, MODEL_ID, MODEL_REVISION
print("model", snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=["*.json", "*.txt", "*.safetensors"]))
for name, (repo, rev, split) in DATASETS.items():
    print("dataset", name, snapshot_download(repo, repo_type="dataset", revision=rev, allow_patterns=[f"data/{split}-*", "*.md"]))
print("ASSETS_OK")
PY
