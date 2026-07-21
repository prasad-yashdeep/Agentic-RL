"""Single source of truth for storage locations (tiers: docs/01-infra.md).

Durability (NFS home) > capacity (node05 /local, wipeable) > hot (/tmp, janitor).
"""

import os
from pathlib import Path

HOME = Path.home()
REPO = HOME / "terminal-agent-rl"

# durability tier (NFS): anything irreplaceable
JOBS = REPO / "jobs"
CHECKPOINTS = REPO / "checkpoints"
HF_CACHE = Path(os.environ.get("HF_HOME", str(HOME / "hf-cache")))

# node-local hot tier on node08 (16 GB /tmp, evicted by scripts/_janitor.sh)
SIF_CACHE = Path(os.environ.get("TARL_SIF_CACHE", "/tmp/sif-cache"))
TB2_LOCAL = Path(os.environ.get("TARL_TB_PATH", "/tmp/tb2"))

# capacity tier: node05 /local — big, wipeable on reprovision; reached via
# ssh/rsync from node08 (scripts/07_sync_node05.sh), never mounted
NODE05_HOST = "node05"
NODE05_ROOT = "/local/yashp/tarl"
