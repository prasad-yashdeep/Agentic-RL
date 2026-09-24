#!/usr/bin/env bash
# Pull the pool's SWE-bench images to node-local SIFs (default /tmp/swe-sif) with node-local apptainer caches.
# Usage (on node08): bash 10_pull_sifs.sh [pool file]   — idempotent; prints a size table.
set -uo pipefail
case "$(hostname)" in bcm11-headnode*) echo "refusing on the headnode"; exit 1;; esac
SWE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"; ARL="$(cd "$SWE/.." && pwd -P)"
POOL="${1:-$SWE/configs/pool_dev.txt}"
exec "$ARL/.venv/bin/python" - "$POOL" <<'PY'
import sys, time
from sarl.env.apptainer import ensure_sif, sif_for
from sarl.tasks import read_pool
tasks = read_pool(sys.argv[1])
for t in tasks:
    t0 = time.time()
    try:
        p = ensure_sif(t.docker_ref, sif_for(t))
        print(f"ok    {t.instance_id:40} {p.stat().st_size/1e9:5.2f} GB  {time.time()-t0:5.0f}s", flush=True)
    except Exception as e:
        print(f"FAIL  {t.instance_id:40} {e}", flush=True)
PY
