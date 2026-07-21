#!/usr/bin/env bash
# Keep node08 /tmp healthy during image-heavy runs. Evicts idle container-image
# artifacts, oldest first. NEVER touches /tmp/tb2 (dataset clone).
# In-use guards: (a) artifact mtime must be idle (shim touches sandbox on use),
# (b) no live process may reference its name (apptainer cmdlines embed paths).
set -u
MIN_FREE_GB="${TARL_TMP_MIN_FREE_GB:-5}"
IDLE_MIN="${TARL_EVICT_IDLE_MIN:-8}"
SIF_CACHE="${TARL_SIF_CACHE:-/tmp/sif-cache}"

free_gb() { df --output=avail -BG "$SIF_CACHE" 2>/dev/null | tail -1 | tr -dc 0-9; }
in_use() { pgrep -f "$(basename "$1")" >/dev/null 2>&1; }

[ "$(free_gb)" -ge "$MIN_FREE_GB" ] && exit 0
# emergency: critically low -> age no longer protects, only liveness does
if [ "$(free_gb)" -lt 3 ]; then IDLE_MIN=0; fi

# pinned artifacts (manually warmed big images) are never evicted
pinned() { grep -qxF "$(basename "$1")" "$SIF_CACHE/.pinned" 2>/dev/null; }

# 1. idle sandboxes (cheap to recreate from sif)
for d in $(ls -dtr "$SIF_CACHE"/*.sandbox 2>/dev/null); do
    [ "$(free_gb)" -ge "$MIN_FREE_GB" ] && exit 0
    find "$d" -maxdepth 0 -mmin +"$IDLE_MIN" 2>/dev/null | grep -q . || continue
    pinned "$d" && continue
    in_use "$d" && continue
    rm -rf "$d" "$d.lock" 2>/dev/null
done
# 2. idle sifs (re-convertible from the registry, costs a re-pull)
for f in $(ls -tr "$SIF_CACHE"/*.sif 2>/dev/null); do
    [ "$(free_gb)" -ge "$MIN_FREE_GB" ] && exit 0
    find "$f" -maxdepth 0 -mmin +"$IDLE_MIN" 2>/dev/null | grep -q . || continue
    pinned "$f" && continue
    in_use "$f" && continue
    rm -f "$f" "$f.lock" 2>/dev/null
done
# 3. OCI blob cache (pure cache)
if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then
    rm -rf /tmp/apptainer-cache/blob 2>/dev/null; rm -rf "$(readlink -f /tmp/apptainer-cache)"/* 2>/dev/null
fi
exit 0
