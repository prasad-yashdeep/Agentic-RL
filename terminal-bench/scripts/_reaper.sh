#!/usr/bin/env bash
# Reap orphaned apptainer runtime parents + their FUSE daemons. Trial teardown
# (SIGTERM) often fails to kill them (fuse wedge); they reparent to PID 1, pin
# their sandbox dirs as "in use" (janitor skips), and leak RAM/mounts.
# Orphan = PPID 1 AND alive > 5 min (an active trial keeps its parent chain).
set -u
reap() {  # $1 = pgrep -f pattern
    for pid in $(pgrep -P 1 -f "$1" 2>/dev/null); do
        et=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d " ")
        [ -n "$et" ] && [ "$et" -gt 300 ] && kill -9 "$pid" 2>/dev/null && echo "reaped $pid ($1)"
    done
}
reap "runtime paren[t]"
reap "squashfuse_l[l]"
reap "fuse-overlayf[s]"
reap "faked-sys[v]"
exit 0
