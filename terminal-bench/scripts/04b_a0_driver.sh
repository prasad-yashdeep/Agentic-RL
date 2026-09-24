#!/usr/bin/env bash
# A0 driver (tmux `eval`): thinking-ON k=5 job, then thinking-OFF k=1 screen.
# Flips the serve template between arms via .serve-thinking + serve restart;
# watch.sh reads the same file, so an unattended crash-restart keeps the mode.
set -uo pipefail
REPO="${TARL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"   # self-locating: terminal-bench/ (real path)
cd "$REPO" || exit 1
. scripts/env.sh >/dev/null 2>&1

restart_serve() {  # $1 = 1|0 (thinking on/off)
    echo "$1" > .serve-thinking
    tmux kill-session -t serve 2>/dev/null || true
    sleep 3
    tmux new-session -d -s serve \
        "cd $REPO && TARL_THINKING=$1 bash scripts/01_serve.sh 2>&1 | tee $REPO/serve.log"
    for _ in $(seq 1 120); do
        curl -sf localhost:8000/v1/models >/dev/null 2>&1 && return 0
        sleep 5
    done
    echo "A0_DRIVER_FATAL: serve did not come up (thinking=$1)"
    return 1
}

echo "A0_ARM_ON_START $(date "+%F %T")"
restart_serve 1 || exit 1
TARL_K=5 bash scripts/04_baseline_eval.sh 2>&1 | tee a0-on.log
echo "A0_ARM_ON_DONE $(date "+%F %T")"
bash scripts/07_sync_node05.sh || true

echo "A0_ARM_OFF_START $(date "+%F %T")"
restart_serve 0 || exit 1
TARL_K=1 bash scripts/04_baseline_eval.sh 2>&1 | tee a0-off.log
echo "A0_ARM_OFF_DONE $(date "+%F %T")"
bash scripts/07_sync_node05.sh || true

# leave the server in thinking-ON mode for subsequent work
restart_serve 1 || true
echo "A0_DRIVER_DONE $(date "+%F %T")"
