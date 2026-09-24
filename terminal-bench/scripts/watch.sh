#!/usr/bin/env bash
# Node-side babysitter (tmux session `watch` on node08). 60s loop:
#  - vLLM health -> restart serve (mode from .serve-thinking) if BOTH dead
#  - /tmp janitor
#  - hourly rsync of jobs/ to node05 capacity tier
# Log: terminal-bench/watch.log (one line per tick, alarms marked ALARM).
set -u
REPO="${TARL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"   # self-locating: terminal-bench/ (real path)
cd "$REPO" || exit 1
. scripts/env.sh >/dev/null 2>&1
LOG="$REPO/watch.log"
tick=0
while true; do
    ts=$(date "+%m-%d %H:%M:%S")
    free=$(df --output=avail -BG /tmp | tail -1 | tr -d " ")
    if curl -sf localhost:8000/v1/models >/dev/null 2>&1; then
        serve=ok
    else
        serve=DOWN
        if ! tmux has-session -t serve 2>/dev/null; then
            mode=$(cat "$REPO/.serve-thinking" 2>/dev/null || echo 1)
            # .serve-model = merged-policy dir written by tarl.serving.harbor_cmd.restart_serve (empty = base)
            pol=$(head -1 "$REPO/.serve-model" 2>/dev/null | tr -d '[:space:]')
            extra=""; [ -n "$pol" ] && [ -d "$pol" ] && extra="TARL_SERVE_MODEL=$pol TARL_SERVE_QUANT=fp8"
            tmux new-session -d -s serve \
                "cd $REPO && TARL_THINKING=$mode $extra bash scripts/01_serve.sh 2>&1 | tee $REPO/serve.log"
            echo "$ts ALARM serve dead -> relaunched (thinking=$mode model=${pol:-base})" >> "$LOG"
        fi
    fi
    bash scripts/_reaper.sh >> "$LOG" 2>&1 || true
    bash scripts/_janitor.sh || true
    TARL_SIF_CACHE="${TARL_ET_SIF:-/var/tmp/tarl/et-sif}" bash scripts/_janitor.sh || true   # Endless Terminals images
    if [ $((tick % 60)) -eq 0 ]; then
        bash scripts/07_sync_node05.sh >> "$LOG" 2>&1 || echo "$ts ALARM node05 sync failed" >> "$LOG"
    fi
    echo "$ts serve=$serve tmp_free=$free jobs=$(ls jobs 2>/dev/null | wc -l)" >> "$LOG"
    tick=$((tick + 1))
    sleep 60
done
