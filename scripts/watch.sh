#!/usr/bin/env bash
# Node-side babysitter (tmux session `watch` on node08). 60s loop:
#  - vLLM health -> restart serve (mode from .serve-thinking) if BOTH dead
#  - /tmp janitor
#  - hourly rsync of jobs/ to node05 capacity tier
# Log: ~/terminal-agent-rl/watch.log (one line per tick, alarms marked ALARM).
set -u
cd "$HOME/terminal-agent-rl"
. scripts/env.sh >/dev/null 2>&1
LOG="$HOME/terminal-agent-rl/watch.log"
tick=0
while true; do
    ts=$(date "+%m-%d %H:%M:%S")
    free=$(df --output=avail -BG /tmp | tail -1 | tr -d " ")
    if curl -sf localhost:8000/v1/models >/dev/null 2>&1; then
        serve=ok
    else
        serve=DOWN
        if ! tmux has-session -t serve 2>/dev/null; then
            mode=$(cat "$HOME/terminal-agent-rl/.serve-thinking" 2>/dev/null || echo 1)
            tmux new-session -d -s serve \
                "cd $HOME/terminal-agent-rl && TARL_THINKING=$mode bash scripts/01_serve.sh 2>&1 | tee $HOME/terminal-agent-rl/serve.log"
            echo "$ts ALARM serve dead -> relaunched (thinking=$mode)" >> "$LOG"
        fi
    fi
    bash scripts/_reaper.sh >> "$LOG" 2>&1 || true
    bash scripts/_janitor.sh || true
    if [ $((tick % 60)) -eq 0 ]; then
        bash scripts/07_sync_node05.sh >> "$LOG" 2>&1 || echo "$ts ALARM node05 sync failed" >> "$LOG"
    fi
    echo "$ts serve=$serve tmp_free=$free jobs=$(ls jobs 2>/dev/null | wc -l)" >> "$LOG"
    tick=$((tick + 1))
    sleep 60
done
