#!/usr/bin/env bash
# A/B of vLLM serving layouts on the real pipeline: the same pool probe (tasks, seeds, images, and sandboxes and grading
# on this host) once per layout, in the order given. Run on node04 in tmux, with the pool images already in SIF_DIR.
#   bash scripts/ab_serving.sh R1=http://node08:8001@8,http://node08:8002@8,http://node04:8003@8 R2=http://node08:8001@8,...
# Per layout: <name>.json/.log (rolling_probe; "timing" holds gate_s and episodes_s) and, in metrics.log, every
# server's vLLM prompt/generation token counters before and after. mon.log samples this host every 10 s:
# epoch, MemAvailable kB, 1-min load, running sandbox instances.
set -euo pipefail
case "$(hostname)" in bcm11-headnode*) echo "refusing on the headnode"; exit 1;; esac
SWE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"; ARL="$(cd "$SWE/.." && pwd -P)"
OUT="${AB_OUT:-$SWE/experiments/ab-serving}"
SIF_DIR="${AB_SIF_DIR:-/var/tmp/swe/sif-dev}"
TASKS="${AB_TASKS:-configs/pool_minicoder_v2.txt}"
mkdir -p "$OUT"
cd "$SWE"
( while true; do
    echo "$(date +%s) $(awk '/MemAvailable/{print $2}' /proc/meminfo) $(cut -d' ' -f1 /proc/loadavg) $(pgrep -u "$USER" -fc 'Apptainer instance' || true)"
    sleep 10
  done ) >> "$OUT/mon.log" &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT
snap() {  # name phase urls...
  local name=$1 phase=$2; shift 2
  for u in "$@"; do
    curl -s --max-time 10 "$u/metrics" | grep -E '^vllm:(generation|prompt)_tokens_total' | sed "s|^|$(date +%s) $name $phase $u |" || true
  done >> "$OUT/metrics.log"
}
for spec in "$@"; do
  name=${spec%%=*}; urls=(); slots=()
  IFS=, read -ra parts <<< "${spec#*=}"
  for p in "${parts[@]}"; do urls+=("${p%@*}"); slots+=("${p##*@}"); done
  echo "$(date +%s) start $name ${spec#*=}" >> "$OUT/mon.log"
  snap "$name" before "${urls[@]}"
  SARL_SANDBOX_VMEM_GB="${SARL_SANDBOX_VMEM_GB:-4}" HF_HOME="${HF_HOME:-$HOME/hf-cache}" HF_HUB_OFFLINE=1 \
    "$ARL/.venv/bin/python" scripts/rolling_probe.py "$TASKS" --split dev --sif-dir "$SIF_DIR" --keep-all \
    --model-key gemma4-e4b --arm gemma=policy --servers "${urls[@]}" --per-server "${slots[@]}" \
    --episodes 8 --batch 11 --max-turns 50 --max-context 24576 --wall-clock 1200 \
    --out "$OUT/$name.json" 2>&1 | tee "$OUT/$name.log"
  snap "$name" after "${urls[@]}"
  echo "$(date +%s) end $name" >> "$OUT/mon.log"
done
