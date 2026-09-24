#!/usr/bin/env bash
# One-shot health report for node08 after a reboot/reprovision. Run FROM the headnode:
#   ssh node08 'bash /home/yashp/Agentic-RL/terminal-bench/scripts/_node08_health.sh'
# Prints what the experiment sequence depends on; never changes anything.
set -u
REPO="${TARL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"   # self-locating: terminal-bench/ (real path)
cd "$REPO" 2>/dev/null || { echo "repo missing on this host"; exit 1; }
ARL_ROOT="$(cd "$REPO/.." && pwd -P)"; VENV="${ARL_VENV:-$ARL_ROOT/.venv}"   # shared monorepo venv
. scripts/env.sh >/dev/null 2>&1 || true

echo "== host =="; hostname; uptime; cat /proc/sys/kernel/unprivileged_userns_clone 2>/dev/null | sed 's/^/userns_clone=/'
echo "== gpus =="; nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu,utilization.gpu,driver_version --format=csv 2>&1
echo "== gpu procs =="; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 | tail -n +2
echo "== disks =="; df -h / /tmp /var/tmp 2>&1 | tail -n +1
echo "== caches =="
for d in /var/tmp/tarl /var/tmp/tarl/sif-cache /var/tmp/tarl/apptainer-cache /tmp/sif-cache /tmp/apptainer-cache /tmp/tb2; do
    if [ -e "$d" ]; then printf '%-32s %s\n' "$d" "$(du -sh "$d" 2>/dev/null | cut -f1) $(readlink "$d" 2>/dev/null)"; else printf '%-32s MISSING\n' "$d"; fi
done
[ -d /tmp/tb2 ] && echo "tb2 tasks: $(ls /tmp/tb2 | wc -l)"
ls /var/tmp/tarl/sif-cache/*.sif 2>/dev/null | wc -l | sed 's/^/sif files: /'
ls -d /var/tmp/tarl/sif-cache/*.sandbox 2>/dev/null | wc -l | sed 's/^/sandboxes: /'
echo "== toolchain =="
command -v apptainer && apptainer --version 2>&1
command -v harbor && harbor --version 2>&1 | head -1
command -v vllm && echo "vllm on PATH"
"$VENV/bin/python" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count())" 2>&1
echo "== hf cache =="; ls "$HOME/hf-cache/hub" 2>/dev/null | head; du -sh "$HOME/hf-cache" 2>/dev/null
echo "== tmux =="; tmux ls 2>&1
echo "== serve =="; (curl -sf --max-time 3 localhost:8000/v1/models >/dev/null && echo "UP") || echo "DOWN"
echo "== stray processes (vllm/harbor/apptainer/python) =="; pgrep -a -f 'vllm|harbor|apptainer|08_grpo|06_grpo|watch.sh' 2>/dev/null | cut -c1-140 | head
echo "== exporters =="; for p in 9100 9256 9400; do (curl -sf --max-time 2 "localhost:$p/metrics" >/dev/null && echo "port $p up") || echo "port $p down"; done
echo "== internet =="; curl -s --max-time 8 -o /dev/null -w "huggingface.co %{http_code}\n" https://huggingface.co || echo "no internet"
