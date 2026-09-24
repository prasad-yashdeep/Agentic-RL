#!/usr/bin/env bash
# scripts/09_experiments.sh — the pre-registered experiment sequence, unattended, ON node08 (tmux `exp`).
#
#   make -C /home/yashp/Agentic-RL/terminal-bench exp     (runs tools/doctor.sh first, then tmux `exp` on node08)
#   tail -f /home/yashp/Agentic-RL/terminal-bench/runs/driver/driver.log
#
# Phases run in order; each is idempotent and skipped once runs/driver/<phase>.done exists
# (delete the marker to redo a phase). TARL_PHASES="a b c" runs a subset.
#   health     node08 sanity, caches/symlinks, /tmp/tb2 restore from ~/datasets/terminal-bench-2-1, exporters
#   serve      base FP8 serve (thinking ON) in tmux `serve` + babysitter in tmux `watch`
#   smoke      one live fix-git episode through `tarl run --live`
#   gpu_cells  notebook §11 on GPU1: real QLoRA step, trainer-vs-vLLM logprob align check, live episode
#   a0_resume  finish the 445-trial A0 job on TB 2.1 (harbor jobs resume; infra errors re-run)       ~15 h
#   et_prep    Endless Terminals: split, convert, leakage guard vs TB2, build .sif images              ~30 min
#   a0_etdev   base model on the ET dev pool, k=1 (cold-start p on the training distribution)         ~1 h
#   b1         GRPO verifier-only (lambda=0) on the ET train pool                                      ~10 h
#   b1_eval    B1 policy on ET dev (k=1) then TB 2.1 (k=TARL_EVAL_K); merged weights freed after       ~1 h + ~15 h
#   b2         GRPO judge-shaped (lambda=0.5, LIVE grok judge, cost cap) — REQUIRES experiments/.allow-judge-spend
#   b2_eval    as b1_eval
# Knobs (env): TARL_ET_TRAIN=160 TARL_ET_DEV=60 TARL_STEPS=50 TARL_G=8 TARL_TASKS_PER_STEP=4 TARL_EVAL_K=5
#              TARL_N_CONCURRENT=4 TARL_JUDGE_CAP_USD=40 TARL_LR=1e-6
#              TARL_A0_RERUN_ERRORS=CancelledError,RuntimeError,EnvironmentStartTimeoutError (A0 infra errors re-run on resume)
set -uo pipefail
REPO="${TARL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"   # self-locating: terminal-bench/ (real path)
cd "$REPO" || exit 1
. scripts/env.sh >/dev/null 2>&1
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy-local}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
ARL_ROOT="$(cd "$REPO/.." && pwd -P)"; VENV="${ARL_VENV:-$ARL_ROOT/.venv}"   # shared monorepo venv
PY="$VENV/bin/python"; TARL="$VENV/bin/tarl"
D="$REPO/runs/driver"; mkdir -p "$D" checkpoints experiments/derived
LOG="$D/driver.log"
ET_HARBOR="${TARL_ET_HARBOR:-$HOME/datasets/endless-terminals-harbor}"
ET_SIF="${TARL_ET_SIF:-/var/tmp/tarl/et-sif}"
A0_JOB="$REPO/jobs/2026-07-21__05-38-50"
TRAIN_POOL="$REPO/experiments/et-train-pool.txt"; DEV_POOL="$REPO/experiments/et-dev-pool.txt"
ET_TRAIN="${TARL_ET_TRAIN:-160}"; ET_DEV="${TARL_ET_DEV:-60}"
STEPS="${TARL_STEPS:-50}"; G="${TARL_G:-8}"; TPS="${TARL_TASKS_PER_STEP:-4}"; EVAL_K="${TARL_EVAL_K:-5}"
NC="${TARL_N_CONCURRENT:-4}"; CAP="${TARL_JUDGE_CAP_USD:-40}"; LR="${TARL_LR:-1e-6}"
RERUN_ERRORS="${TARL_A0_RERUN_ERRORS:-CancelledError,RuntimeError,EnvironmentStartTimeoutError}"  # infra-only; timeouts stay failures
PHASES="${TARL_PHASES:-health serve smoke gpu_cells a0_resume et_prep a0_etdev b1 b1_eval b2 b2_eval}"
STAMP=$(date +%Y%m%d)

log()  { echo "$(date '+%F %T') [driver] $*" | tee -a "$LOG"; }
die()  { log "FATAL: $*"; exit 1; }
done_marker() { echo "$(date '+%F %T')" > "$D/$1.done"; log "phase $1 DONE"; }
serve_up() { curl -sf --max-time 5 localhost:8000/v1/models >/dev/null 2>&1; }
serve_restart() {  # $1 = merged-policy dir or "" (base); $2 = thinking 1|0
    local pol="$1" thinking="${2:-1}" extra=""
    echo "$thinking" > .serve-thinking; echo "$pol" > .serve-model
    [ -n "$pol" ] && extra="TARL_SERVE_MODEL=$pol TARL_SERVE_QUANT=fp8"
    tmux kill-session -t serve 2>/dev/null || true; sleep 3
    tmux new-session -d -s serve "cd $REPO && TARL_THINKING=$thinking $extra bash scripts/01_serve.sh 2>&1 | tee $REPO/serve.log"
    log "serve launching (model=${pol:-base}, thinking=$thinking)"
    for _ in $(seq 1 180); do serve_up && { log "serve healthy"; return 0; }; sleep 5; done
    die "serve did not come up (model=${pol:-base}); see serve.log"
}
ensure_serve() {  # make sure the serve runs policy $1 ("" = base)
    local want="$1" have; have=$(head -1 .serve-model 2>/dev/null | tr -d '[:space:]')
    if serve_up && [ "$have" = "$want" ]; then return 0; fi
    serve_restart "$want" 1
}
ensure_watch() { tmux has-session -t watch 2>/dev/null || { tmux new-session -d -s watch "bash $REPO/scripts/watch.sh"; log "watch.sh started"; }; }
latest_job() { ls -dt "$1"/2* 2>/dev/null | head -1; }
harbor_eval() {  # $1 jobs-dir  $2 tasks-root  $3 k  $4 tasks-file|"" ; prints the job dir
    local jobs="$1" root="$2" k="$3" pool="${4:-}" sel=()
    [ -n "$pool" ] && while read -r t; do [ -n "$t" ] && [ "${t#\#}" = "$t" ] && sel+=(-i "$t"); done < "$pool"
    mkdir -p "$jobs"
    harbor run -p "$root" -a terminus-2 -m openai/gemma-4-E4B-it \
        --ak api_base=http://localhost:8000/v1 --ak temperature=1.0 \
        -e singularity --ek singularity_image_cache_dir="$TARL_SIF_CACHE" --ek singularity_no_mount=home,tmp \
        -k "$k" --n-concurrent "$NC" --timeout-multiplier 1.0 \
        --max-retries 1 --retry-exclude AgentTimeoutError,VerifierTimeoutError \
        --jobs-dir "$jobs" "${sel[@]}" >> "$D/harbor.log" 2>&1
    latest_job "$jobs"
}
pool_flags() { local sel=(); while read -r t; do [ -n "$t" ] && [ "${t#\#}" = "$t" ] && sel+=("$t"); done < "$1"; echo "${sel[@]}"; }

run_phase() {
    local ph="$1"
    if [ -f "$D/$ph.done" ]; then log "phase $ph already done ($(cat "$D/$ph.done")) — skip"; return 0; fi
    log "=== phase $ph START ==="
    "phase_$ph" || die "phase $ph failed"
    done_marker "$ph"
}

# ---------------------------------------------------------------------------- phases
phase_health() {
    [ "$(hostname)" = "node08" ] || die "run this on node08 (hostname=$(hostname))"
    bash scripts/_node08_health.sh 2>&1 | tee "$D/health.txt" | tail -n +1 >> "$LOG"
    [ "$(nvidia-smi -L 2>/dev/null | grep -c A4000)" = "2" ] || die "expected 2x A4000"
    mkdir -p /var/tmp/tarl/sif-cache /var/tmp/tarl/apptainer-cache "$ET_SIF"
    [ -e /tmp/sif-cache ] || ln -s /var/tmp/tarl/sif-cache /tmp/sif-cache
    [ -e /tmp/apptainer-cache ] || ln -s /var/tmp/tarl/apptainer-cache /tmp/apptainer-cache
    if [ ! -f /tmp/tb2/build-pmars/task.toml ]; then
        [ -d "$HOME/datasets/terminal-bench-2-1/tasks" ] || die "no TB 2.1 source at ~/datasets/terminal-bench-2-1"
        mkdir -p /tmp/tb2 && rsync -a --exclude README.md "$HOME/datasets/terminal-bench-2-1/tasks/" /tmp/tb2/
        log "restored /tmp/tb2 from ~/datasets/terminal-bench-2-1 ($(ls /tmp/tb2 | wc -l) tasks)"
    fi
    local free; free=$(df --output=avail -BG /var/tmp | tail -1 | tr -dc 0-9)
    [ "$free" -ge 12 ] || log "WARNING: only ${free}G free on /var/tmp (images + merged policy need ~30G)"
    (bash "$HOME/exporters/exporters_ctl.sh" start >> "$D/exporters.log" 2>&1 || log "exporters start returned non-zero") &
    return 0
}
phase_serve() { ensure_serve ""; ensure_watch; log "probe: $(curl -s --max-time 120 localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{"model":"gemma-4-E4B-it","messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":8}' | head -c 300)"; }
phase_smoke() {
    ensure_serve ""
    "$TARL" run --task fix-git --k 1 --live --jobs-dir "$REPO/jobs/dev" >> "$D/smoke.log" 2>&1 || return 1
    local j; j=$(latest_job "$REPO/jobs/dev"); [ -n "$j" ] || return 1
    ls "$j"/fix-git__*/result.json >/dev/null 2>&1 || { log "smoke: no result.json in $j"; return 1; }
    "$TARL" eval "$j" --json "$D/smoke.json" >> "$D/smoke.log" 2>&1 || true
    log "smoke episode ok: $j"
}
phase_gpu_cells() {
    ensure_serve ""
    mkdir -p notebooks/_out
    HF_HOME="$HOME/hf-cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg OMP_NUM_THREADS=8 \
    TARL_LIVE_JUDGE=0 TARL_GPU_CELLS=1 CUDA_VISIBLE_DEVICES=1 \
    "$PY" -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=tarl \
        --output-dir notebooks/_out --output 00_terminal_agent_rl_poc.ipynb notebooks/00_terminal_agent_rl_poc.ipynb >> "$D/gpu_cells.log" 2>&1 || return 1
    "$PY" - <<'EOF' | tee -a "$LOG"
import json; h = json.load(open("docs/blog/figures/headline.json"))
print("[driver] headline.gpu =", json.dumps(h.get("gpu"), indent=None)[:1200])
EOF
}
phase_a0_resume() {
    ensure_serve ""
    local f=(); IFS=, read -ra types <<< "$RERUN_ERRORS"; for t in "${types[@]}"; do [ -n "$t" ] && f+=(-f "$t"); done
    log "harbor jobs resume $A0_JOB (re-running error types: $RERUN_ERRORS)"
    harbor jobs resume -p "$A0_JOB" "${f[@]}" >> "$D/a0_resume.harbor.log" 2>&1 || log "harbor jobs resume exited non-zero (see a0_resume.harbor.log)"
    "$TARL" eval "$A0_JOB" --json "$D/a0.json" > "$D/a0_eval.txt" 2>&1 || return 1
    log "A0: $("$PY" -c "import json;d=json.load(open('$D/a0.json'));print({k:d.get(k) for k in ('n_clean','n_pass','pass_at_1','pass_at_1_ci95','n_error')})")"
    bash scripts/07_sync_node05.sh >> "$LOG" 2>&1 || true
}
phase_et_prep() {
    local root="${TARL_ET_ROOT:-/nfs-share/datasets/endless-terminals}"
    [ "$(ls -d "$root"/task_* 2>/dev/null | wc -l)" -ge 1000 ] || die "Endless Terminals snapshot incomplete at $root (run: tarl data et download)"
    [ -f experiments/et-splits.json ] || "$TARL" data et split --root "$root" --dev-frac 0.1 --seed 0 | tee -a "$LOG"
    "$PY" - "$ET_TRAIN" "$ET_DEV" <<'EOF' | tee -a "$LOG"
import json, sys
from pathlib import Path
s = json.load(open("experiments/et-splits.json")); n_tr, n_dev = int(sys.argv[1]), int(sys.argv[2])
Path("experiments/et-train-pool.txt").write_text("# Endless Terminals train pool (first %d of the seeded train split)\n" % n_tr + "\n".join(s["train"][:n_tr]) + "\n")
Path("experiments/et-dev-pool.txt").write_text("# Endless Terminals dev pool (first %d of the seeded dev split)\n" % n_dev + "\n".join(s["dev"][:n_dev]) + "\n")
print(f"[driver] pools: train {n_tr}/{len(s['train'])}, dev {n_dev}/{len(s['dev'])}")
EOF
    "$TARL" data et table --root "$root" --csv experiments/derived/et-tasks.csv --show 3 >> "$D/et_prep.log" 2>&1 || true
    "$TARL" data et convert --root "$root" --tasks $(pool_flags "$TRAIN_POOL") --out-root "$ET_HARBOR" --sif-dir "$ET_SIF" | tee -a "$LOG" || return 1
    "$TARL" data et convert --root "$root" --tasks $(pool_flags "$DEV_POOL") --out-root "$ET_HARBOR" --sif-dir "$ET_SIF" | tee -a "$LOG" || return 1
    "$TARL" data et leakage --root "$root" --tasks $(pool_flags "$TRAIN_POOL") --tb-root /tmp/tb2 --threshold 0.85 \
        --csv experiments/derived/et-leakage.csv --show 10 > "$D/et_leakage.txt" 2>&1; local rc=$?
    tail -4 "$D/et_leakage.txt" | tee -a "$LOG"
    [ $rc -eq 0 ] || die "leakage guard flagged training tasks (see $D/et_leakage.txt)"
    "$TARL" data et build-sif --root "$root" --tasks $(pool_flags "$TRAIN_POOL") $(pool_flags "$DEV_POOL") --sif-dir "$ET_SIF" --workers 4 >> "$D/et_build.log" 2>&1 || { tail -5 "$D/et_build.log" | tee -a "$LOG"; return 1; }
    log "images: $(tail -1 "$D/et_build.log")"
}
phase_a0_etdev() {
    ensure_serve ""
    local j; j=$(harbor_eval "$REPO/jobs-et/A0-dev" "$ET_HARBOR" 1 "$DEV_POOL"); [ -n "$j" ] || return 1
    "$TARL" eval "$j" --json "$D/a0_etdev.json" > "$D/a0_etdev_eval.txt" 2>&1 || true
    log "A0 on ET dev: $j $("$PY" -c "import json;d=json.load(open('$D/a0_etdev.json'));print({k:d.get(k) for k in ('n_clean','n_pass','pass_at_1','n_error')})" 2>/dev/null)"
}
grpo_arm() {  # $1 arm  $2 lambda  $3.. extra args
    local arm="$1" lam="$2"; shift 2
    local adir="/var/tmp/tarl/$arm"; rm -rf "$adir"; mkdir -p "$adir"
    ensure_serve ""   # every arm starts from the base policy
    local name="$arm-$STAMP"
    log "GRPO $arm: steps=$STEPS G=$G tasks/step=$TPS lr=$LR lambda=$lam pool=$TRAIN_POOL"
    CUDA_VISIBLE_DEVICES=1 HF_HOME="$HOME/hf-cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$PY" scripts/08_grpo_run.py --task-source endless --tb-path "$ET_HARBOR" --tasks-file "$TRAIN_POOL" --tasks-per-step "$TPS" \
        --G "$G" --steps "$STEPS" --lr "$LR" --sync-every 5 --lambda-judge "$lam" --n-concurrent "$NC" \
        --jobs-dir "$REPO/jobs-et/$arm" --adapter-dir "$adir/adapter" --merged-dir "$adir/merged" --run-name "$name" \
        --sif-workers 4 "$@" >> "$D/$arm.log" 2>&1 || { tail -20 "$D/$arm.log" | tee -a "$LOG"; return 1; }
    mkdir -p "checkpoints/$name" && cp -r "$adir/adapter" "checkpoints/$name/" && log "$arm adapter saved to checkpoints/$name/adapter"
    "$TARL" run-summary "runs/$name" > "$D/$arm.summary.txt" 2>&1 || true
    tail -3 "$D/$arm.summary.txt" | tee -a "$LOG"
}
arm_eval() {  # $1 arm
    local arm="$1" adir="/var/tmp/tarl/$1" j jt
    [ -d "$adir/merged" ] || die "$arm merged policy missing at $adir/merged (rerun phase $arm)"
    ensure_serve "$adir/merged"
    j=$(harbor_eval "$REPO/jobs-et/$arm-dev" "$ET_HARBOR" 1 "$DEV_POOL"); "$TARL" eval "$j" --json "$D/${arm}_etdev.json" > /dev/null 2>&1 || true
    log "$arm on ET dev: $j $("$PY" -c "import json;d=json.load(open('$D/${arm}_etdev.json'));print({k:d.get(k) for k in ('n_clean','n_pass','pass_at_1')})" 2>/dev/null)"
    jt=$(harbor_eval "$REPO/jobs" /tmp/tb2 "$EVAL_K" ""); "$TARL" eval "$jt" --json "$D/${arm}_tb2.json" > "$D/${arm}_tb2_eval.txt" 2>&1 || true
    log "$arm on TB2 (k=$EVAL_K): $jt $("$PY" -c "import json;d=json.load(open('$D/${arm}_tb2.json'));print({k:d.get(k) for k in ('n_clean','n_pass','pass_at_1','pass_at_1_ci95')})" 2>/dev/null)"
    "$TARL" compare "$A0_JOB" "$jt" > "$D/${arm}_vs_a0.txt" 2>&1 || true
    echo "{\"arm\": \"$arm\", \"et_dev_job\": \"$j\", \"tb2_job\": \"$jt\", \"eval_k\": $EVAL_K}" > "$D/${arm}_eval.json"
    rm -rf "$adir/merged"; log "$arm merged weights removed from /var/tmp (adapter kept on NFS)"
    bash scripts/07_sync_node05.sh >> "$LOG" 2>&1 || true
}
phase_b1()      { grpo_arm B1 0; }
phase_b1_eval() { arm_eval B1; }
phase_b2() {
    [ -f experiments/.allow-judge-spend ] || die "B2 uses the LIVE grok judge (est. \$20-40): touch experiments/.allow-judge-spend to arm it"
    grpo_arm B2 0.5 --judge live --yes-spend --cost-cap-usd "$CAP" --judge-on-miss error
}
phase_b2_eval() { arm_eval B2; }

# ---------------------------------------------------------------------------- main
log "driver start on $(hostname): phases = $PHASES"
for ph in $PHASES; do run_phase "$ph"; done
ensure_serve "" || true
log "driver finished"
