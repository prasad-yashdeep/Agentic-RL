#!/usr/bin/env bash
# Gate S1: run the Vizuara reference exactly as published (abgoswam/swe_in_prod_vizuara_01 @ 8cddfb60), on node08:
#   1. swe_grpo_one_step.py        (G=6 model rollouts, reward_random, one LoRA REINFORCE step)
#   2. the 44-cell notebook         (G=8 = 7 model rollouts + the scripted oracle), via nbconvert, kernel `sarl`
# Upstream files come from reference/_upstream (sha256-verified); utils.py is staged next to them, so the
# script's unpinned urlretrieve fallback never fires. Only the notebook's two `!pip` lines are dropped.
set -uo pipefail
case "$(hostname)" in bcm11-headnode*) echo "refusing on the headnode"; exit 1;; esac
SWE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"; ARL="$(cd "$SWE/.." && pwd -P)"; PY="$ARL/.venv/bin/python"
bash "$SWE/reference/fetch.sh" >/dev/null || { echo "fetch/verify failed"; exit 1; }
TS=$(date +%Y%m%d-%H%M%S); OUT="$SWE/runs/phase1/$TS"; mkdir -p "$OUT/script" "$OUT/nb"
U="$SWE/reference/_upstream"
cp "$U/utils.py" "$U/swe_grpo_one_step.py" "$OUT/script/"; cp "$U/utils.py" "$U/swe_grpo_one_step.cluster.ipynb" "$OUT/nb/"
# Online, as published (Colab): upstream loads the model and dataset WITHOUT a revision, which offline mode cannot
# resolve. We record what `main` resolved to afterwards and compare with our pins (tarl/sarl pin by commit).
export HF_HOME="${HF_HOME:-$HOME/hf-cache}"; unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES="${SARL_PHASE1_GPU:-1}"
echo "phase1 -> $OUT (GPU ${CUDA_VISIBLE_DEVICES})"
( cd "$OUT/script" && /usr/bin/time -v "$PY" swe_grpo_one_step.py ) > "$OUT/script/run.log" 2> "$OUT/script/time.log"
echo "script exit=$?"
( cd "$OUT/nb" && "$PY" -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 \
    --ExecutePreprocessor.kernel_name=sarl --output executed.ipynb swe_grpo_one_step.cluster.ipynb ) > "$OUT/nb/nbconvert.log" 2>&1
echo "notebook exit=$?"
for r in "models--Qwen--Qwen2.5-Coder-0.5B-Instruct:ea3f2471cf1b1f0db85067f1ef93848e38e88c25" "datasets--princeton-nlp--SWE-bench_Verified:c104f840cc67f8b6eec6f759ebc8b2693d585d4a"; do
    d="${r%%:*}"; pin="${r##*:}"; got=$(cat "$HF_HOME/hub/$d/refs/main" 2>/dev/null || echo none)
    [ "$got" = "$pin" ] && echo "revision ok  $d main=$got" || echo "REVISION DRIFT $d main=$got pinned=$pin"
done | tee "$OUT/revisions.txt"
"$PY" -m sarl.phase1_summary "$OUT" && echo "PHASE1_DONE $OUT"
