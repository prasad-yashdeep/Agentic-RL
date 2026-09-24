# swe-bench (Agentic-RL). `make help`. Compute never runs on the headnode: targets ssh to NODE (tests) or node08 (GPU).
include ../mk/common.mk
SWE := $(ARL)/swe-bench
GPU_NODE ?= node08

.PHONY: help install fetch assets test phase1 sifs grader serve train notebook status

help:        ## this list
> @grep -hE '^[a-z0-9-]+:.*## ' Makefile | sed -E 's/^([a-z0-9-]+):[^#]*## /\1\t/'

install:     ## editable install into the shared venv (--no-deps, never re-resolve) + Jupyter kernel `sarl`
> ~/.local/bin/uv pip install --python $(PY) --no-deps -e $(SWE)
> $(PY) -m ipykernel install --user --name sarl --display-name "sarl (Agentic-RL .venv)"

fetch:       ## upstream reference at the pinned SHA into reference/_upstream (sha256-verified, gitignored)
> bash $(SWE)/reference/fetch.sh

assets:      ## headnode, download only: pinned Qwen2.5-Coder-0.5B-Instruct + SWE-bench datasets into ~/hf-cache
> bash $(SWE)/scripts/00_assets.sh

test:        ## test suite on NODE (default node03; NODE=node08 while node03 is down)
> ssh -o BatchMode=yes $(NODE) 'cd $(SWE) && $(ENV) $(PY) -m pytest tests'

phase1:      ## run the upstream script + notebook as-is on the GPU node (GPU1)
> ssh -o BatchMode=yes $(GPU_NODE) 'bash $(SWE)/scripts/phase1_reference.sh'

sifs:        ## pull the pool's SWE-bench images to node08 /tmp/swe-sif
> ssh -o BatchMode=yes $(GPU_NODE) 'bash $(SWE)/scripts/10_pull_sifs.sh'

grader:      ## node-local swebench==5.0.2 grader venv on node08 (/var/tmp/swe/grader-venv)
> ssh -o BatchMode=yes $(GPU_NODE) 'bash $(SWE)/scripts/11_grader_venv.sh'

serve:       ## vLLM policy server on node08 GPU0 :8001 (tmux swe-serve)
> ssh -o BatchMode=yes $(GPU_NODE) 'tmux has-session -t swe-serve 2>/dev/null && echo "swe-serve already running" || tmux new -d -s swe-serve "SARL_MODEL=$(or $(MODEL),0.5b) bash $(SWE)/scripts/20_serve_policy.sh"'

train:       ## basic GRPO run on node08 GPU1 against the serve (tmux swe-train); CONFIG=configs/basic_run.yaml
> ssh -o BatchMode=yes $(GPU_NODE) 'tmux has-session -t swe-train 2>/dev/null && echo "swe-train already running" || tmux new -d -s swe-train "bash $(SWE)/scripts/21_train.sh $(or $(CONFIG),configs/basic_run.yaml)"'

notebook:    ## execute notebooks/01_swe_rl_anchor.ipynb on the GPU node
> ssh -o BatchMode=yes $(GPU_NODE) 'cd $(SWE) && mkdir -p notebooks/_out && $(ENV) $(PY) -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=sarl --output-dir notebooks/_out notebooks/01_swe_rl_anchor.ipynb'

status:      ## tmux sessions, GPU use, latest run
> ssh -o BatchMode=yes $(GPU_NODE) 'tmux ls 2>/dev/null; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'; ls -dt $(SWE)/runs/*/ 2>/dev/null | head -3
