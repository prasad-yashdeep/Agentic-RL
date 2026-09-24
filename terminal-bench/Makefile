# Dev workflow for terminal-bench (Agentic-RL monorepo). GNU make >= 3.82 (RECIPEPREFIX avoids tab-vs-space accidents).
.RECIPEPREFIX = >
SHELL := /bin/bash
# Self-locating: REPO = terminal-bench/ (real path), ARL = the Agentic-RL monorepo root, PY = the shared venv.
# (No inline comments on assignments: make keeps the spaces before '#' in the value.)
REPO  := $(patsubst %/,%,$(dir $(realpath $(lastword $(MAKEFILE_LIST)))))
ARL   := $(patsubst %/,%,$(dir $(REPO)))
PY    := $(ARL)/.venv/bin/python
UV    := $(HOME)/.local/bin/uv
NODE  ?= node03
ENV   := PYTHONDONTWRITEBYTECODE=1 HF_HOME=$(HOME)/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MPLBACKEND=Agg
NB    ?= 00_terminal_agent_rl_poc.ipynb
T     ?= tests/test_advantages.py

.PHONY: install-dev kernel embed-model test test-local dry-run notebook lint clean-figures exp exp-log exp-status doctor

install-dev:  ## dev deps + console script (headnode ok; never re-resolves the CUDA stack)
> df -h $(HOME) | tail -1
> $(UV) pip install --python $(PY) --no-deps -e .
> $(UV) pip install --python $(PY) pytest matplotlib ipykernel nbconvert nbclient nbformat

kernel:       ## register the venv as a Jupyter kernel named `tarl`
> $(PY) -m ipykernel install --user --name tarl --display-name "tarl (.venv)"

embed-model:  ## download MiniLM (~90 MB) into $HF_HOME (headnode has internet)
> HF_HOME=$(HOME)/hf-cache $(PY) -c "from huggingface_hub import snapshot_download as s; print(s('sentence-transformers/all-MiniLM-L6-v2', allow_patterns=['*.json','*.txt','model.safetensors']))"

test:         ## full suite on a compute node (standing rule: no compute on the headnode)
> ssh -o BatchMode=yes $(NODE) 'cd $(REPO) && $(ENV) $(PY) -m pytest -q tests'

test-local:   ## ONE test file on the headnode (light)
> $(ENV) $(PY) -m pytest -q $(T) -x

dry-run:      ## replay jobs-g4 through scripts/08 on a compute node (CPU; parity vs g4.log)
> ssh -o BatchMode=yes $(NODE) 'cd $(REPO) && $(ENV) $(PY) scripts/08_grpo_run.py --dry-run --replay-jobs jobs-g4 --g4-log g4.log --judge replay --lambda-judge 0.5 --tasks fix-git regex-log --G 4 --steps 5'

notebook:     ## execute the anchor notebook headlessly on a compute node
> ssh -o BatchMode=yes $(NODE) 'cd $(REPO) && mkdir -p notebooks/_out && $(ENV) TARL_LIVE_JUDGE=0 TARL_GPU_CELLS=0 $(PY) -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=tarl --output-dir notebooks/_out --output $(NB) notebooks/$(NB)'

lint:         ## byte-compile + whitespace check
> $(PY) -m compileall -q src tests scripts && git diff --check

clean-figures:
> rm -f docs/blog/figures/*.svg docs/blog/figures/*.md docs/blog/figures/*.csv docs/blog/figures/*.html docs/blog/figures/headline.json

exp:          ## launch the unattended experiment driver on node08 (tmux `exp`); TARL_PHASES="..." to subset
> bash $(ARL)/tools/doctor.sh
> ssh -o BatchMode=yes node08 'tmux has-session -t exp 2>/dev/null && echo "exp already running" || tmux new -d -s exp "cd $(REPO) && TARL_PHASES=\"$(TARL_PHASES)\" bash scripts/09_experiments.sh"'

exp-log:      ## follow the driver log
> tail -n 40 -f $(REPO)/runs/driver/driver.log

exp-status:   ## phase markers + last log lines
> ls -la $(REPO)/runs/driver/*.done 2>/dev/null; tail -n 15 $(REPO)/runs/driver/driver.log 2>/dev/null

doctor:       ## monorepo health: stale paths, venv, kernel, A0 resume consistency
> bash $(ARL)/tools/doctor.sh
