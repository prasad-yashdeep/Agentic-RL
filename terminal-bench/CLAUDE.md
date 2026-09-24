# CLAUDE.md — terminal-bench (Agentic-RL)

Project notes for `terminal-bench/` (package `tarl`). Cluster map, standing rules and the monorepo rules live in
the root `CLAUDE.md`. Design: `docs/00–05`, `experiments/EXPERIMENTS.md`.

## Quick pointers

- Purpose and design: `README.md`, `docs/00–04`, `experiments/EXPERIMENTS.md`
  (gates G0–G4 all passed 2026-07-21; next = full A0 baseline, then B1/B2).
- Runtime contract: `scripts/env.sh` (`NCCL_P2P_DISABLE=1`, `HF_HOME`, PATH
  order shims → fakeroot-compat → apptainer-env → `~/.local/bin`, secrets from
  `~/.config/tarl/env`). Pins: `configs/pins.md`, `configs/lockfile.txt`.
  Serve: `scripts/01_serve.sh` (vLLM FP8 gemma-4-E4B-it on GPU0 :8000, 32k
  ctx). Babysitter: `scripts/watch.sh` (tmux `watch` on node08). Capacity
  sync: `scripts/07_sync_node05.sh` → `node05:/local/yashp/tarl`. GRPO
  micro-loop: `scripts/06_grpo_micro.py`.
- Run artifacts (gitignored, on NFS): `jobs/`, `jobs-g4/`, `*.log`,
  `.serve-thinking`.
- Judge: grok-4.5 via xAI (`configs/judge/judge.yaml`); `XAI_API_KEY` in
  `~/.config/tarl/env` — never commit.

## Code map, tests, notebook, blog page (added 2026-09-09)

- `src/tarl/` is now real code (data loader, judge replay/mock/cost/alarms/config, training math + run loop,
  eval stats, embeddings, CLI `tarl`, viz). Map and workflow: `docs/05-code-map.md`. `scripts/06_grpo_micro.py`
  is frozen (gate-G4 artifact); `scripts/08_grpo_run.py` is the forward path (`--dry-run` replays `jobs-g4`).
- Dev loop: `make test` (runs over ssh on `NODE`, default node03; use `NODE=node08` while node03 is down), `make dry-run`, `make notebook`
  (`notebooks/00_terminal_agent_rl_poc.ipynb` → `docs/blog/figures/headline.json` + SVG/CSV twins),
  `../.venv/bin/python docs/blog/build_page.py` (→ `docs/blog/terminal-agent-rl.html`, published as an Artifact).
  Remote runs set `PYTHONDONTWRITEBYTECODE=1`: NFS attribute caching once served node03 a stale `__pycache__`
  right after a headnode edit — clear `__pycache__` if a remote run ignores a fresh change.
- Installs: `uv pip install --python ../.venv/bin/python --no-deps -e .` only (the pyproject `transformers==5.5.4`
  pin is stale vs the installed 5.14.1; a re-resolve would break the CUDA lane). Dev deps (pytest, matplotlib,
  ipykernel, nbconvert) and MiniLM (`~/hf-cache`, 88 MB) are installed. Jupyter kernel `tarl` registered;
  the cluster `jupyter` module is broken, always use `../.venv/bin/python -m jupyter`.
- Judge API key: `~/.config/tarl/env` holds `XAI_API_KEY` (never print it). All judge use defaults to
  replay/mock; live calls only with `--yes-spend`.
- Training corpus (2026-09-10): **Endless Terminals** at `/nfs-share/datasets/endless-terminals` (2,492 Harbor
  tasks, 106 MB); converted tree `~/datasets/endless-terminals-harbor`; images built per task on node08 into
  `/var/tmp/tarl/et-sif` (≈50 MB each, `apptainer build --fakeroot`, validated on node03). TB 2.1 source clone:
  `~/datasets/terminal-bench-2-1` (restores `/tmp/tb2`). Never train on TB2 tasks (eval set); the leakage guard
  (`tarl data et leakage`) must pass before any run. Pools: `experiments/et-{train,dev}-pool.txt`.
- Experiments run through `scripts/09_experiments.sh` (`make exp`, tmux `exp` on node08, log `runs/driver/`).
  B2 spends real money on the xAI judge and only runs once `experiments/.allow-judge-spend` exists.
- Two findings to keep in mind (details in `docs/04-gaps-and-risks.md` #20–22, `experiments/EXPERIMENTS.md`
  2026-09-09): the G4 trainer tokenised prose not actions (and the Gemma template merges consecutive
  same-role turns), and the judge pilot saw prose only. Use `render="terminus_chat"` and
  `compress(include_actions=True, task_only=True)` for anything new.

