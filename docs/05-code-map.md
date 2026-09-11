# Code map (added 2026-09-09) — the `tarl` package, CLI, run script, notebook, blog page

Everything runnable today runs on CPU against the recorded artifacts (`jobs/`, `jobs-g4/`, `g4.log`,
`experiments/judge-pilot.jsonl`, `watch.log`). GPU paths are guarded and skip with a note.
`scripts/06_grpo_micro.py` is frozen as the gate-G4 artifact; `scripts/08_grpo_run.py` is the forward path.

| Module | What it does | Extracted from / notes |
|---|---|---|
| `tarl.common.paths` / `env` | repo-relative paths; `is_headnode()`, `has_cuda()`, `require_cuda()` | never compute on the head node |
| `tarl.data.loader` | `Trial`/`Episode`, `iter_trials`, `load_episode`, `job_summary` (one row per trial), `turns_table`, `legacy_rollout_order` (06's unsorted glob order), `lane_pack` | tolerant of errored trials; statuses clean / errored_with_trajectory / errored_no_trajectory / unverified / incomplete |
| `tarl.data.instruction` | `task_instruction()` from the first user step (`Task Description:` … `Current terminal state:`) | the TB2 clone is not on disk |
| `tarl.data.cast` | asciicast v2 reader: `keystrokes()` = what was typed | ground truth of commands |
| `tarl.data.domains` | 89 tasks → 8 hand-curated domains (`configs/tasks/tb2_domains.yaml`) | replace via `tarl data task-meta --tb-path /tmp/tb2` |
| `tarl.data.examples` | `build_example(tok, traj, render="legacy06"|"terminus_chat")` → ids + assistant mask (+ spans); `reconstruction_gap`; offsets-based mask fallback when prefix tokenisation is unstable | **finding 1**: legacy06 = 06's prose rendering; the Gemma template merges consecutive same-role turns |
| `tarl.data.watchlog` / `canary` | heartbeat log parser; canary-string scan | |
| `tarl.judge.compressor` | legacy prompt (byte-identical to pilot/G4) + `include_actions=True, task_only=True` v2 | **finding 2**: the pilot judge saw prose only |
| `tarl.judge.{protocol,config,replay,mock,cost,scoring,alarms,g4log}` | `Judge` protocol; `judge.yaml` finally read; `ReplayJudge` (12 exact pilot + 40 order-inferred G4 verdicts keyed by prompt hash); `MockJudge`; cost cap; `score_rollouts` with alarms; `g4.log` parser | live calls only with `--yes-spend`; key in `~/.config/tarl/env` |
| `tarl.training.{advantages,loss,model,curriculum,coldstart,run_config,metrics,rollouts,loop}` | Dr.GRPO/GRPO advantages; clipped surrogate + `grpo_step` (06 verbatim); GPU-guarded QLoRA load/merge; Beta-posterior curriculum + dynamic sampling; cold-start arithmetic + SYNTHETIC simulator; argparse `RunConfig`; JSONL metrics; replay/harbor rollout sources; the run loop with g4.log parity check | `make dry-run` → `DRY_RUN_PARITY_OK` |
| `tarl.eval.{stats,report,manifest}` | Wilson, pass@k (Harbor's estimator), bootstrap, McNemar, paired compare; summary/per-task/per-domain/taxonomy tables; run manifests | `summary_table(A0)` reproduces 5/150, 3.61 %, 5 tasks |
| `tarl.embed.{model,analysis}` | MiniLM via `transformers` mean pooling (TF-IDF fallback); PCA, k-means (numpy); leakage report; failure clusters | weights: `make embed-model` |
| `tarl.serving.{harbor_cmd,replay}` + `tarl.cli` | the single `harbor run` builder; serve lifecycle; `tarl replay|eval|compare|run-summary|g4log|manifest|judge|coldstart|embed|run|data` | the CLI is the whole "interaction platform" |
| `tarl.viz` | SVG figures with `currentColor` text, table twins, token grid, `headline.json` schema | consumed by the blog page |
| `notebooks/build_notebook.py` → `00_terminal_agent_rl_poc.ipynb` | the anchor notebook (41 cells, ≈20 s on node03) | `make notebook` |
| `docs/blog/build_page.py` + `template.html` → `terminal-agent-rl.html` | the explainer page; every number from `figures/headline.json` | publish as an Artifact |

## Workflow
```
make install-dev && make kernel && make embed-model   # once
make test          # 64 tests on node03 (~1 min)
make dry-run       # G4 replay through scripts/08 → DRY_RUN_PARITY_OK
make notebook      # figures + headline.json (node03)
.venv/bin/python docs/blog/build_page.py             # rebuild the page, then republish the Artifact
```
After node08 returns: `make exp` launches `scripts/09_experiments.sh` in tmux `exp` on node08 — health/serve/smoke,
notebook GPU cells, A0 resume, Endless Terminals prep, A0 on ET-dev, B1, B1 eval, B2 (only once
`experiments/.allow-judge-spend` exists), B2 eval; `make exp-log` follows it, `runs/driver/<phase>.done` markers make
every phase idempotent. Then register the job dirs in `experiments/runs.yaml`, rerun the notebook, rebuild and
republish the page.

### Training corpus (added 2026-09-10)

| module / file | what | notes |
|---|---|---|
| `tarl.data.endless` | Endless Terminals snapshot → Harbor tree → `.sif` images; splits; leakage guard | `download`, `list_tasks/read_task/task_table`, `split/write_splits`, `convert` (adds `docker_image`), `build_sif/ensure_sifs` (apptainer, node-local, locked), `leakage_guard` (MiniLM vs TB2 instructions) |
| `tarl data et …` | CLI: `download | table | split | convert | build-sif | leakage` | `leakage` exits 3 when any training task is within cosine 0.85 of a TB2 task |
| `RunConfig` flags | `--task-source endless`, `--tasks-file`, `--tasks-per-step N`, `--sample-strategy`, `--sif-workers` | per-step curriculum sampling; `HarborRolloutSource(before_collect=…)` builds images first |
| `scripts/09_experiments.sh` | the unattended sequence on node08 (tmux `exp`) | knobs via `TARL_*` env; logs in `runs/driver/` |
| `scripts/_node08_health.sh` | one-shot node08 report | run before anything else after a reboot |
| paths | `ET_ROOT` `/nfs-share/datasets/endless-terminals`, `ET_HARBOR` `~/datasets/endless-terminals-harbor`, `ET_SIF` `/var/tmp/tarl/et-sif`, `ET_SPLITS` | env overrides `TARL_ET_ROOT/HARBOR/SIF` |

## Findings recorded while building this (2026-09-09)
1. The G4 trainer tokenised Harbor's "Analysis/Plan" prose (`message`), not the JSON actions, commands or
   terminal observations; and the Gemma-4 template merges consecutive same-role messages, so the episode
   collapsed into one model turn (incremental tokenisation was not prefix-stable). Plumbing proof only.
2. The judge compressor rendered the same prose, so pilot/G4 judge scores are narrative-only.
3. Reproduced from disk: A0 = 5/150 clean = 3.33 % (Wilson [1.4 %, 7.6 %]), mean per-task 3.61 %, 5/83
   tasks, 26 errors by type; all 40 G4 composite rewards reproduce from replayed judge scores within 5e-4.
