# Agentic-RL

Reinforcement learning for tool-using agents, run on a small on-prem cluster: 2× RTX A4000 16 GB on node08, no Docker, apptainer only.
The monorepo holds two subprojects. They share one Python environment and the same GRPO-style training ideas.

| Subproject | Task family | Harness | Reward | Status (2026-09-24) |
|---|---|---|---|---|
| [`terminal-bench/`](terminal-bench/) | Terminal tasks: train on Endless Terminals, evaluate on Terminal-Bench 2.1 | Terminus 2 via Harbor, singularity backend | Verifier pass, plus a gated LLM-judge shaping term | Pipeline proven (gates G0–G4). Interim baseline 3.3 % pass@1. The A0/B1/B2 driver is ready and waits for GPU time on node08. |
| [`swe-bench/`](swe-bench/) | Repository bug fixing: train on the SWE-bench dev split, evaluate on SWE-bench Verified | Basic agentic loop, one bash action per turn (after the Vizuara SWE-RL lecture) | Tests of the replayed patch in a clean, network-less container | Phase 1 done (2026-09-24). The reference ran unmodified; the diagram's loop runs end to end on node08: vLLM serve → apptainer SWE-bench sandboxes → test reward → LoRA GRPO → adapter sync. The 0.5B policy resolves 0 tasks, so this proves the plumbing, not learning. |

```
Agentic-RL/
├── terminal-bench/   package `tarl`  (formerly the terminal-agent-rl repo; history kept)
├── swe-bench/        package `sarl`
├── tools/            doctor.sh (health check), check_layering.py (tarl and sarl stay independent)
├── mk/common.mk      shared make settings (venv, NODE, offline HF env)
└── .venv/            one shared venv (gitignored)
```

## Quick start

```bash
cd /home/yashp/Agentic-RL
make help
make doctor                     # stale paths, venv, kernels, Harbor resume consistency
make test NODE=node08           # both test suites over ssh (never on the head node)
make lint                       # compileall + whitespace + import boundaries
make tb-exp                     # terminal-bench experiment driver on node08 (doctor runs first)
make swe-help                   # swe-bench targets
```

Cluster facts, standing rules and monorepo rules are in [`CLAUDE.md`](CLAUDE.md). Each subproject has its own `README.md`, `CLAUDE.md`, `docs/` and `experiments/EXPERIMENTS.md`.

## History

Until 2026-09-24 this was `terminal-agent-rl`, checked out at `/home/yashp/terminal-agent-rl`. The move into `terminal-bench/` is a single pure-rename commit, and `git log --follow` works across it. The tag `pre-agentic-rl` marks the old layout.

There is no compatibility symlink. Every stored absolute path was rewritten, and `make doctor` verifies this. The rewritten places are the venv shebangs, the editable install, the Jupyter kernels and 576 Harbor job/run records.
