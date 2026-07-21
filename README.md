# terminal-agent-rl

Turning a small open model into a stronger agentic-reasoning model via RL on
terminal tasks, with a frontier teacher (Claude Sonnet 5) nudging the policy.

## The idea

1. **Baseline** — evaluate a small Gemma model on [Terminal-Bench](https://www.tbench.ai/)
   through an agentic harness.
2. **RL** — improve it with GRPO: the environment is a real terminal inside a
   container, the episode is a multi-turn agent loop, the reward is binary task
   success from the benchmark's own verifier.
3. **Teacher guidance** — collect Sonnet 5 trajectories through the *same*
   harness and use them to (a) SFT warm-start the policy and (b) mix teacher
   traces into GRPO groups during RL (LUFFY-style off-policy guidance).
4. **Long-term** — generalize the recipe: any open-weights policy + any agentic
   harness acting on a computer/VM environment + verifiable task reward.

## Docs

| File | Contents |
|---|---|
| [docs/00-project-brief.md](docs/00-project-brief.md) | Goals, scope, milestones |
| [docs/01-infra.md](docs/01-infra.md) | Cluster reality (GPUs, docker), deployment topology |
| [docs/02-architecture.md](docs/02-architecture.md) | System design: eval loop, training loop, teacher pipeline |
| [docs/03-math.md](docs/03-math.md) | Mathematical formulations: POMDP, GRPO + variants, teacher-guidance objectives |
| [docs/04-gaps-and-risks.md](docs/04-gaps-and-risks.md) | Known gaps, risks, and mitigations |
| [experiments/EXPERIMENTS.md](experiments/EXPERIMENTS.md) | Running experiment log |

## Status

- [x] Repo created, infra surveyed (2026-07-20)
- [x] Container path solved: unprivileged userns enabled on node08 → user-space
      apptainer + Harbor singularity backend (no admin needed)
- [ ] **Anthropic written authorization for teacher-trace distillation (ToS gate — send request first)**
- [ ] Apptainer installed on node08; Harbor oracle run passes on 3 tasks
- [ ] vLLM serves gemma-4-E4B-it FP8 on GPU0; parser validated on easy tasks
- [ ] Baseline eval: Gemma E4B on Terminal-Bench
- [ ] Teacher trace collection (Sonnet 5)
- [ ] SFT warm-start
- [ ] GRPO training run 1 (no teacher) — ablation C1
- [ ] GRPO + teacher-mixed groups — ablation C2
