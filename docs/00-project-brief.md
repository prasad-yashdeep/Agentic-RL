# Project Brief

## Hypothesis

A small open model that is mediocre at agentic reasoning can be turned into a
substantially better *agentic reasoning* model by RL against verifiable
terminal tasks, and the sample-efficiency/ceiling of that RL is materially
improved by guidance from a frontier teacher's trajectories (no teacher
logits required).

## Concrete instantiation (phase 1)

- **Policy:** `google/gemma-4-E4B-it` (Apache 2.0; thinking is a runtime mode
  via `<|think|>`, no separate checkpoint). 4.5B effective / ~8B total params.
  Published Terminal-Bench score: **8.0**.
- **Benchmark:** Terminal-Bench **2.1** — *eval-only* (no train split exists).
  **RL environment/training corpus:** Endless Terminals (~2,500 Harbor-format
  tasks) + terminal-bench-rl (~331) + optional self-generated tasks. Each
  task = container + NL instruction + hidden verifier. Reward = verifier 0/1.
- **Harness:** the benchmark's reference agent (Terminus) for leaderboard
  comparability; an identical-format minimal harness inside the training loop.
- **Teacher:** Claude Sonnet 5 through the *same harness*, traces logged.
- **Algorithm:** GRPO (with DAPO-style fixes) + teacher-trace mixing
  (LUFFY-style) + SFT warm-start from teacher traces.

## Ablation plan

| Arm | Warm-start | RL | Teacher in RL | Question answered |
|---|---|---|---|---|
| A0 | — | — | — | baseline: raw model on benchmark |
| A1 | SFT on teacher traces | — | — | how far does pure distillation go? |
| A2 | — | GRPO | — | can RL alone lift a weak policy? (cold-start risk) |
| A3 | SFT | GRPO | — | standard pipeline |
| A4 | SFT | GRPO | mixed teacher traces | full method — does trace-nudging beat A3? |

Success metric: pass@1 on a held-out task split (n runs ≥ 5, report mean ± CI).

## Milestones

1. **M0 — plumbing:** vLLM serves the policy on node08; harness runs one task
   end-to-end against it (off-cluster docker if needed).
2. **M1 — baseline:** A0 numbers on the eval split; teacher (Sonnet 5) numbers
   on the same split as the ceiling reference.
3. **M2 — traces + SFT:** teacher trajectories on the *training* split;
   A1 trained and evaluated.
4. **M3 — RL:** A2/A3 GRPO runs stable (nonzero gradient, no collapse).
5. **M4 — full method:** A4; write-up.

## Long-term direction

Generalize beyond Terminal-Bench: policy = any open model; env = computer/VM
(OSWorld-style GUI or broader tool spaces); reward = task completion. The
architecture below keeps the env behind a thin interface so the terminal env
is just the first plug-in.
