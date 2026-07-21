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
- **Teacher = LLM-as-judge (clarified 2026-07-20):** Claude Sonnet 5 (or
  Haiku for cheap scoring) shapes the GRPO reward by scoring trajectories
  against a rubric — NOT knowledge distillation. Gated composite: pass ⇒ R=1,
  else R = λ·r_J (λ ≤ 0.5), so the verifier always dominates. See
  docs/03 §4.4 for the full formulation and anti-hacking design.
- **Algorithm:** GRPO (DAPO/Dr.GRPO fixes) + judge-shaped reward; optional
  teacher-free warm-start via self-STaR. Trace-mixing (LUFFY/CHORD) kept only
  as documented extensions (they'd require distillation authorization).

## Ablation plan

| Arm | Warm-start | RL reward | Question answered |
|---|---|---|---|
| A0 | — | — | baseline: raw model on benchmark |
| B1 | — | verifier only (sparse 0/1) | can sparse-reward GRPO lift a weak policy? (cold-start control) |
| B2 | — | verifier + judge-shaped (gated) | **headline: does judge shaping beat sparse reward?** |
| B3 | self-STaR (own passing rollouts) | verifier + judge-shaped | does a teacher-free warm-start add on top? |

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
