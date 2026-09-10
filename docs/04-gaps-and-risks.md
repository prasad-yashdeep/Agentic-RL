# Gaps & Risks Register

From an adversarial review of the plan against the research findings
(2026-07-20). Status: ✅ resolved in architecture · 🔶 mitigation designed,
work pending · 🔴 open — needs a decision/action before the affected phase.

## Blockers

| # | Gap | Status | Action |
|---|---|---|---|
| 1 | **Anthropic ToS.** Original distillation plan was squarely prohibited without prior authorization. Clarified design (judge-only, no trace imitation) has *much lighter* exposure — but a strict reading of "utilization of outputs to train an AI model" still covers reward scores used in training. | 🔶 (downgraded from 🔴 after judge-only re-scope) | Send the (cheap) authorization email anyway; framing = "LLM-as-judge reward signal, no output imitation". Zero-exposure fallback ready: open judge (Qwen3-235B Apache 2.0 / GLM-4.6 MIT) behind the same `JudgeClient` interface; self-STaR replaces any teacher-text warm-start. |
| 2 | No container runtime on the cluster; rootless docker needs admin | ✅ | Unprivileged userns is ENABLED on node08 → user-space apptainer install + Harbor's first-class singularity backend (docs/02 §2). Docker-fidelity cross-check runs off-cluster. Admin ask (uidmap) filed only if singularity's task-tail misbehaves. |
| 3 | **Terminal-Bench has no train split — RL on TB tasks = training on the test set** (canary-stringed, eval-only) | ✅ design / 🔶 work | Train on Endless Terminals (~2,500 Harbor tasks) + terminal-bench-rl (~331) + optional self-generated; dev split carved from training corpus; TB 2.1 held out for milestone evals only. |
| 4 | **GRPO cold start: E4B baseline ≈ 8% ⇒ most groups all-fail ⇒ zero gradient** | ✅ design | SFT warm-start (A1) is mandatory, not optional; DAPO dynamic sampling; difficulty curriculum ($\hat p \in [0.05,0.95]$); BREAD-style expert prefixes for stubborn tasks; LUFFY mixing on all-fail groups. |
| 5 | User premise "node08 = 2× A100" is false — **no A100s exist; node08 = 2× RTX A4000 16 GB** | ✅ | Plan re-sized: FP8 serve on GPU0, QLoRA train on GPU1, E2B fallback. Expectations re-anchored: prove the method, 8%→mid-teens is success. If real A100/H100 time becomes available later, the stack ports unchanged. |

## High

| # | Gap | Status | Action |
|---|---|---|---|
| 6 | Gemma 4 ecosystem sharp edges (PEFT `Gemma4ClippableLinear` crash, `mm_token_type_ids`, vLLM LoRA #39246 silently ignored, prime-rl grad-norm blowups) | 🔶 | Pinned stack + TRL merge-then-sync path chosen precisely to dodge vLLM-LoRA; **half-day smoke test is a hard gate** before any long run. |
| 7 | **Reward hacking**: >15% of TB2 verifiers bypassable; top leaderboard entries caught reading verifier code; RL finds exploits reliably. **With judge-shaped reward (docs/03 §4.4) there is a second surface: the judge reads terminal output the policy writes — a prompt-injection channel inside the reward function** (fake "tests passed" output, instructions to the judge in stdout). | 🔶 | Verifier side: isolated `[verifier.environment]` containers, artifact allowlists, network restrictions. Judge side: gated composite (judge can never outrank a real pass), provenance delimiters + "score actions/state, never claims" rubric, judge never sees verifier internals, judge-score-vs-pass-rate divergence alarm, spot-audit high-r_J failures. Keep ATIF trajectories for audit. |
| 8 | Harness/format coupling (eval harness ≠ train harness ≠ teacher format ⇒ no transfer / format drift) | ✅ design | One harness (Terminus 2) everywhere; teacher traces normalized into Terminus-2 JSON + Gemma 4 template before any SFT/mixing. |
| 9 | Tokenizer/template alignment rollout-vs-trainer + env-token loss masking | 🔶 | Token-in/token-out plumbing (TRL auto `tool_mask`); CI assertion: trainer recomputes rollout logprobs on a sample batch and must match; mask *all* terminal output tokens from loss and KL; append `<turn\|>` (not `<eos>`) in training data. |
| 10 | Rollout throughput is the binding constraint (not training FLOPs) — weeks of wall-clock | 🔶 | Small G (4–8), context cap 32k, warm .sif cache, ~6–8 concurrent containers; move to SkyRL async if stalled; accept modest target deltas. |
| 11 | Eval statistics: 89 binary tasks ⇒ ±6–8pp single-run CI swallows a +3–5pp effect | ✅ design | n≥5 seeds, paired per-task tests, pinned TB **2.1** (2.0 numbers not comparable), fixed agent config; also report pass@k on the dev set. |
| 12 | Context-length management: Terminus 2 summarization breaks naive GRPO credit assignment; truncation punished as failure | ✅ design | Default = DAPO overlong filtering (mask truncated/summarized episodes); `linear_history=True` splitting; SUPO if we later train through summarization. |

## Medium

| # | Gap | Status | Action |
|---|---|---|---|
| 13 | Teacher/judge cost. Judge-only design is ~10× cheaper than trace collection: compressed-trace scoring ≈ $0.03–0.06/call (Sonnet 5) or $0.01–0.02 (Haiku 4.5); full run O($100s). (Trace-collection numbers — ~$700/1k cached — kept in docs/02 §6 in case distillation extensions are ever authorized.) | 🔶 | Haiku for routine scoring + Sonnet for calibration audits; skip judging passed rollouts; cache rubric prefix; verify cache hits on a pilot. |
| 14 | Vanilla GRPO biases (length, difficulty, entropy collapse, KL drag) | ✅ design | DAPO clip-higher (0.2/0.28) + token-level norm + dynamic sampling; Dr.GRPO mean-only advantage; β=0 or ProRL-style reference resets; lr 1e-6, grad-clip 0.1, temp ~1.0–1.2. |
| 15 | Framework choice risk — no framework documents Gemma 4 GRPO end-to-end | 🔶 | TRL primary / SkyRL runner-up / rLLM alternative; decision gated on the smoke test (gap 6). |
| 16 | Cluster-local footguns (headnode rule, /local wiped on reprovision, NFS overlay, watcher for long runs) | ✅ design | Everything on node08; .sif cache + checkpoints on node-local disk with rsync to NFS; tmux + session-owned watcher for multi-day runs. |
| 17 | Baseline protocol underspecified (version/agent/sampling/thinking/parser) | ✅ design | docs/02 §7 pins all of it; validate the gemma4 reasoning parser on 2–3 easy tasks with transcript inspection first (a broken parser silently zeroes actions → spuriously low baseline). |

## Low / later

| # | Gap | Status | Action |
|---|---|---|---|
| 18 | Phase 2 (computer/VM GUI envs) is a different research problem | ✅ scoped | Explicit phase gate; keep env behind Harbor's abstraction (already has computer-use backends); don't promise it on 2×A4000. |
| 19 | Leaderboard submission integrity rules (post-scandal): ATIF trajectories + judge review; hacked verifiers ⇒ zeroed | ✅ design | We keep ATIF trajectories and run the internal audit anyway (same artifact as gap 7). |

## Added 2026-09-09 (found while building the code map / notebook)

| # | Gap | Status | Action |
|---|---|---|---|
| 20 | **Trainer tokens ≠ rollout tokens (gap 9 was real).** `06_grpo_micro.build_example` rendered `steps[].message` prose; no commands/observations/thinking in the G4 sequence; the Gemma-4 template merges consecutive same-role turns → one model turn, unstable prefix tokenisation. | 🔶 | `tarl.data.examples` renders the real alternating conversation (`terminus_chat`) with offsets-based masks; `08_grpo_run.py` defaults to it; verify `--collect-rollout-details` (vLLM `return_token_ids`) on node08 and add the trainer-vs-vLLM align check (notebook §11) as a startup gate. |
| 21 | **Judge scored claims, not actions.** The compressor sent narrative only, contradicting the rubric's "score actions, never claims". | 🔶 | `compress(include_actions=True, task_only=True)` is the default for new runs; re-run G2/G3 in that mode; keep legacy rendering only for replaying historical verdicts. |
| 22 | Judge scores were never persisted per trial (only `g4.log` stdout). | ✅ | `scripts/08_grpo_run.py` writes `rollouts.jsonl` with score + rationale + prompt hash; `experiments/derived/g4-replay.jsonl` reconstructs the G4 verdicts (order-inferred). |
