# Gaps & Risks Register

From an adversarial review of the plan against the research findings
(2026-07-20). Status: ✅ resolved in architecture · 🔶 mitigation designed,
work pending · 🔴 open — needs a decision/action before the affected phase.

## Blockers

| # | Gap | Status | Action |
|---|---|---|---|
| 1 | **Anthropic ToS: using Sonnet 5 outputs to train Gemma is "model distillation" — prohibited *without prior authorization*** (Usage Policy "Do Not Abuse our Platform"; Commercial ToS §D.4). Applies to SFT, trace-mixing, and arguably judge-in-the-loop rewards. | 🔴 | **Email Anthropic for written research authorization BEFORE collecting any traces.** Parallel hedge: keep `TeacherClient` swappable to DeepSeek-R1/V3 (MIT, explicitly permits distillation), Qwen3-235B (Apache 2.0), GLM-4.6 (MIT) — all also expose logprobs, which *upgrades* the available math (on-policy distillation). |
| 2 | No container runtime on the cluster; rootless docker needs admin | ✅ | Unprivileged userns is ENABLED on node08 → user-space apptainer install + Harbor's first-class singularity backend (docs/02 §2). Docker-fidelity cross-check runs off-cluster. Admin ask (uidmap) filed only if singularity's task-tail misbehaves. |
| 3 | **Terminal-Bench has no train split — RL on TB tasks = training on the test set** (canary-stringed, eval-only) | ✅ design / 🔶 work | Train on Endless Terminals (~2,500 Harbor tasks) + terminal-bench-rl (~331) + optional self-generated; dev split carved from training corpus; TB 2.1 held out for milestone evals only. |
| 4 | **GRPO cold start: E4B baseline ≈ 8% ⇒ most groups all-fail ⇒ zero gradient** | ✅ design | SFT warm-start (A1) is mandatory, not optional; DAPO dynamic sampling; difficulty curriculum ($\hat p \in [0.05,0.95]$); BREAD-style expert prefixes for stubborn tasks; LUFFY mixing on all-fail groups. |
| 5 | User premise "node08 = 2× A100" is false — **no A100s exist; node08 = 2× RTX A4000 16 GB** | ✅ | Plan re-sized: FP8 serve on GPU0, QLoRA train on GPU1, E2B fallback. Expectations re-anchored: prove the method, 8%→mid-teens is success. If real A100/H100 time becomes available later, the stack ports unchanged. |

## High

| # | Gap | Status | Action |
|---|---|---|---|
| 6 | Gemma 4 ecosystem sharp edges (PEFT `Gemma4ClippableLinear` crash, `mm_token_type_ids`, vLLM LoRA #39246 silently ignored, prime-rl grad-norm blowups) | 🔶 | Pinned stack + TRL merge-then-sync path chosen precisely to dodge vLLM-LoRA; **half-day smoke test is a hard gate** before any long run. |
| 7 | **Reward hacking**: >15% of TB2 verifiers bypassable; top leaderboard entries caught reading verifier code; RL finds exploits reliably | 🔶 | Isolated `[verifier.environment]` containers, artifact allowlists, network restrictions on training tasks; spot-audit passing trajectories (judge screen); treat suspiciously fast reward gains as a hacking alarm; keep ATIF trajectories for audit. |
| 8 | Harness/format coupling (eval harness ≠ train harness ≠ teacher format ⇒ no transfer / format drift) | ✅ design | One harness (Terminus 2) everywhere; teacher traces normalized into Terminus-2 JSON + Gemma 4 template before any SFT/mixing. |
| 9 | Tokenizer/template alignment rollout-vs-trainer + env-token loss masking | 🔶 | Token-in/token-out plumbing (TRL auto `tool_mask`); CI assertion: trainer recomputes rollout logprobs on a sample batch and must match; mask *all* terminal output tokens from loss and KL; append `<turn\|>` (not `<eos>`) in training data. |
| 10 | Rollout throughput is the binding constraint (not training FLOPs) — weeks of wall-clock | 🔶 | Small G (4–8), context cap 32k, warm .sif cache, ~6–8 concurrent containers; move to SkyRL async if stalled; accept modest target deltas. |
| 11 | Eval statistics: 89 binary tasks ⇒ ±6–8pp single-run CI swallows a +3–5pp effect | ✅ design | n≥5 seeds, paired per-task tests, pinned TB **2.1** (2.0 numbers not comparable), fixed agent config; also report pass@k on the dev set. |
| 12 | Context-length management: Terminus 2 summarization breaks naive GRPO credit assignment; truncation punished as failure | ✅ design | Default = DAPO overlong filtering (mask truncated/summarized episodes); `linear_history=True` splitting; SUPO if we later train through summarization. |

## Medium

| # | Gap | Status | Action |
|---|---|---|---|
| 13 | Teacher trace cost/mechanics unbudgeted (caching not automatic; thinking effort inflates cost; only *passing* traces usable) | 🔶 | LiteLLM `cache_control` + verify cache hits on a 50-trace pilot; effort=low/medium; ~$700/1k standard, ~$460 intro (through 2026-08-31); budget ÷ teacher pass-rate. |
| 14 | Vanilla GRPO biases (length, difficulty, entropy collapse, KL drag) | ✅ design | DAPO clip-higher (0.2/0.28) + token-level norm + dynamic sampling; Dr.GRPO mean-only advantage; β=0 or ProRL-style reference resets; lr 1e-6, grad-clip 0.1, temp ~1.0–1.2. |
| 15 | Framework choice risk — no framework documents Gemma 4 GRPO end-to-end | 🔶 | TRL primary / SkyRL runner-up / rLLM alternative; decision gated on the smoke test (gap 6). |
| 16 | Cluster-local footguns (headnode rule, /local wiped on reprovision, NFS overlay, watcher for long runs) | ✅ design | Everything on node08; .sif cache + checkpoints on node-local disk with rsync to NFS; tmux + session-owned watcher for multi-day runs. |
| 17 | Baseline protocol underspecified (version/agent/sampling/thinking/parser) | ✅ design | docs/02 §7 pins all of it; validate the gemma4 reasoning parser on 2–3 easy tasks with transcript inspection first (a broken parser silently zeroes actions → spuriously low baseline). |

## Low / later

| # | Gap | Status | Action |
|---|---|---|---|
| 18 | Phase 2 (computer/VM GUI envs) is a different research problem | ✅ scoped | Explicit phase gate; keep env behind Harbor's abstraction (already has computer-use backends); don't promise it on 2×A4000. |
| 19 | Leaderboard submission integrity rules (post-scandal): ATIF trajectories + judge review; hacked verifiers ⇒ zeroed | ✅ design | We keep ATIF trajectories and run the internal audit anyway (same artifact as gap 7). |
