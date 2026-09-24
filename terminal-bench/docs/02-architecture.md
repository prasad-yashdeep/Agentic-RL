# System Architecture

All external facts below were verified against current sources on 2026-07-20
(research sweep; URLs in the per-section notes).

## 0. The model — corrected identity

"Gemma 4 E4B IT – Thinking" = **`google/gemma-4-E4B-it`** (released
2026-04-02, **Apache 2.0**). There is **no separate thinking checkpoint** —
thinking is a runtime mode of the single IT model:

| Fact | Value |
|---|---|
| Params | **4.5B effective / ~8B total** (per-layer embeddings inflate total) |
| Context | 128K |
| Chat template | new format: `<\|turn\|>user … <turn\|>` turns; assistant role = `model`; **IT model ends turns with `<turn\|>`, not `<eos>`** (training data must match) |
| Thinking | enable via `<\|think\|>` in the system turn / `enable_thinking=True`; reasoning is emitted in `<\|channel\|>thought … <channel\|>` spans (NOT `<thinking>` XML) |
| Tool calls | `<\|tool_call\|>call:fn{args}<tool_call\|>` / `<\|tool_response\|>…` |
| Published Terminal-Bench score | **E4B = 8.0** (tech report; E2B 3.0, 12B 18.0) |
| Sampling (recommended) | temp 1.0, top_p 0.95, top_k 64 |

Pinned stack: `transformers ≥ 5.5.4`, current stable vLLM with
`--reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice
--chat-template tool_chat_template_gemma4.jinja --limit-mm-per-prompt image=0,audio=0`.

Known sharp edges (all confirmed in the wild):
- **PEFT bug:** `Gemma4ClippableLinear` (vision/audio encoders) crashes LoRA
  targeting — list text-decoder linear names explicitly in `target_modules`.
- Collator must emit `mm_token_type_ids` even for text-only batches.
- **vLLM LoRA for Gemma 4 is an open feature request (#39246)** — adapters can
  be silently ignored. ⇒ choose a trainer that **merges LoRA into weights
  before syncing to vLLM** (TRL does; verl/SkyRL's `lora_request` path doesn't).
- prime-rl reported million-scale grad norms on Gemma 4 SFT — grad-clip early,
  smoke-test before long runs.
- On our 16 GB A4000s: bf16 weights (~16 GB) don't fit with KV — serve **FP8**
  (`prithivMLmods/gemma-4-E4B-it-FP8`) or 8-bit (9–12 GB) on GPU0; train QLoRA
  4-bit (~6 GB base) on GPU1.

## 1. Benchmark & harness

- **Eval target: Terminal-Bench 2.1** (bugfix of TB 2.0's 89 tasks; 28 tasks
  repaired; scores NOT comparable across versions — pin 2.1). Runs on
  **Harbor** (`uv tool install harbor`), the official harness:
  provision env → run agent → run verifier; verifier writes
  `/logs/verifier/reward.txt` (0/1).
- **Agent = Terminus 2** (reference agent): one persistent tmux session, model
  emits strict JSON (`analysis`, `plan`, `commands`, `task_complete`) →
  `tmux send-keys`; built-in context summarization; `max_turns=100`;
  **`collect_rollout_details` + `TrajectoryConfig(raw_content=True,
  linear_history=True)` exist specifically for RL/SFT data collection.**
- Self-hosted model plumbing: Harbor talks LiteLLM — `--model
  openai/<served-name>` + `api_base` pointing at our vLLM server.
- Reference points on TB 2.0: Terminus 2 + Opus 4.6 = 62.9%; untuned 4–9B
  open models = 0–10% (Qwen3.5-9B 9.2%); RL/SFT-specialized 8–9B = 13–27%
  (Tmax 9B = 27%, Nemotron-Terminal-8B = 13%). Sets realistic expectations:
  **8.0 → mid-teens is a strong outcome for E4B at our compute.**

**Harness decision:** Terminus 2 everywhere (baseline eval, teacher traces, RL
rollouts, final eval). Score is a property of (model, harness) *jointly* —
mixing harnesses between train and eval destroys transfer. Mini-SWE-Agent kept
as a sanity-check second harness for the baseline only.

## 2. Container runtime on node08 (solved, zero admin)

Probes (2026-07-20): no docker/podman/apptainer anywhere; no uidmap binaries;
BUT **unprivileged user namespaces are enabled on node08**
(`kernel.apparmor_restrict_unprivileged_userns = 0`, `unshare --user` works).

⇒ Install **apptainer unprivileged** (official `install-unprivileged.sh` into
`~/apptainer`) and use **Harbor's first-class Singularity backend**
(`-e singularity`, NVIDIA-contributed, built "for HPC clusters (SLURM node)"):
pulls each task's pinned `docker_image`, converts to `.sif` (file-locked
cache), runs a FastAPI exec server inside the instance.

```bash
harbor run -d terminal-bench/terminal-bench-2 -a terminus-2 \
  -m openai/gemma-4-E4B-it -e singularity \
  --ek singularity_image_cache_dir=/tmp/sif-cache \
  --ek singularity_no_mount=home,tmp   # workaround issue #1712 (DNS dead if resolv.conf stripped)
```

- Pin harbor ≥ 0.20; known open issues: #1712 (DNS, workaround above), #1713
  (60s health-check too short; fix PR #1722); expect a small tail of
  fakeroot-incompatible tasks.
- Singularity shares the host network namespace → agent containers reach vLLM
  at `127.0.0.1:8000` directly (an advantage over rootless docker, which
  blocks loopback).
- **Fidelity cross-check:** before reporting final numbers, re-run the eval
  once on a docker machine (laptop; harness is CPU-only, vLLM reached via
  `ssh -L`) to confirm singularity-backend scores match docker-backend scores.
- Fallback if singularity misbehaves: one-time admin ask = `apt install uidmap`
  + subuid/subgid entries → rootless docker fully self-service after that.

## 3. Training data — never train on the eval set

**Terminal-Bench has NO train split** (canary strings in every file; the HF
"train" split is a packaging artifact). Training on TB tasks = training on the
test set. Training corpus instead:

1. **Endless Terminals** (arXiv:2601.16443) — ~2,500 procedurally generated
   tasks already in Harbor format. Primary corpus.
2. **terminal-bench-rl** (Danau5tin) — ~331 synthetic tasks w/ weighted pytest
   verifiers. Secondary.
3. Optionally: self-generated tasks (Tmax-style taxonomy generation,
   arXiv:2606.23321) — this doubles as a good use of the Sonnet 5 budget.
4. Carve a **dev split** (~10%) from the training corpus for model selection;
   TB 2.1 is touched only for milestone evals.

Harden training verifiers: isolated `[verifier.environment]` containers with
explicit artifact allowlists (tests never visible to the agent),
`network_mode` restrictions. >15% of TB2 verifiers were shown bypassable —
RL *will* find exploits that eval agents didn't.

## 4. Training stack

**Primary: HF TRL v1.8 `GRPOTrainer` + `trl.experimental.harbor`.** Rationale:
- Only stack where all constraints line up: transformers-native training
  (best Gemma 4 odds), PEFT QLoRA **without** depending on vLLM's incomplete
  Gemma-4 LoRA (adapters merged before weight sync), automatic token-level
  `tool_mask` over env tokens, Harbor task suites as `environment_factory`
  (docker + e2b/daytona backends), documented 1-GPU colocate and 2-GPU
  server modes (GPU0 `trl vllm-serve`, GPU1 trainer) — exactly our topology.
- Requires: `vllm ≥ 0.22`, `transformers ≥ 5.2` (we pin 5.5.4), Python ≥ 3.12.
- Known gap: `trl.experimental.harbor`'s sandbox backend list is docker-first;
  the singularity path may need a thin `HarborEnv` subclass that shells out to
  our apptainer-backed Harbor. Budget 1–2 days.

**Runner-up:** SkyRL (first-party Harbor terminal integration, async rollouts)
if rollout throughput becomes the bottleneck; rLLM if we adopt an existing CLI
harness wholesale (terminal-bench-rl precedent, 2×A100 tested minimum — note
that was for Qwen3-32B; E4B on 2×A4000 is proportionally similar).

**Gate:** half-day smoke test before committing — Gemma 4 E4B QLoRA
forward/backward in transformers + merged-weight sync into vLLM + grad-norm
sanity (prime-rl saw million-scale norms on Gemma 4). No framework documents
Gemma 4 GRPO end-to-end; assume nothing.

## 5. Node08 runtime topology

```
┌──────────────────────────── node08 (24 CPU / 62 GB RAM) ───────────────────────────┐
│                                                                                    │
│  GPU0 (A4000 16GB)                     GPU1 (A4000 16GB)                           │
│  ┌──────────────────────┐              ┌──────────────────────┐                    │
│  │ vLLM OpenAI server   │   adapter    │ TRL GRPOTrainer      │                    │
│  │ gemma-4-E4B-it FP8   │◄─merge+sync──│ QLoRA 4-bit base     │                    │
│  │ :8000                │              │ tool-masked loss     │                    │
│  └─────────▲────────────┘              └─────────▲────────────┘                    │
│            │ HTTP (localhost)                    │ trajectories + rewards          │
│  ┌─────────┴────────────────────────────────────┴─────────────┐                    │
│  │ Harbor + Terminus 2   (-e singularity, ~6-8 concurrent)    │                    │
│  │ apptainer instances: task containers (1-2 CPU, 2-4GB each) │                    │
│  │ verifier → reward.txt                                      │                    │
│  └────────────────────────────────────────────────────────────┘                    │
│  NCCL note: never TP=2 without NCCL_P2P_DISABLE=1 (broken PCIe P2P)                │
└────────────────────────────────────────────────────────────────────────────────────┘
         ▲ SSH port-forward (baseline/fidelity checks from laptop docker)
```

Throughput realism: episodes are 10–100 model calls; at ~6–8 concurrent
containers + one 16 GB serving GPU, **one GRPO step (e.g. 8 tasks × 8
rollouts) will take tens of minutes**. This is a weeks-scale experiment at
modest scale — sized to prove the method, not chase the leaderboard.

## 6. Teacher pipeline (Sonnet 5)

**⚠ Legal gate first:** Anthropic's Usage Policy prohibits "utilization of
inputs and outputs to train an AI model … without prior authorization", and
Commercial ToS §D.4 restricts training competing models. Distilling Sonnet 5
traces into Gemma **requires prior written authorization from Anthropic**
(academic research requests are granted — ask before collecting). The
architecture keeps the teacher behind a `TeacherClient` interface so an open
teacher (DeepSeek-R1/V3 — MIT, explicitly blesses distillation; Qwen3-235B —
Apache 2.0; GLM-4.6 — MIT) can be swapped in; open teachers also expose
logprobs, unlocking on-policy distillation later.

Collection mechanics (once authorized):
- Run `harbor run … -a terminus-2 -m anthropic/claude-sonnet-5` on the
  **training corpus** with `TrajectoryConfig(raw_content=True,
  linear_history=True)`.
- **Enable prompt caching via LiteLLM `cache_control`** (not automatic!) and
  verify `usage.cache_read_input_tokens > 0` on a pilot; watch the 20-block
  lookback in long tool loops. Set thinking `effort=low/medium` (default
  `high` inflates output 1.5–2×).
- Cost model (verified pricing $3/$15, intro $2/$10 through 2026-08-31):
  ~$0.31–1.13/trajectory cached ⇒ **~$700 per 1,000 (standard) / ~$460
  (intro)**. Pilot 50 traces first; measure teacher pass-rate (only *passing*
  traces are usable — if Sonnet solves 60%, effective cost ÷0.6).
- Post-process: keep verifier-passing episodes → normalize into **Terminus 2's
  exact JSON action format and Gemma 4's chat template** (teacher formatting ≠
  student formatting; unnormalized traces teach format drift) → retokenize
  with the Gemma tokenizer (LUFFY's π_φ≡1 handles the cross-tokenizer case by
  construction).

## 7. Evaluation protocol (pre-registered)

- TB **2.1**, Terminus 2, fixed sampling (t=1.0/top_p=0.95/top_k=64), fixed
  `max_turns`, record harness+model+parser versions with every run.
- Baselines both thinking-on and thinking-off (thinking tokens eat context in
  100-turn episodes — may hurt).
- **n ≥ 5 seeds per condition**; report mean ± CI **and paired per-task
  McNemar/bootstrap** (89 binary tasks: a single run's CI is ±6–8pp — a +3–5pp
  RL gain is invisible without repeats).
- Keep full ATIF-format trajectories + run an internal judge audit for reward
  hacking before believing (or submitting) any number.

## 8. Long-term generalization

Everything env-specific sits behind Harbor's env abstraction, which already
supports computer-use backends (`use-computer`, `cua-cloud`). Phase 2
(GUI/VM, OSWorld-class) is a *separate research problem* — vision-token
training, VM snapshotting, much more compute — gated on phase 1 results, not
promised within the 2×A4000 budget.
