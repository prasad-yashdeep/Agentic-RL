# 02 — Phase 2 roadmap

Phase 1 proved the plumbing on node08 (gates S0–S8, `experiments/EXPERIMENTS.md`):
- vLLM serves πθ;
- the harness is token-exact;
- apptainer runs the SWE-bench sandboxes, offline;
- the reward is the real tests, graded from a read-only input in a fresh container;
- the LoRA GRPO update is synced back into vLLM.

An adversarial review of `sarl` followed (15 findings: 14 confirmed, 13 of them fixed in commit `0801504`), and the final S8 run `s8-basic-v3b` repeated the loop with every fix: oracle 20/20, 0 degenerate groups, 0 errors.

It produced **no learning signal from the policy itself**. Qwen2.5-Coder-0.5B resolved 0 of 140 rollouts in each S8 run, it does not follow the diagram's JSON tool call (0/48), and every non-degenerate group came from the scripted oracle. Phase 2 makes the policy earn reward and measures it on held-out data.

## Done in phase 1 (no longer on this roadmap)

| Item | Where it landed |
|---|---|
| Re-run S8 after the LANG fix | `s8-basic-v2` (reference sampling, network on), then `s8-basic-v3b` (all review fixes): 0 degenerate groups in both |
| Sample from the distribution the loss scores | Plain sampling by default; `policy.REFERENCE_SAMPLING` keeps the reference's settings |
| No network in the sandbox | `--network none` for rollouts and grading; pip and conda fail fast; pool gate re-verified 8/8 offline |
| Cleanup of `/var/tmp/swe/grade/*` | Workdirs deleted after each grade (`keep=True` keeps one); logs capped at 2 MB |
| Seeded vLLM requests | Torch seed plus a per-request vLLM seed per rollout |
| Per-token trajectories off NFS | gzip'd to node-local `/var/tmp/swe/trajectories/<run>/` |
| Integrity fixes from the review | Read-only grading input, harness-file filter, patch against the pinned HEAD, invalid rollouts out of groups, special-token sanitising, pooled mismatch, sandbox sweeps (`EXPERIMENTS.md`) |
| Bound one straggler's cost | `basic_run.yaml`: `command_timeout_s: 60`, `wall_clock_s: 600` (was 120 s and 30 min) |

## Constraints that shape every item

- **One compute node.** node08 is the only compute node up (2026-09-24), with 2× A4000 16 GB. PCIe P2P between them is broken, so any 2-GPU collective needs `NCCL_P2P_DISABLE=1`, and vLLM TP=2 also needs `--disable-custom-all-reduce`.
- **GPU sharing with terminal-bench.**
  - terminal-bench's experiment driver (`make tb-exp`) uses GPU0 (vLLM gemma FP8, :8000) and GPU1 (trainer).
  - swe-bench uses GPU0 (:8001) and GPU1.
  - The two cannot co-run, so they take turns in blocks.
  - Before starting either: `make -C swe-bench status`, then stop `swe-serve` before `tb-exp`. terminal-bench's A0/B1/B2 driver is next on node08, now that phase 1 is done.
- **Disk.**
  - `/tmp` is 16 GB: about 12–14 SWE-bench images of roughly 1 GB each.
  - `/var/tmp` is 39 GB and shared: terminal-bench's `/var/tmp/tarl` plus `/var/tmp/swe` (grader venv, adapters, gzip'd per-token trajectories; grading workdirs are deleted after each grade).
  - NFS `/home` is 92 % full (63 GB free), and large files never go there.
  - node05's `/local` (2.43 TiB) would lift the image limit once node05 is back.
- **Reprovisioning.** A reprovision wipes node-local state, as happened to node08 between 2026-09-08 and 09-22. After one: `make sifs grader` rebuilds `/tmp/swe-sif` and `/var/tmp/swe`.

## Priorities

| P | Item | Why at this priority | GPU |
|---|---|---|---|
| P0 | S9: the policy follows the tool-call protocol (SFT warm start or a larger model) | Without it, the policy never earns reward | yes |
| P0 | S10: Verified eval on a fixed hashed subset | Nothing is measurable without it. The gold-control half is CPU-only and can run while the GPUs are busy. | partly |
| P1 | Bigger, cleaner training pools with a leakage guard | 8 tasks is too few for RL | no (disk) |
| P1 | Token-level clipped surrogate with importance ratios | Needed before mixing in SFT or oracle data, async rollouts, or a quantized server | small |
| P1 | Async rollouts, straggler handling | The synchronous step waits for its slowest rollout: in `s8-basic-v3b`, one episode held step 1 for 1,806 s against a 2.3 s median episode. The 60 s / 600 s limits bound this but do not remove it, and it grows with model size and episode length. | yes |
| P2 | Context management | First S8 run: 8/140 context exits, 8 dropped trajectories. Final run: 0 context exits, 4 dropped. | no |
| P2 | Partial-credit shaping | Only after a binary signal exists | no |
| P3 | Shared `arl` core with terminal-bench | After terminal-bench's driver lands | no |

## S9 — the diagram's tool-call protocol

**Goal:** the policy emits `<tool_call>{"name":"bash","arguments":{"command":…}}</tool_call>` reliably and explores before submitting.

**Gate (proposed).** Re-run the S6 probe (16 episodes with the `tool_call` protocol on the dev pool) and require:
- ≥ 90 % valid calls (S6: 0/48);
- ≤ 25 % of episodes submitting on turn 1 (S6 with backticks: 14/16; final S8 run with backticks: 27/140);
- then a 10-step run with `protocol: tool_call` in which ≥ 1 non-oracle rollout earns reward.

### Option A: SFT warm start (keeps a small model)

- **Data.** Filter trajectories rendered by *our* harness with `protocol="tool_call"`, so the tokens are exactly the ones RL will later see.
  - (a) Oracle trajectories with exploration: prepend read steps to `harness.oracle_replies` (`ls`, `grep -rn` on a symbol from the issue, `sed -n` on the gold-patch file) before the write/apply/submit. The current oracle is only 3 actions and teaches "paste the fix".
  - (b) Stronger-model trajectories: serve a stronger model (Option B) on GPU0 and run `sarl episode … --policy vllm` on dev tasks **outside** the RL pool. Keep the reward-1 episodes (rejection sampling, as SWE-Gym and R2E-Gym did) plus format-valid episodes that explore.
- **Training.** Reuse `LoRATrainer` with A = 1 on the same ids/mask, which is masked log-likelihood. The loss path is then identical to RL.
- **Risk.** SFT on the pool's own gold patches leaks the answers into RL. Keep the SFT tasks disjoint from the RL pool, and record both lists in `experiments/runs.yaml`.

### Option B: a larger model

The 0.5B figures were measured at S6. Every other row is an estimate from the model configs and has not been measured. KV bytes per token = 2 × layers × KV heads × head_dim × 2 bytes. The serve budget is GPU0 at `--gpu-memory-utilization 0.85`, about 13.6 GiB.

| Model | Weights | KV bytes/token | Serve on GPU0 (est.) | Train on GPU1 (est.) |
|---|---|---|---|---|
| Qwen2.5-Coder-0.5B-Instruct bf16 | 0.96 GiB (measured) | 12,288 (24 × 2 × 64) | 11.99 GiB KV = 1.05M tokens (measured) | LoRA bf16 at 8k tokens (done, S8) |
| Qwen2.5-Coder-1.5B-Instruct bf16 | about 3.1 GB | 28,672 (28 × 2 × 128) | about 9 GiB KV ≈ 0.33M tokens: G = 16 × 16k fits | LoRA bf16 with checkpointing and chunked logits, 16k tokens plausible |
| Qwen2.5-Coder-3B-Instruct bf16 | about 6.2 GB | 36,864 (36 × 2 × 128) | about 6 GiB KV ≈ 0.17M tokens | LoRA bf16, 8k tokens plausible |
| Qwen2.5-Coder-7B-Instruct bf16 | about 15.2 GB | 57,344 (28 × 4 × 128) | does not fit one card; TP=2 would take both GPUs and leave none for training | — |
| Qwen2.5-Coder-7B-Instruct-AWQ (int4) | about 5.6 GB | 57,344 | about 6.5 GiB KV ≈ 0.12M tokens: about 4 × 32k or 16 × 7k | QLoRA nf4 (bitsandbytes 0.49.2 is in the venv): about 5 GB base plus activations, 8k tokens plausible |

**The 7B catch.** The served base (AWQ) and the trained base (nf4) are different quantizations, so the train–inference mismatch will be much larger than the final S8 run's (pooled mean 0.0018–0.0077, max 0.401). Measure it with the existing `mismatch` metric before any 7B RL. Ratios (P1) become mandatory. Also check vLLM 0.25.1 runtime LoRA on an AWQ base.

**Published context** (see the research notes behind `01-lecture-map.md`):
- Nothing is published for 0.5B or 1.5B on SWE-bench Verified.
- Qwen2.5-Coder-7B scores 1.8 % base (OpenHands scaffold). After SFT or RL it reaches 10.6 % (SWE-Gym), 19.0 % (R2E-Gym) and 23.4 % (SWE-Dev).

**Order.**
1. Qwen2.5-Coder-1.5B-Instruct in bf16 first. It has the same tokenizer and stop ids and the same exact-bf16 code path. The model id and revision are pinned in three places, `paths.py`, `basic_run.yaml` and `20_serve_policy.sh`; pin the new revision there, as at S0.
2. Add SFT (Option A) if its valid-call rate is below 90 %.
3. 7B-AWQ as the stretch.

## S10 — Verified eval on a fixed hashed subset

- **Subset.**
  - Take `SWE-bench/SWE-bench_Verified@78f471bf` (500 tasks), excluding `astropy__astropy-12907` (used by the phase-1 replica).
  - Sort by `sha256("sarl-eval-v1:" + instance_id)` and take the first N: N = 50 first, then 100.
  - Write `configs/eval_verified_v1.txt` with the salt and N in its header. Never change it once results exist; a new subset gets a new salt.
- **Gold control.** Run `scripts/pool_gate.py` on the eval list. Every task must score gold = 1 and empty = 0 under apptainer, with 100 % as the target.
  - Replace failing tasks deterministically with the next hash and report them. Do not silently drop them.
  - This half needs no GPU.
- **Disk.** One image per instance, about 1 GB each (Verified images for some repos may be larger; measure). Stream in batches of ≤ 8: pull, gold/empty control, episodes, grade, delete the SIF. Or use node05's `/local` once it is up.
- **Metric.**
  - pass@1 over tasks, with the same harness, protocol and budgets as training, and k samples per task (k = 1 first).
  - Report Wilson 95 % intervals, and paired base-vs-trained comparisons on identical tasks (McNemar or a paired bootstrap).
  - Also report the valid-call rate, exit reasons and the turn-1 submit rate.
- **Resolution.** 0/50 → [0, 7.1] %; 0/100 → [0, 3.7] %; 5/100 → [2.2, 11.2] %. A small model will likely read 0, and the interval says what that rules out.
- **Guard.** `contamination_report(train_pool, eval_subset)` must be clean, and Verified tasks never enter a training pool.

## Bigger, cleaner training pools

| Source | Size | Images | Fit on node08 | Notes |
|---|---|---|---|---|
| SWE-bench dev (current split) | 225 tasks, 6 repos: pvlib 63, pydicom 56, sqlfluff 50, astroid 31, pyvista 16, marshmallow 9. 124 have a single-file gold patch. `select_pool` defaults (≤ 700 chars, 1–3 F2P, ≥ 1 P2P) give 26; ≤ 2,000 chars gives 67. | One per instance, about 1 GB | About 12–14 at a time in `/tmp`: rotate SIFs per step window, or use `/var/tmp` | Already disjoint from Verified |
| SWE-smith (MIT) | 50k+ tasks, 128 repos, mostly synthetic single-function bugs | **One per repo**, about 1.3 GB compressed | A handful of SIFs covers hundreds of tasks | Best disk economy. Decontamination is not confirmed: drop the 12 Verified repos. |
| SWE-Gym (MIT) | 2,438 tasks, 11 repos; Lite 230 | One per instance, about 1.2 GB compressed | Same limit as dev | No repo overlap with SWE-bench |
| R2E-Gym-Subset (Apache-2.0) | 4,578 tasks, 10 repos | One per commit | Same limit as dev | Decontaminated (SWE-bench repos removed); used by DeepSWE |

The sizes come from the HF and Docker Hub metadata in the research notes and have not been re-checked here.

**Rules for every new pool:**
1. **Structural guard.** `tasks.contamination_report` against Verified (repo, instance_id, base_commit, plus the 12 `VERIFIED_REPOS`) must be clean.
2. **Semantic guard.** Check for near-duplicate problem statements against Verified (a MiniLM check like terminal-bench's `tarl data et leakage`). `tasks.py` mentions a `scripts/pool_guard.py` for this, but it does not exist yet.
3. **Pool gate.** `scripts/pool_gate.py` must show gold = 1 and empty = 0 under apptainer, graded offline (`--network none`, the default) with the empty patch's tests actually run (this is how the LANG bug was found).
4. **Difficulty filter.** Pre-filter by pass rate (lecture 2:00–2:02) with the S9 model or a stronger one. Drop tasks at 0/G and G/G, so groups are informative without the oracle.
5. **Versioning.** Version the pool file and record its guard results in `experiments/`.

A dataset-agnostic `Task` loader is needed for SWE-smith, SWE-Gym and R2E-Gym (different image naming and eval scripts). The grader must still produce `resolved` from real tests.

## Loss: token-level clipped surrogate with importance ratios

- The behaviour log-probs are already stored per sampled token (`Trajectory.logprobs`). Use the ratio `ρ_t = exp(log πθ − log π_behaviour)` with a clipped surrogate. ε_low 0.2 / ε_high 0.28 matches terminal-bench's plan.
- Normalise at token level rather than sequence mean, and allow several minibatch passes per rollout batch.
- Add truncated importance sampling for the train–inference mismatch (lecture 2:31). In the final S8 run the pooled |Δlogp| had p99 ≤ 0.114 and max 0.401, with no token above 0.5; that is small at 0.5B bf16, and it will not be for an AWQ server.
- Sampling already matches the scored distribution (plain sampling, raw log-probs). If truncated sampling (`REFERENCE_SAMPLING`) is ever needed again, request processed log-probs (check vLLM 0.25.1's `logprobs-mode`) so the ratios use the real behaviour policy.
- **Oracle and SFT data.** These have no behaviour log-probs. Either drop the oracle once policy groups are informative, or score its tokens under the current weights with vLLM `prompt_logprobs` and cap the ratio.
- **Tests.** Ratio = 1 must reduce to the current REINFORCE loss, and clipping must bound the update. terminal-bench has such a test; copy the pattern, not the import.

## Context management

The first S8 run had 8 of 140 episodes end at the 16k context, 8 trajectories dropped as longer than 8,192 training tokens, and a longest trajectory of 18,473 tokens (one observation past the budget). The final run had 0 context exits, 4 dropped trajectories and a longest trajectory of 14,657 tokens.
- At 80 % of the context, inject one "finalize and submit now" user turn (mask 0), as in abgoswam's `miniswe_agent_loop` notes.
- Reject a turn whose observation would overflow the budget, instead of appending it.
- Compact filtering (DeepSWE): mask the loss on trajectories that end on context, turn or wall-clock limits, rather than training them as failures.
- Raise `max_train_tokens` to 16k once the trainer's memory at 16k has been measured on GPU1.

## Async rollouts and stragglers

- **The straggler finding.** In `s8-basic-v3b`, step 1 took 1,818 s. One policy rollout on pydicom-1694 ran `pip install --upgrade pydicom -r …/requirements.txt` followed by `pytest /testbed` on 15 turns. pip failed fast offline, but the full pydicom suite takes more than 120 s, so every command hit the 120 s timeout, until the 30-min episode wall clock ended the episode at 1,806 s. The median episode took 2.3 s, and the other nine steps took 18–46 s.
- **Mitigation done.** `basic_run.yaml` now sets `command_timeout_s: 60` and `wall_clock_s: 600`, which bounds one straggler to about 10 minutes but does not remove the wait.
- In the final run, everything after grading (diagnostics, update, sync) took 4.2–7.4 s per step (total − rollouts − grading), so the loop is bounded by its slowest episode, not by training.
- Next: start grading each rollout as soon as it finishes (per-episode futures) rather than after the whole batch.
- Straggler handling: a step-level deadline, after which unfinished rollouts are stopped and left out of their groups, as crashed rollouts already are. This biases groups against long episodes, so log how many are cut per step.
- Then: generate step k+1's rollouts while step k trains, with staleness ≤ 1 step. This requires the ratios above; rLLM's `harbor_swe` example runs with staleness 0.5.
- The gain grows with model size and episode length.

## Partial-credit shaping (lecture 2:47)

- Reward = 1 if resolved; otherwise λ·s, with λ between 0.1 and 0.2, and only when every PASS_TO_PASS test still passes (never reward breaking things).
- s is the fraction of FAIL_TO_PASS tests passing. The grader already reports `f2p_passed` and `f2p_total`. Every current pool task has exactly 1 F2P test, so this term is binary there.
- The alternative s is the file or hunk overlap of the candidate patch with the gold patch's files.
- Watch for hacking: touching the right file without fixing it. Log the shaped and binary rewards separately. Eval stays binary.

## Shared `arl` core with terminal-bench

The shared pieces are group advantages, degenerate handling, masked and chunked token log-probs, the LoRA trainer, mismatch stats, JSONL metrics and Wilson CIs.
- Extract them into a core package only **after** terminal-bench's A0/B1/B2 driver (next on node08) has run, so its code does not move mid-experiment.
- Extend `tools/check_layering.py`: `tarl` → `arl` and `sarl` → `arl` are allowed; `tarl` ↔ `sarl` is still forbidden. Until then, nothing is copied between the two packages.

## Housekeeping

- `sarl episode` has no `--protocol` flag; it always uses `tool_call`. Add one before the S9 probe.
- Grading workdirs are now always deleted unless `keep=True`. Consider keeping those of resolved and infra-failed grades for inspection.
- Seeds are set, but no run has been repeated to check how reproducible it is under concurrent vLLM batching.
