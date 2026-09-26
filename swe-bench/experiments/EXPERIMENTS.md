# swe-bench experiment log

Newest first. Every entry names:
- the run dir (under the gitignored `runs/`) and the committed summary, if any;
- the config;
- the result;
- the fixes it forced.

The run registry is `experiments/runs.yaml`.

Template:

```
## <date> <run-id> — <one-line purpose>
- Gate:            Config:            Node/GPU:
- Result:
- Findings / fixes:
- Reproduce:
```

---

## 2026-09-24 grpo-minicoder-v1 — GRPO on top of a public SFT checkpoint (mini-coder-1.7b)

**Why.** S9's reward was all-zero: GRPO had nothing on-policy to reinforce. The evaluation (distil, then GRPO) found an existing off-policy distillation of our exact student, so we skip building SFT and test GRPO on top of it.
- `ricdomolm/mini-coder-1.7b` is Qwen3-1.7B, SFT'd on about 400k Qwen3-Coder-30B-A3B trajectories (mini-swe-agent, SWE-smith tasks, MIT). The card reports 18.6 % pass@1 on Verified bash-only.

**Harness changes (commit `b926fbd`):**
- Protocol `mswea`, which is mini-swe-agent v1:
  - the system prompt, task template, observation text and format-error text are byte-identical to the SFT corpus (41 rows checked, one template);
  - a reply must contain exactly one ```bash block.
- mini-coder is rendered with `enable_thinking=True`. Its SFT turns carry no `<think>` block, and the plain `<|im_start|>assistant\n` prefix matches how the authors evaluated it.
- Trainer changes:
  - truncated importance sampling (cap 2) against vLLM's log-probs;
  - a token-mean loss;
  - over-long trajectories are cut, not dropped;
  - mismatch statistics come from the update's own forward pass.

### Probe: mini-coder-1.7b on the 8-task pool (`experiments/probe-minicoder-pool-{a,b}.json`)

Setup: 16 episodes per task, 50 turns / 32k tokens, T = 1.0 plain sampling. It ran on two vLLM servers, one per GPU, with 8 episodes in flight on each.
- 16 episodes on one server thrashed the 95k-token KV cache: the prefix-cache hit rate fell from 42 % to 17 %, and decoding ran at 130–180 tokens/s.
- At 8 per server, the hit rate was 92–94 % and decoding about 280 tokens/s.

| Task | Resolved | Non-empty patches |
|---|---|---|
| pydicom-811 | **16/16** | 16 |
| pydicom-1694 | **8/16** | 16 |
| astroid-1614 | **6/16** | 16 |
| pydicom-1256 | **1/16** | 13 |
| astroid-946, marshmallow-1359, sqlfluff-1625, astroid-1268 | 0/16 each | 63 of 64 |

- **Total: 31/128 (24 %).** Inside the 5–30 % band from the evaluation. Base Qwen3-1.7B scored 0/140 in S9.
- **Actions:** 99–100 % valid, 0 % turn-1 submits.
- **Exits:** 104 submitted, 24 `max_turns`. 31–36 % of episodes repeat a command 3 or more times.
- **Budget:** resolved episodes took a median of 22 turns (p90 43, max 50) and at most 23.2k tokens. A 30-turn cap would have lost 10 of the 31 successes, so GRPO uses 50 turns / 24k tokens.
- **Caveat:** mini-coder's SFT data contains SWE-smith bugs from all four pool repos. These pool numbers measure familiarity with those repos as much as skill.

**Held-out check failed: the images are broken upstream (`experiments/holdout-gate.json`).** The 79 pvlib and pyvista dev tasks are the only dev repos absent from the SFT data. For all 7 selected tasks, the gold patch scores 0:
- the pvlib images ship NumPy 2, and pvlib 0.8 uses `np.Inf`, so it fails at import;
- pyvista fails with an `AttributeError` in `pyvista/core/dataset.py`.

Any generalisation claim therefore needs the Verified subset.

**Trainer memory** (`scripts/trainer_memcheck.py`; r = 16, all linear layers, 17.4M parameters, one update on GPU1):

| Tokens | Peak allocated | Time |
|---|---|---|
| 16k | 8.1 GiB | 10.6 s |
| 24k | 10.7 GiB | 17.8 s |
| 32k | 13.1 GiB | 28.3 s |

### Run: `configs/grpo_minicoder_check.yaml`

- **Schedule:** 30 steps × 2 tasks × G = 8, no oracle rollouts.
- **Pool:** `configs/pool_minicoder_mixed.txt`, the 3 mixed tasks.
- **Sampling and limits:** T = 1.0; 50 turns / 24k tokens.
- **Update:** LoRA r 16 / α 32 on all linear layers, lr 2e-5, token-mean loss, TIS cap 2, clip 1.0.
- **Layout:** vLLM on GPU0 (`swe-serve`, 0.88 memory, rollout concurrency 8), trainer on GPU1 (`swe-train`).
- **Started** 08:20:33 PDT.

**Result (08:20–14:29 PDT, 30 steps, about 12.3 min per step; records in `runs/grpo-minicoder-v1/`):**

| Steps | Resolved | Rate |
|---|---|---|
| 1–10 | 49/160 | 31 % |
| 11–20 | 48/160 | 30 % |
| 21–30 | 56/160 | 35 % |
| **All** | **153/480** | **32 %** (pre-GRPO probe on these 3 tasks: 15/48 = 31 %) |

Per task, over the run:
- pydicom-1694: 68/159 (43 %); probe 8/16.
- astroid-1614: 59/144 (41 %); probe 6/16.
- pydicom-1256: 26/176 (15 %); probe 1/16.

**The mechanism works:**
- Rewards were non-zero on every step, and 53 of 60 groups were mixed; the 7 degenerate groups were all 0/8 on pydicom-1256.
- The objective rose on 24 of 30 steps.
- TIS weights averaged 1.000, with none capped. Train/serve mismatch mean was 0.009–0.012.
- The served adapter moved 0.031–0.082 nats per token from the base. 5 over-long rollouts were truncated, not dropped.

**The outcome did not move.** The success rate is flat within noise (±5 points per block). GRPO v1 did not measurably improve the policy. Likely reasons:
- small updates: lr 2e-5, grad norm 0.03–0.05 against a clip of 1.0;
- only 16 rollouts per step for 30 steps;
- a 3-task pool.

The user chose the next plan: screen more dev tasks for a larger pool, then run v2 with more tasks per step and a higher learning rate.

**A caveat found later, affecting v1 and the probe.** Every turn of an episode sent the same vLLM seed. vLLM re-seeds per request, so token j of every turn used the same random draw and the turns of an episode were correlated. This is fixed from commit `21f4a23` on: the seed now changes per turn.

### Screening for the v2 pool (`scripts/rolling_probe.py`, `experiments/screen-dev-v2.json`)

Setup: 8 episodes per task, same limits (wall clock 1,200 s), both GPUs, images pulled in rolling batches of 10 (no layer cache) and deleted unless the task was mixed. Before it ran, the script got an adversarial review (4 lenses, 25 confirmed findings, all fixed; commit `21f4a23`).

**List 1** (`configs/screen_dev_v2.txt`: the 40 smallest single-file fixes; 14:32–16:05):

| Outcome | Tasks |
|---|---|
| Usable (image present, gate passed) | 22 |
| Mixed | 5: astroid-1719 7/8, astroid-1903 4/8, marshmallow-1343 3/8, pydicom-1192 2/8, astroid-1262 1/8 |
| Base success on the usable ones | 17/176 (10 %) |
| Gate negative | 9, all pydicom: the gold patch passes its fail-to-pass tests but some pass-to-pass tests fail offline (likely tests that download pydicom test data) |
| No image on Docker Hub | 9 ("requested access to the resource is denied"); 34 of the 39 remaining sqlfluff dev images are missing too |

**Incident.** During list 1, a grading run (an astroid test suite on a policy patch) grew to 61.7 GB RSS on node08's 62 GB and stalled the node. I killed that process. The fixes:
- a watchdog in tmux `swe-watchdog` that kills any `testbed` process above 16 GB;
- `ulimit -v` 12 GB on every sandbox command and grading run (commit `8dc7586`).

**List 2** (`configs/screen_dev_v2b.txt`, 64 more candidates that do have images) runs with `--stop-after-usable 18`. That gives 40 usable tasks in total, as the user asked.

**List 2 result** (16:07–17:15; `experiments/screen-dev-v2b.json`):
- It stopped as designed after 2 batches, at 20 usable tasks.
- Mixed: 3 — marshmallow-1164 3/8, sqlfluff-884 1/8, astroid-984 1/8.
- Base success: 5/160 (3 %). These tasks are larger fixes.
- 0 gate failures, because astroid, marshmallow and sqlfluff come first in the list.
- The 10 prefetched but unscreened images were deleted.
- The watchdog log stayed empty: the 12 GB `ulimit -v` held.

**Screening total:** 42 usable dev tasks, 8 mixed. With v1's 3 mixed tasks, the v2 pool is 11 tasks (`configs/pool_minicoder_v2.txt`). Their images are in node08 `/var/tmp/swe/sif-dev`: 8 files plus 3 symlinks into `/tmp/swe-sif`.

### GRPO v2: `runs/grpo-minicoder-v2`, `configs/grpo_minicoder_v2.yaml`

**Config:**
- 30 steps × 4 tasks × G = 8 (32 rollouts per step), lr 5e-5 (v1: 2e-5), LoRA r 16 on all linear layers, token-mean loss, TIS cap 2.
- Pool: the 11 mixed tasks.
- Rollouts on both GPUs: GPU0 at 0.88; GPU1 at 0.84 with `--max-num-seqs 16` and sleep mode. The trainer sits on the CPU until the GPU1 server sleeps.
- Before the run, the dual-server code got an adversarial review (3 lenses, 9 confirmed findings, all fixed) and a 2-step smoke test (`runs/grpo-v2-smoke`).

**Timeline:**
- 17:36: started.
- Steps 11 and 12 slowed to 39 and 45 min (2,326 s and 2,714 s) and the trainer was OOM-killed during step 13: one sandbox wrote about 32 GB into its RAM-backed tmpfs, swap filled, and the OOM killer took the trainer (details in `src/sarl/env/apptainer.py` and commit `400134d`). Several post-resume steps were also slow (step 14 1,306 s, step 24 2,095 s).
- 21:24: resumed from step 12 with `--resume` (adapter hash verified, task draws replayed, AdamW moments reset).
- 01:37: finished.

**Speed:**
- Steps took a median of 756 s (interquartile 656–935 s, min 561 s) for 32 rollouts, against a v1 median of 743 s for 16 rollouts. That is about 2× the rollout throughput.
- The worst step, 2,714 s, was during the memory incident.
- Sleep/wake took 1–3 s per step.

**Health:**
- 0 rollout errors; 14 over-long trajectories truncated.
- TIS weights averaged 1.000 and none were capped; mismatch mean 0.009–0.012.
- The objective rose on **30/30** steps.
- The served policy drifted from the base by 0.044 nats per token at step 1 and 0.095 at step 30.
- 34 of 120 groups were degenerate: astroid-1262, astroid-984 and pydicom-1256 are mostly 0/8, and astroid-1719 is mostly 8/8.

**Outcome on the training tasks:**

| Task | Pre-GRPO base | Steps 1–15 | Steps 16–30 |
|---|---|---|---|
| marshmallow-1164 | 3/8 | 15/32 (47 %) | 10/32 (31 %) |
| marshmallow-1343 | 3/8 | 26/56 (46 %) | 25/48 (52 %) |
| pydicom-1192 | 2/8 | 16/64 (25 %) | 9/32 (28 %) |
| pydicom-1256 | 1/16 | 5/32 (16 %) | 6/40 (15 %) |
| pydicom-1694 | 8/16 | 24/56 (43 %) | 18/56 (32 %) |
| astroid-1262 | 1/8 | 1/56 | 0/40 |
| astroid-1614 | 6/16 | 10/32 (31 %) | 9/24 (38 %) |
| astroid-1719 | 7/8 | 30/32 (94 %) | 54/56 (96 %) |
| astroid-1903 | 4/8 | 34/40 (85 %) | 44/56 (79 %) |
| astroid-984 | 1/8 | 2/56 | 1/40 |
| sqlfluff-884 | 1/8 | 4/24 (17 %) | 14/56 (25 %) |
| pooled | 37/112 (33 %) | 167/480 (35 %) | 190/480 (40 %) |

The pooled rise from 35 % to 40 % comes from the task mix. Comparing each task only with itself:
- **Late vs early: −1.2 ± 2.6 points.**
- **During the run vs the pre-GRPO baseline: +2.2 ± 4.1 points** (1 SE).

So GRPO v2 did not measurably raise the success rate even on the tasks it trained on, despite a clean mechanism and twice v1's drift. The held-out test is the Verified evaluation below.

### Evaluation: base vs GRPO v2 on the gated Verified subset

- Tasks: `configs/eval_verified_v1_gated.txt`, 50 tasks.
- Episodes: 4 per arm, with the same per-turn seeds for both arms.
- Arms: `base=policy` and `grpo=policy-051b4b-s30`, both served on both GPUs.
- Limits: as in training.
- Output: `experiments/eval-verified-v1.json`. Started 01:39.

**Result:** 01:39–05:40, plus a resume for matplotlib-25960, whose image pull failed at SIF creation.

| Arm | Resolved | Rate |
|---|---|---|
| base mini-coder-1.7b | 38/200 | 19.0 % |
| GRPO v2 (step 30) | 41/200 | 20.5 % |

- Per-task mean difference: **+1.5 pp**, 95 % bootstrap CI [−3.5, +6.5], sign-flip p = 0.71 (15 tasks differ).
- Paired by seed: GRPO alone solved 16 episodes, base alone 13, both 25.
- 20 of 50 tasks were solved by either arm.

**No detectable effect on held-out Verified.** This matches the flat curve on the training tasks. Base mini-coder at 19 % agrees with its card (18.6 %, bash-only), which confirms the harness reproduces the model.

---

## 2026-09-25 — node04 as a third Gemma server: ×1.78 rollout throughput (A/B on the real pipeline)

Question: can node04 (RTX 4060 Laptop, 8 GB, sm_89) serve Gemma 4 E4B next to node08's two A4000s, and is it a real
speed-up for the pipeline? Records: `experiments/ab-serving/` (`R1..R4.json/.log`, `metrics.log`, `mon.log`,
`summary.json`); report with `python scripts/ab_serving_report.py experiments/ab-serving --baseline R2`.

**Fitting E4B on 8 GB** (`scripts/vllm_ple_offload.py`; `SARL_PLE_OFFLOAD=1` in `20_serve_policy.sh` / `22_serve_gemma4.sh`):
- The FP8 checkpoint loads 10.76 GiB on node08. 5.25 GiB of that is the per-layer embedding table (262,144 × 42 × 256,
  bf16), of which each token reads one 21.5 KB row.
- vLLM 0.25.1's `--cpu-offload-params` reaches only the decoder layers (`make_layers`); the PLE is built outside them.
  The launcher wraps `VocabParallelEmbedding.__init__`: the table is built on CPU, pinned in place with
  `cudaHostRegister`, and handed to the model as a UVA CUDA view (vLLM's own `UVAOffloader` mechanism). Lookups gather
  rows over PCIe; everything else stays on the GPU. Model load: **5.46 GiB**.
  - torch's `pin_memory=True` rounds 5.25 GiB up to 8 GiB of locked RAM and left the 15 GB host with 1.3 GB
    available; exact-size registration leaves 5.5 GB pinned.
- Compiled mode still fails: Inductor copies the table back to the GPU (5.25 GiB OOM). CUDA graphs without Inductor
  work: `-cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY` captured in 2 s and 0.13 GiB (`SARL_EAGER=0`).
- Final server: `SARL_PLE_OFFLOAD=1 SARL_EAGER=0 SARL_SERVE_MORE="-cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY"
  SARL_MAX_NUM_SEQS=8 SARL_MAX_BATCHED=2048 bash scripts/22_serve_gemma4.sh 0 8003 0.935` → 72.6k KV tokens; tmux
  `swe-serve4`; weights from a node-local copy of the snapshot (`/var/tmp/swe/hf-cache`, 13 GB, 116 s from NFS);
  `oom_score_adj` 1000 so the OOM killer takes the server before a sandbox.
- Same model: greedy cross-check on 4 real pool prompts against node08 (`xcheck`, 96 tokens): first divergence at
  token 5–74, max |Δlogp| 0.33, mean 0.008–0.027 on the shared prefix. Different attention kernels (Triton on sm_89), so
  equivalent at sampling level, not bit-identical.

**Synthetic capacity** (`scripts/bench_servers.py`: 9-turn episodes, 2.5k prompt + 400 generated + 400 observation
tokens per turn, 1.5 s env, 120 s window after 30 s warm-up):

| server | concurrency | gen tok/s | per-seq tok/s | p50 turn (s) |
|---|---|---|---|---|
| node08 GPU0, eager (as served) | 8 | 213 | 29.0 | 13.8 |
| node08 GPU0, eager | 12 | 320 | 28.3 | 14.0 |
| node04, eager, 0.93 | 4 / 6 / 8 | 93 / 120 / 160 | 23.8 / 23.4 / 23.0 | 16.8 / 17.0 / 17.4 |
| node04, decode graphs, 0.90 | 4 / 8 | 107 / 170 | 31.6 / 25.6 | 12.8 / 14.2 |
| **node04, decode graphs, 0.935** | 6 / **8** | 140 / **210** | 29.4 / 27.5 | 13.6 / 14.2 |

**A/B on the real pipeline** (`scripts/ab_serving.sh`): the 11-task pool probe (88 episodes; the seeds, images, limits
and T = 1 of probe 2), with sandboxes and grading on node04 for every layout; one run per layout, in the order R1 R2 R3 R4.

| layout | servers × slots | episode phase (s) | turns/min | gen tok/s | resolved | valid | node04 min avail | vs R2 (turns/min) |
|---|---|---|---|---|---|---|---|---|
| R1 | node08 8+8, node04 8 | 581 | 76.6 | 528 | 7/88 | 0.868 | 3.55 GB | **×1.34** |
| **R2** (every probe so far) | node08 8+8 | 743 | 57.0 | 396 | 7/88 | 0.853 | 3.91 GB | ×1.00 |
| R3 | node08 12+12 | 599 | 70.8 | 499 | 5/88 | 0.851 | 3.79 GB | ×1.24 |
| **R4** | node08 12+12, node04 8 | **453** | **101.3** | **692** | 6/88 | 0.852 | 3.90 GB | **×1.78** (wall ×1.64) |

Findings:
- node04 adds about one node08 GPU's worth of rollouts: in R1 it served 28 of 88 episodes at 176 tok/s (node08's two:
  173 and 179) and 15.2 s per turn (node08: 14.5–16.1).
- Two independent gains. Adding node04 to the current layout is ×1.34; node08 also had unused KV headroom, and 12 slots
  per A4000 alone is ×1.24 (the 86k-token GPU1 shows it: 18 s/turn in R3). Both together: ×1.78 turns/min, ×1.75
  generated tokens/s, ×1.64 wall-clock (R4 had 8 % more turns to do). node04 on top of R3: ×1.43.
- Quality is unchanged: node04's episodes have valid-action rates of 0.877 (R1) and 0.895 (R4) against 0.834–0.878 on
  node08, and the same submit rate, patch rate and token counts. Resolved 7 / 7 / 5 / 6 of 88 across the four runs is
  sampling noise at T = 1 (equal seeds, but batching differs, so trajectories diverge).
- The host held: up to 32 sandboxes plus grading plus the server on node04 with MemAvailable never below 3.55 GB, load
  1.7–3.3 on 22 threads, no watchdog kills.
- Caveats: one run per layout; the wall-clock includes the straggler tail (longest episode 260–325 s); node08's prefix
  caches were warm for every run. Turns/min and generated tokens/s are the robust comparators.

**Default layout for probes from now on:** `--servers http://node08:8001 http://node08:8002 http://node04:8003
--per-server 12 12 8` (R4). During training node08 GPU1 sleeps while the trainer steps, so GPU0 and node04 carry the
rollouts then.

Code: `rolling_probe.py` takes `--per-server` per server, identifies a server's weights by HF snapshot (repo and
revision) rather than the cache path, and records `timing` per batch; `tests/test_swe_serving.py` covers these and the
launcher's PLE detection.

Next lever, untested: node08 is still eager. Decode-only CUDA graphs without Inductor (what node04 runs) gave node04
+25–30 % per sequence; the same flags on node08 need a server restart to measure.

---

## 2026-09-25 — Gemma 4 E4B: exploration and serving probe

**User decisions.**
- After v2: "the model size is way too small and we have run far too less steps"; explore a Gemma ~4B policy.
- A read-only exploration workflow (13 agents: research, synthesis, skeptic verification) recommended `google/gemma-4-E4B-it`, served from the FP8 copy, with a probe first.
- The user chose to probe E4B only, to decide after the probe, and no long mini-coder run.
- Later the user asked to put images and sandboxes on node04, then to focus on Gemma only; a started mini-coder-4b probe on node04 was stopped.

**Harness changes** (commits `a9f3ca6` and later):
- `ChatFormat` derives the turn terminator (`<turn|>` 106).
- `stop_ids()` gives `[106, 1, 50]`.
- `sanitize` covers every special token.
- The policies require stop ids.
- The answer is the text after the last `<channel|>` (below).

**Serving** (`scripts/22_serve_gemma4.sh`; FP8 checkpoint, W8A16 on sm_86):
- `--language-model-only` and `--max-num-batched-tokens 4096`.
- **`--enforce-eager`**: vLLM 0.25.1's torch.compile (Inductor) tries to allocate a 5.25 GiB copy of the per-layer embedding table and runs out of memory on the A4000, with or without LoRA.
- LoRA serving starts.
- KV cache: 109k tokens on GPU0 (0.87) and 86k on GPU1 (0.84).
- Decoding is about 26 tok/s per sequence at 6 concurrent in eager mode.

**Layout:**
- node04 holds the images (`/var/tmp/swe/sif-dev`; 11 GB copied from node08 at 114 MB/s) and runs the probe harness, sandboxes (`ulimit -v` 4 GB, 5 GB watchdog) and grading (node-local grader venv).
- node08 only serves.

**Probe 1** (`experiments/probe-gemma-pool.json`; 11-task pool × 8, 50 turns / 24k, T = 1.0):

| Metric | Result | Pre-registered "Go" bar |
|---|---|---|
| Resolved | 6/88 (6.8 %) | 5–30 % ✓ (just) |
| Mixed tasks | 4: sqlfluff-884 3/8, astroid-1614 1/8, marshmallow-1164 1/8, pydicom-1192 1/8 | ≥ 4 ✓ (just) |
| Valid actions | **74.1 %** turn-weighted (533/719; 69.8 % as the mean of per-episode rates) | ≥ 90 % ✗ |

- 73 of 88 episodes submitted, after 8.2 turns on average; 15 ended on 3 format errors.
- For comparison, mini-coder-1.7b scored 31 % on its own probe pool.

**Why the actions were invalid.** Replaying 7 episodes (`gemma_format_diag`) showed the pattern. Gemma 4 leaks its thinking channel even with thinking off, writing `reasoning … ```bash X``` <channel|>```bash X````. The reasoning repeats the command, so the reply holds two bash blocks and mini-swe-agent's exactly-one rule rejects it. 14 of 20 invalid replies contained `<channel|>`; **13 of those 14 parse as valid after the last `<channel|>`**.

**Fix:** `ChatFormat.answer_text` parses the answer only.
- Gemma's own template strips thoughts from history the same way.
- Qwen tokenizers have no special `<channel|>`, so Qwen parsing is unchanged.
- The whole reply stays in the token transcript and is trained on.
- Turns record `leaked_thought`.

Probe 2 (the same 88 episodes with the fix, 8 per server) started at 07:04: `experiments/probe-gemma-pool-v2.json`.

**Probe 2** (`experiments/probe-gemma-pool-v2.json`; same pool, settings and seeds, parsing after `<channel|>`; 15 min
for the batch, gate included):

| Metric | Probe 1 | Probe 2 | Pre-registered "Go" bar |
|---|---|---|---|
| Resolved | 6/88 (6.8 %) | **9/88 (10.2 %)** | 5–30 % ✓ |
| Mixed tasks | 4 | **6**: sqlfluff-884 3/8, astroid-1614 2/8, pydicom-1694, astroid-1719, astroid-1903, marshmallow-1343 1/8 each | ≥ 4 ✓ |
| Valid actions | 74.1 % | **86.3 %** (102 invalid of 746 turns) | ≥ 90 % ✗ |
| Exits | 73 submitted, 15 format errors | 81 submitted, 5 format errors, 2 context | |
| Non-empty patches | 42 | 61 (23 submissions had no patch) | |

- By the pre-registered rule this is **Stop** (valid actions below 90 %), even though the solve rate and mixed-task
  count are in the band. The decision is the user's.
- The invalid turns are spread thin: 69 of 88 episodes have at least one, about one per episode, on every task
  (3/38 to 15/60 turns). The format-error message is fed back and the model usually recovers on the next turn; only 5
  episodes ended on three in a row.

**Why the remaining 14 % were invalid** (replay of 22 episodes / 192 turns with the current parser, `gemma_invalid_diag`,
no grading): 36 invalid turns. 28 of them had `finish_reason = length`: the reply hit `Limits.max_new_tokens = 1024`
before the bash block closed (18 with no block yet, 10 cut during the thinking that repeats the block, before the
`<channel|>`). Only 8 of 192 turns were genuinely malformed (two or three bash blocks in the answer itself). 31 of 192
turns hit the cap overall, and vLLM's counters show 13 % of all Gemma requests so far ended on `length`. Gemma thinks at
length before its answer, so the cap set for mini-coder truncates it. `rolling_probe.py` gained `--max-new-tokens`.

**Probe 3** (`experiments/probe-gemma-pool-v3.json`; same pool, seeds and settings, `--max-new-tokens 4096`; servers
node08 12+12 + node04 8; 494 s for the batch):

| Metric | Probe 1 | Probe 2 | Probe 3 | Pre-registered "Go" bar |
|---|---|---|---|---|
| Resolved | 6/88 | 9/88 | **4/88 (4.5 %)** | 5–30 % ✗ (by one episode) |
| Mixed tasks | 4 | 6 | **4**: astroid-1614, astroid-1903, marshmallow-1343, sqlfluff-884 (1/8 each) | ≥ 4 ✓ |
| Valid actions | 74.1 % | 86.3 % | **94.9 %** (33 invalid of 650 turns) | ≥ 90 % ✓ |
| Exits | 73 / 15 / 0 | 81 / 5 / 2 | 87 submitted / 0 format errors / 1 context | |
| Non-empty patches | 42 | 61 | 46 | |
| Mean turns | 8.2 | 8.5 | 7.4 | |

- Pooled over the three probes: **19/264 resolved (7.2 %)**; the per-probe counts 6, 9, 4 are within binomial noise
  of that rate (sd ≈ 2.4 solves per 88 episodes). Mixed tasks over all probes: 6 of 11. The raised cap removed the
  format errors entirely and shortened episodes (fewer patches, more direct submits); whether it lowered the solve
  rate cannot be told from one probe.
- Verdict against the rule: the last probe meets the valid-action and mixed-task bars and misses the solve band by
  one episode; the pooled probes meet all three. The decision is the user's. If Go: the base rate of ~7 % means most
  groups will be degenerate, so the pool needs more mixed tasks (screen more dev tasks with Gemma, as for mini-coder)
  alongside the trainer port.

**Decision (user, 2026-09-25 14:20): Go Gemma.** Two tracks started:
- **Pool screen** (`configs/screen_gemma_v1.txt` → `experiments/screen-gemma-v1.json`, log `/var/tmp/swe/screen-gemma-v1.log`
  on node04): 86 dev tasks = the 42 mini-coder-usable tasks not in pool v2 + the 44 `screen_dev_v2b` candidates never
  screened; 18 known bad ones excluded (9 gate failures, 9 missing images). 8 episodes per task, `--max-new-tokens 4096`,
  `--keep-mixed`, batches of 10 with rolling pulls; servers node08:8001 (12) + node04:8003 (8). node08 GPU1's server
  (`swe-serve1`) was stopped to free the GPU for trainer development.
- **Trainer port** (below).
- **Screen result** (14:22–16:52, 9 batches of ~12 min, `experiments/screen-gemma-v1.json`): 69 of 86 tasks gated
  (the 17 failures are all pydicom, the offline test-data issue); 552 episodes, **19 resolved (3.4 %)**, valid actions
  93.9 %. Mixed: 7 (pydicom-1192 2/8, marshmallow-1343 2/8, sqlfluff-884 2/8, pydicom-965 2/8, marshmallow-1164 1/8,
  sqlfluff-905 1/8, pydicom-1562 1/8); always solved: sqlfluff-880 (8/8, excluded). Astroid: 0/8 on every task.
  (The list builder re-screened the 11 pool-v2 tasks by mistake; their results agree with the probes.)
- **Pool v1** (`configs/pool_gemma_v1.txt`, built by `scripts/gemma_pool.py` from the three probes and the screen: any
  task solved sometimes and never always): **11 tasks**, pooled 30/264 episodes solved (11 %): marshmallow-1164,
  marshmallow-1343, pydicom-1192, pydicom-1562, pydicom-1694, pydicom-965, astroid-1614, astroid-1719, astroid-1903,
  sqlfluff-884, sqlfluff-905. Images copied to node08 (`/var/tmp/swe/sif-dev`, 14 SIFs). The run: `grpo-gemma-v1`
  (`configs/grpo_gemma_v1.yaml`, 30 steps × 4 tasks × 8, launched 16:50 in tmux `swe-train` on node08).

### Trainer port to Gemma 4 E4B (`src/sarl/gemma.py`, `configs/grpo_gemma_v1.yaml`, `tests/test_swe_gemma.py`)

Design (from a 5-agent read of transformers 5.14.1, vLLM 0.25.1, PEFT 0.19.1 and our trainer, then verified against the
sources):
- **Class.** `AutoModelForCausalLM` maps the checkpoint to the multimodal `Gemma4ForConditionalGeneration`; with
  `vision_config`/`audio_config` set to None it builds the language model only, under the checkpoint's own names
  (`model.language_model.layers.N…`). PEFT then writes adapters as `base_model.model.model.language_model.layers.N…`,
  which vLLM's `--language-model-only` mapper (`model.language_model.` → `language_model.model.`) accepts.
- **Base weights = the served FP8 checkpoint.** `FP8Linear` keeps the float8_e4m3fn weight and the per-channel bf16
  scale and dequantises per forward (`F.linear(x, w8.to(bf16) * scale)`), what vLLM's W8A16 kernel computes on
  sm_86, so trainer and sampler share weights. The FP8 checkpoint (`prithivMLmods/gemma-4-E4B-it-FP8`) holds 379 FP8
  tensors (3.75 GiB; all 42 layers' q/k/v/o, gate/up/down, per_layer_input_gate, per_layer_projection, and
  per_layer_model_projection) plus bf16 norms, the 1.25 GiB embedding and the 5.25 GiB PLE table; 36 k/v tensors of
  the 18 KV-shared layers (24–41) have no module in HF and are skipped. Loading = `from_pretrained` on the bf16
  Google snapshot (structure and buffers) then the FP8 overlay: 343 `FP8Linear`, 280 bf16 tensors copied, 6 s from
  node08's local copy (`/var/tmp/swe/hf-cache`, 16 GB). NF4 was rejected on evidence: terminal-bench measured an
  NF4-vs-FP8 gap of 0.28 nats mean with 16 % of tokens > 0.5.
- **PLE on the CPU.** `CpuEmbedding` wraps `embed_tokens_per_layer`: the batch's rows (21.5 KB each) are gathered in
  host memory and moved; `_apply` is a no-op so `.to(device)` leaves the table alone.
- **Soft-cap.** `chunk_lp` applies `30·tanh(x/30)` in bf16 after `lm_head`, before the fp32 log-softmax, exactly as
  the HF heads (`modeling_gemma4.py:1895-1899`) and vLLM's logits processor do.
- **LoRA scope.** `targets` may be a regex (PEFT fullmatch): the language model's seven projections only; layers
  24–41 have no k/v, so 258 targets, **34,881,536** parameters at r 16.
- **`_core()`** returns the text decoder (`m.model.language_model`), skipping the wrapper's placeholder-mask work.
- PEFT gotcha: it reads `module.weight` for the adapter's device/dtype; `FP8Linear` exposes a one-element bf16 probe
  as `weight` (never the fp8 tensor), so adapters are created in bf16 and upcast to fp32 by PEFT's default.
- Tests (`tests/test_swe_gemma.py`, tiny random Gemma 4 on the CPU): chunked log-probs equal HF's forward with the cap
  and differ without it; adapter keys stay in the language model with no k/v in shared layers; the pinned PLE gives
  identical log-probs; an FP8 base layer trains under PEFT with fp32 adapters. 19/19 trainer tests pass on node08.

**G1/G3 memcheck** (`scripts/trainer_memcheck.py configs/grpo_gemma_v1.yaml`, node08 GPU1, math SDPA disabled, one
trajectory per update, 30 % sampled positions; logs `/var/tmp/swe/memcheck-gemma{,2}.log`):

| Tokens | Without `save_on_cpu` | With `save_on_cpu` (activations for backward in pinned host RAM) |
|---|---|---|
| after load | 5.08 GiB resident (343 FP8Linear, PLE on cpu, lm_head tied) | same |
| 8,192 | 10.06 GiB peak, 19.6 s | 8.89 GiB, 20.5 s |
| 12,288 | — | 10.94 GiB, 37.6 s |
| 16,384 | **OOM** | **12.75 GiB (13.38 reserved), 65.8 s** |
| 20,480 | OOM | 14.67 GiB (15.37 reserved), 89.2 s: too tight |
| 24,576 | OOM | OOM |

- Neither flash nor memory-efficient SDPA refused head_dim 256/512 (no "no available kernel" error with the math
  kernel disabled).
- The plan's activation estimate was low: checkpointing keeps one bf16 input per layer (0.2 MiB per token over 42
  layers) and the per-layer recompute holds the [T, 10240] MLP intermediates. `save_on_cpu: true` (a
  `torch.autograd.graph.save_on_cpu(pin_memory=True)` context around the body) removes the first term at a PCIe copy
  per step. **`max_train_tokens: 16384`** with ~2.2 GiB headroom. Probe-3 episodes ran 5.9k–9.9k tokens at the median
  8.5k, so most trajectories fit; longer ones are truncated (`truncate_overlong`).
- Cost: 66 s per 16k-token trajectory (mini-coder 1.7B: 17.8 s at 24k). With ~7 % base success most groups are
  degenerate and skipped, so a step trains 8–16 trajectories: 5–15 min of update per step on top of ~8 min of rollouts.

**G2 trainer-vs-server log-prob gap** (`scripts/logprob_gap.py sample` on node04 against node08:8001, then `score`
on node08 GPU1; `runs/gap/base`, log `/var/tmp/swe/gap-score-base.log`): 8 real mswea episodes on the 4 gate tasks
(`configs/pool_gemma_gate.txt`), T = 1, 7.2k–19.2k tokens, 2.0k–4.5k sampled each; the trainer (fresh r-16 adapter,
B = 0) re-scores exactly the sampled tokens against vLLM's `token_logprobs`:

| | mean \|Δ\| | p50 | p99 | max | frac > 0.5 | signed mean |
|---|---|---|---|---|---|---|
| per episode | 0.012–0.019 | 0.0000–0.0014 | 0.12–0.15 | 0.21–0.36 | 0 | −0.0013…+0.0004 |
| **pooled, 23,999 tokens** | **0.0151** | 0.0003 | 0.139 | 0.36 | 0.0000 | −0.0005 |

Pass (bar: mean ≤ 0.05, p99 ≤ 0.5, frac > 0.5 ≤ 1 %, |signed| ≤ 0.02). Same regime as mini-coder's bf16/bf16 runs
(0.009–0.012 mean): the FP8-faithful base plus the bf16 soft-cap reproduces the sampler. Scoring cost: 4–14 s per
episode at 6.8–8.9 GiB peak (no grad).

**G4 adapter round trip** (`logprob_gap.py step` → `push` → `sample` under the adapter on node04 → `score --adapter`;
logs `/var/tmp/swe/gap-{step,push,sample-adapter,score-adapter}.log`):
- One update on the 8 gate trajectories (A = 1, 23,999 tokens, lr 5e-5): loss 0.614, grad norm 0.109, TIS weights ≈ 1,
  stepped; adapter `f7f603a7e37d` saved to `/var/tmp/swe/adapters/gate/step-1` (34.9 M params).
- `POST /v1/load_lora_adapter` on node08:8001 → **200** (`policy-gate-s1`): vLLM accepted the packed qkv/gate-up
  mapping with q/o-only adapters in the KV-shared layers.
- `served_dlogp_vs_base` on the 25-token probe: **0.069** mean, 0.31 max (bar > 0.01): the adapter is applied, not
  merely registered.
- 4 episodes sampled under the adapter (3.8k–13.4k tokens, 1.5k–4.0k sampled), re-scored by the trainer with the same
  adapter loaded: pooled 11,632 tokens, mean |Δ| **0.0169**, p50 0.0008, p99 0.148, max 0.45, 0 % > 0.5, signed
  −0.0006 (bar ≤ 1.5 × 0.0151 = 0.023). Adapter unloaded afterwards.

Gates G0–G4 pass.

**G5 one end-to-end step** (`bash scripts/21_train.sh configs/grpo_gemma_gate.yaml --run-name gemma-gate-s1`, node08;
`runs/gemma-gate-s1/`, log `/var/tmp/swe/train-gemma-gate.log`; 16:13–16:21): 2 tasks × G = 4 on node08:8001 (12) +
node04:8003 (8), trainer on GPU1.
- Two fixes were needed for a two-node run first (`train.py`): `preflight_servers` compares weights by HF snapshot
  (repo + revision) rather than cache path, and `LoraSync.load` copies each new adapter to a remote server's host under
  the same path (`scp`, ~140 MB) before `load_lora_adapter`, deleting the copy once superseded.
- Rollouts 404 s, step 492 s. sqlfluff-884 [1, 1, 0, 1] (mixed), astroid-1903 0/4 (degenerate, skipped); 4 trajectories,
  9,019 tokens trained; valid actions 94.5 %; no truncation.
- Update: loss 0.098, grad norm 0.142, `objective_gain` +0.0023; TIS mean weight 0.9996, none capped, min 0.30.
- Mismatch (in the update's own forward): mean 0.032, p50 0.001, p99 0.34, max 1.21, 0.29 % > 0.5 (bar ≤ 1 %). Higher
  than G2's 0.015 because half the rollouts came from node04, whose attention kernels differ from node08's (the greedy
  cross-check showed max |Δlogp| 0.33 between the two servers); the importance weights absorb it.
- Sync: adapter `policy-ffefc1-s1` loaded on both servers (the node04 copy worked), `served_dlogp_vs_base` 0.078,
  greedy probe unchanged (expected for one small step).

**All gates pass.** The trainer port is complete; the first Gemma GRPO run waits only for the pool
(`configs/pool_gemma_v1.txt` from the screen).

### SWE-smith as the next task source: scoping (2026-09-25, read-only; 4 agents over the HF table, Docker Hub, the swesmith code and our loader)

The user's direction: spend effort on more tasks (SWE-smith), not on the sandbox network switch. Findings (all checked
against the pinned table `SWE-bench/SWE-smith @ ea6d7173`, 59,136 rows, 7 columns: instance_id, patch, FAIL_TO_PASS,
PASS_TO_PASS, image_name, repo, problem_statement):
- **Usable.** Dataset and code MIT; images public on Docker Hub under `jyangballin/swesmith.x86_64.<owner>_1776_<repo>.<sha8>`
  (one image per repo, 1.3–1.6 GB, identical digests under `swebench/`). 222 repos; **none of the 12 Verified repos**
  has rows (flask and sympy have profiles and images but 0 tasks; keep the repo filter anyway).
- **Our four dev repos are in it:** astroid 1,770 tasks, pydicom 1,328, sqlfluff 1,293, marshmallow 389 (4,780 behind
  4 images, 5.7 GB); webargs 321 and apispec 301 are the cheapest extensions. pvlib/pyvista are absent.
- **Semantics.** `patch` *introduces* the bug into the mirror's clean `main` (one commit); the gold fix is the reversed
  patch. Official evaluation: check out the instance branch (bug + a commit that deletes the F2P test files), restore the
  tests, apply the prediction, run the repo's pytest command, parse `PASSED/FAILED` lines; resolved iff all F2P and P2P
  pass; 90 s test timeout at validation time. 30 % of rows have an empty problem statement (LM-written issues); pool
  repos: 912 of 4,780 empty.
- **Contamination.** Instance ids and commits cannot collide with Verified. Against *our dev split*: 7 `pr_<n>` mirror
  instances (astroid 1959/1962/1978/2015/2307/2309, sqlfluff-5074) have byte-identical problem statements to dev tasks →
  exclude every `pr_` instance. mini-coder's SFT corpus saw 100 % of these repos' SWE-smith tasks (irrelevant for
  Gemma; disqualifies any mini-coder comparison on them).
- **Integration (9–12 h hands-on):** `DATASETS["smith"]`; Task gains `bug_patch`, `hidden_files`, image-keyed
  `sif_name`; `reverse_unidiff` for the gold; `ApptainerEnv` setup hook (apply the bug patch, delete the F2P test files,
  commit, then pin HEAD); `grade.py` setup pre-step in the fresh container plus a synthesized `pytest -rA` eval script
  parsed by swebench's `parse_log_pytest`; `scripts/smith_pool.py` (filter: non-empty issue, non-empty P2P, not `pr_`,
  single-file patch ≤ 2,000 chars, 1–5 F2P; hashed 50 per repo → `configs/screen_smith_v1.txt`, 200 tasks); screens
  run with `--keep-all` (four tasks share one image, and `rolling_probe` deletes per task).
- **Added requirement (not in the agents' plan): no bug in git history.** Committing the bug on top of `main` leaves
  `git show HEAD` / `git diff main` inside the sandbox showing the exact reversed fix. The setup must leave a
  single-commit history of the buggy state (orphan branch, delete `main`, expire reflogs, `git gc --prune=now`) and
  the grader must rebuild the same state. The official SWE-agent setup has this leak (origin/main and the bug commit
  are present); ours must not.
- **Gates:** G0 image state (HEAD = mirror main, bug patches apply); G1 grading gate on 200 tasks (gold = 1, empty = 0,
  ≥ 80 % per repo, CPU only, 2–3 h); G2 unit tests + one episode; G3 Gemma screen (200 × 8 episodes, 15–25 h of the two
  serving GPUs: run after the dev-pool GRPO run or at reduced slots); G4 two-step training smoke.
- **Plan:** implement and gate G0–G2 while `grpo-gemma-v1` trains on the dev pool (CPU work on node04); screen after.

**Implemented (2026-09-25 evening, commit below):** `src/sarl/smith.py` (mirror-name parsing, `reverse_unidiff` = the
gold, `setup_script` for sandbox and grader, `eval_script_for`), `DATASETS["smith"]` + `SMITH_REPOS` (`paths.py`), the
loader (`load_split("smith")`: pyarrow filter on the four repos, `pr_` mirrors dropped; 4,662 tasks), `Task.bug_patch`
/ `hidden_files` / image-keyed `sif_name` / `sandbox_setup` / `grader_setup`, `ApptainerEnv(setup=…)` (runs the setup
before the diff base is pinned), the grader pre-step (`/sarl_in/setup.sh`, `>>>>> Reset Failed` on failure), the
contamination guard mapping mirrors back to `owner/repo` and matching copied issue text, `scripts/smith_pool.py`.
The sandbox setup ends with `rm -rf .git; git init; git add -A; git commit`: one commit, no remote, no reflog, so the
bug diff is not recoverable from git. Tests: `tests/test_swe_smith.py` (5) pass with the rest of the suite.
- **Screen list** `configs/screen_smith_v1.txt`: 1,303 candidates after the filter (marshmallow 103, astroid 380,
  sqlfluff 484, pydicom 336), hashed 50 per repo (salt `smith-v1`) → 200 tasks, sha256 `339b5c42…`; strategies:
  func_basic 105, func_pm_* 65, combine_file 15, lm_rewrite 15. Guard (`experiments/smith-v1-guard.json`): clean
  against Verified; against dev, only the repositories are shared (no ids, commits or issue text).
- **G0 (node04, all four images):** one commit at the mirror's main, clean tree, one remote (`origin`, removed by our
  setup), every sampled bug patch applies (`git apply --check`), testbed python 3.10/3.11, pytest 8.3.
- **G1 grading gate** (node04, 17:03–18:27, `rolling_probe --episodes 0`, 4 workers; `experiments/gate-smith-v1.json`):
  **179 of 200 pass** (gold = 1 with every F2P and P2P test, empty = 0). marshmallow 50/50, astroid 50/50, sqlfluff
  50/50, **pydicom 29/50**: in all 21 failures the gold patch passes its F2P tests but a few P2P tests fail offline
  (e.g. 389/394, 500/511, 42/43), the same network-dependent pydicom tests seen on the dev split. The screen will run on
  the 179 gated tasks; a P2P-exclusion calibration for pydicom (drop the tests that fail under gold offline) is the
  follow-up if pydicom's diversity is wanted.

**grpo-gemma-v1 progress** (steps 1–7, 16:50–18:43; 13–22 min per step): policy reward 0.19, 0.16, 0.00, 0.16, 0.06,
0.16 (steps 2–7: 6, 5, 0, 5, 2, 5 of 32); degenerate groups 2 of 4 on every step but step 4 (4 of 4, no update);
16 trajectories trained per informative step; loss −0.05 to −0.20, grad norm 0.065–0.072, objective gain +0.0005 to
+0.0010 on every trained step; served drift vs base 0.07–0.12 nats per token; mismatch mean ≈ 0.03 on 46k–62k tokens.

---

## 2026-09-24 s9-qwen3-1.7b-v1 — S9: the diagram's tool-call protocol with a model that follows it

- Gate: S9 (`docs/02-phase2-roadmap.md`). Format part **passes**; reward part (≥ 1 non-oracle rollout earns reward) **fails**.
- Config: `configs/s9_qwen3_1.7b.yaml`. Qwen3-1.7B @70d244cc runs in non-thinking mode with `protocol: tool_call`. Everything else matches `basic_run.yaml`, except `max_train_tokens` rises to 16,384.
- Node/GPU: node08. vLLM is on GPU0 (`make serve MODEL=qwen3-1.7b`, tmux `swe-serve`), the trainer on GPU1 (log `runs/train-20260924-041556.log`).
- Records: `experiments/runs/s9-qwen3-1.7b-v1/`. Trajectories and adapters are node-local under `/var/tmp/swe/{trajectories,adapters}/s9-qwen3-1.7b-v1`.

### Model probe first (`scripts/s9_probe.py`: 4 pool tasks × 4 episodes, 20 turns, no training)

| Model | Protocol | Valid actions | Turn-1 submit | Mean turns | Non-empty patches | Resolved | Record |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-0.5B | tool_call | 0 % (0/48 turns) | — | — | 0 | 0 | S6 entry below |
| Qwen2.5-Coder-1.5B | tool_call | **0 %** (16/16 exit on format errors) | 0 % | 3.0 | 0 | 0/16 | `s9-probe-1.5b.json` |
| Qwen2.5-Coder-1.5B | backticks | 94 % | 0 % | 11.0 | 2 | 0/16 | same |
| Qwen2.5-Coder-7B-AWQ | tool_call | 0 valid calls (no JSON kept) | | | | | |
| **Qwen3-1.7B** | tool_call | **97 %** | 0 % | 16.1 | 3 | 0/16 | `s9-probe-qwen3-1.7b.json` |
| Qwen3-1.7B | backticks | 89 % | 0 % | 11.2 | 0 | 0/16 | same |

Every Qwen2.5-Coder size writes the JSON inside a ```json markdown fence instead of `<tool_call>` tags, so the harness counts a format error. Qwen3 emits native `<tool_call>` tags. The 1.5B and 7B-AWQ snapshots were deleted after the probe; `sarl.paths.MODELS` keeps their revisions.

Code this needed (this commit):
- `sarl.paths.MODELS` + `model_snapshot()`; `20_serve_policy.sh` takes `SARL_MODEL=<key>`.
- `ChatFormat(template_kwargs=…)` defaults `enable_thinking=False`.
- The Qwen3 template is not append-only: it adds `<think>\n\n</think>\n\n` to the last assistant turn. The delta probe therefore cuts at the reply's `<|im_end|>`, and the tool-delta suffix ends with the empty think block. The new test `test_qwen3_nonthinking_deltas_and_episode` covers this.

### Run: 10 steps × 2 tasks × G = 8 (7 policy + 1 oracle), lr 1e-5, LoRA r 8, seed 0

| Step | Tasks | Policy resolved | Oracle | Degenerate | Trained (dropped > 16k) | Tokens | Loss | Grad norm | J gain | Served Δlogp | Mismatch mean / p99 / max | Valid actions | Non-empty patches | Seconds (roll / grade / total) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | pydicom-1694, astroid-1268 | 0/14 | 2/2 | 0 | 11 (5) | 9,363 | 1.598 | 3.61 | -6.4e-03 | 0.0809 | 0.0037 / 0.124 / 1.423 | 99% | 2 | 40.9 / 5.8 / 91.8 |
| 2 | pydicom-1694, astroid-946 | 0/14 | 2/2 | 0 | 14 (2) | 12,404 | 1.274 | 2.85 | 5.8e-03 | 0.0933 | 0.0017 / 0.029 / 1.138 | 98% | 1 | 51.7 / 5.6 / 114.0 |
| 3 | sqlfluff-1625, astroid-1268 | 0/14 | 2/2 | 0 | 12 (4) | 7,695 | 1.581 | 3.60 | 8.6e-03 | 0.0914 | 0.0035 / 0.101 / 0.949 | 89% | 1 | 35.2 / 10.6 / 83.4 |
| 4 | astroid-1268, astroid-1614 | 0/14 | 2/2 | 0 | 9 (7) | 5,951 | 1.909 | 4.94 | -3.3e-04 | 0.1174 | 0.0035 / 0.102 / 2.121 | 99% | 1 | 38.1 / 5.6 / 71.0 |
| 5 | sqlfluff-1625, astroid-1614 | 0/14 | 2/2 | 0 | 13 (3) | 11,060 | 1.355 | 3.46 | 1.2e-02 | 0.0851 | 0.0015 / 0.024 / 0.697 | 99% | 0 | 40.7 / 10.4 / 85.5 |
| 6 | pydicom-1256, sqlfluff-1625 | 0/14 | 2/2 | 0 | 15 (1) | 13,162 | 1.292 | 3.06 | -1.9e-03 | 0.0887 | 0.0031 / 0.092 / 0.948 | 95% | 1 | 39.6 / 10.6 / 89.3 |
| 7 | astroid-1614, sqlfluff-1625 | 0/14 | 2/2 | 0 | 12 (4) | 9,941 | 1.454 | 3.82 | 7.8e-03 | 0.0996 | 0.0025 / 0.076 / 1.182 | 93% | 0 | 43.7 / 10.3 / 94.8 |
| 8 | pydicom-811, astroid-1268 | 0/14 | 2/2 | 0 | 13 (3) | 10,475 | 1.274 | 3.02 | 1.0e-02 | 0.0923 | 0.0039 / 0.117 / 0.970 | 92% | 5 | 40.8 / 6.0 / 84.5 |
| 9 | pydicom-811, pydicom-1694 | 0/14 | 2/2 | 0 | 14 (2) | 11,777 | 1.093 | 2.66 | -2.0e-05 | 0.1063 | 0.0050 / 0.161 / 1.112 | 91% | 3 | 42.0 / 5.0 / 83.6 |
| 10 | marshmallow-1359, sqlfluff-1625 | 0/14 | 2/2 | 0 | 12 (4) | 8,670 | 1.416 | 3.33 | -5.4e-04 | 0.0974 | 0.0026 / 0.074 / 0.848 | 98% | 0 | 38.4 / 10.4 / 82.0 |

**Reward.**
- Policy: **0 of 140** rollouts resolved. 11 patches were non-empty and 10 applied, but no fail-to-pass test passed.
- Oracle: 20 of 20 resolved, with 0 degenerate groups (the LANG fix holds).

**Behaviour: the policy explores and loops; it rarely edits.**
- Commands: 1,717 of 2,142 turns were `grep`. Only 109 commands look like edits (`sed -i`, heredocs, `python -c`).
- Repetition: a rollout repeats an identical command 11 times on average, and 100 of 140 rollouts repeat one at least 5 times.
- Exits: 87 `max_turns`, 35 `context`, 10 `format_errors`, 8 `submitted`.
- Sentinel: 13 turns ran the bare sentinel without `echo`; the harness does not count those as submits.

**Dropped trajectories.** All 35 `context` exits exceeded `max_train_tokens` (16,384) and were left out of the update, so 25 % of the policy's samples never reached the trainer. The longest was 21,493 tokens and the median 4,680.

**Update.**
- All positive advantage again came from the off-policy oracle, as in S8.
- At step 1 the oracle's mean per-token logp under π was −3.1 / −3.5, against ≈ −0.02 for π's own samples: π is near-deterministic and far from the gold trajectory.
- The update moved things only a little: J gain was positive on 5 of 10 steps (−6.4e-03 to 1.2e-02), and the oracle's logp rose on 14 of 20.
- Loss was 1.09–1.91. Grad norm was 2.7–4.9, so every step was clipped to 1.0.

**Plumbing health.**
- Weight sync: `served_dlogp_vs_base` 0.081–0.117, about 4× S8 (larger clipped steps on a bigger model).
- Train–inference mismatch: the pooled mean stays at 0.0015–0.0050 (p99 0.024–0.161). The tail is new, though: max 0.70–2.12, and 0.02–0.12 % of tokens above 0.5 (S8: 0). This matters once the update uses importance ratios.
- Timing: 71–114 s per step, about 15 min for the run, with no stragglers under the 60 s / 600 s limits.

**Verdict.** Qwen3-1.7B is the smallest model tried that follows the diagram's protocol, but it solves none of the 8 pool tasks. Every policy group is therefore all-zero apart from the oracle, and GRPO has nothing on-policy to reinforce. The failure is too little capability for any signal, not too little headroom. The next model or setting has to land in the band the user asked for: a non-zero but low pass rate on the pool, so that groups contain mixed rewards and RL still has room to show a gain.

### Reproduce

```bash
make -C swe-bench serve MODEL=qwen3-1.7b     # tmux swe-serve, GPU0 :8001
ssh node08 'cd /home/yashp/Agentic-RL/swe-bench && ../.venv/bin/python scripts/s9_probe.py --model-key qwen3-1.7b --grade --out experiments/s9-probe-qwen3-1.7b.json'
ssh node08 "tmux new -d -s swe-train 'bash /home/yashp/Agentic-RL/swe-bench/scripts/21_train.sh configs/s9_qwen3_1.7b.yaml --run-name s9-qwen3-1.7b-v1'"
make -C swe-bench test NODE=node08           # 27 tests
```

### Open after this entry

- Measure pass@8 on the pool before the next training run, for Qwen3-4B (bf16) and for Qwen3-1.7B with thinking on. Pick the setting whose pass rate falls in about 5–30 %.
- Harness changes worth trying with any model:
  - accept the bare sentinel as a submit;
  - nudge the model after repeated identical commands;
  - truncate-and-train overlong trajectories instead of dropping them.
- S10 (Verified eval, `configs/eval_verified_v1.txt`) waits until a model/setting is chosen.

---

## 2026-09-24 — Adversarial review of sarl, fixes (`0801504`), S8 re-runs, straggler finding

Later the same day as the phase-1 entry below, on node08. The final S8 run is **`s8-basic-v3b`**; it supersedes the S8 numbers below.
- Record copies (config, metrics, log, rollouts, summary): `experiments/runs/{s8-basic-v2, s8-basic-v3b}/`. The originals are in the gitignored `runs/`.
- `s8-basic-v3b`'s per-token trajectories are node-local on node08 (`/var/tmp/swe/trajectories/s8-basic-v3b`).
- The anchor notebook (`notebooks/build_notebook.py` → `01_swe_rl_anchor.ipynb`) now reads `s8-basic-v3b`.

| Time | Event |
|---|---|
| 01:25:13–01:32:53 | `s8-basic-v2`: S8 re-run after the LANG fix, before the review (reference sampling, network on) |
| 01:35 | commit `a8f854f` (the `sarl` library) |
| — | Adversarial review of `sarl`: 15 findings |
| 02:01:40 | `s8-basic-v3` started with the fixes but stalled on 120 s offline pip retries, before the pip/conda fast-fail fix. Aborted; its run dir was deleted. |
| 02:05:28 | S5 pool gate re-run offline → `experiments/pool-gate.json` |
| 02:05:57–02:40:57 | `s8-basic-v3b`: the final S8 run, with every fix |
| 02:06:54 | commit `0801504` (the review fixes) |
| after the run | `configs/basic_run.yaml`: `command_timeout_s: 60`, `wall_clock_s: 600` (the straggler finding below) |

### Review findings

14 of 15 were confirmed: 13 are fixed in `0801504`, and one is kept by design. One was refuted.

| # | Finding | Verdict | Fix |
|---|---|---|---|
| 0 | The grading bind mount was writable, so code under test (fakeroot uid 0) could rewrite `instance.json` or the log and forge reward 1 | confirmed | Only an input folder (`patch.diff`, `eval.sh`) is bound, read-only, at `/sarl_in`. The spec, log and report are host-only; the log is the container's stdout, captured by the host. |
| 1 | Sandboxes shared the host network: they could reach the vLLM LoRA admin API on :8001, the cluster, and the internet (including upstream fix commits) | confirmed | `--network none` by default for rollouts and grading; pip and conda fail fast offline (`PIP_NO_INDEX=1`, `PIP_DISABLE_PIP_VERSION_CHECK=1`, `CONDA_OFFLINE=true`) |
| 2 | Crashed rollouts, failed grades and infra failures counted as reward 0 | confirmed | Left out of their group; a group with < 2 valid members is skipped |
| 3 | Over-long trajectories are dropped after the advantages are computed | kept by design | DAPO-style overlong filtering: their reward still sets the group baseline. Counted per step as `dropped_overlong`. |
| 4 | `patch()` could be truncated, merged with stderr, or miss committed changes | confirmed | Diff against the HEAD pinned at sandbox start, stdout only, never truncated; a git failure raises |
| 5 | Grading workdirs were never deleted; logs were unbounded | confirmed | Workdirs deleted after grading; logs capped at 2 MB (head + tail) |
| 6 | "The empty-patch gate never runs the tests" | refuted | `run_empty=True` runs them. `pool-gate.json` now records the empty run's counts (e.g. F2P 0/1, P2P 155/155). |
| 7 | A policy patch could edit the test harness | confirmed | Sections touching `conftest.py`, `pytest.ini`, `tox.ini`, `setup.cfg`, `noxfile.py`, test files or test dirs are dropped and reported (`filtered_paths`) |
| 8 | Rollouts were sampled with top_p 0.95, top_k 20 and rep. 1.05 but trained as the raw π | confirmed | Plain sampling by default; the reference settings kept as `REFERENCE_SAMPLING` |
| 9 | Scripted-oracle truncation: executed text ≠ trained ids | confirmed | Truncated text is decoded from the kept ids |
| 10 | Apptainer instances could leak | confirmed | Run-prefixed sandboxes are swept at start and exit; half-started instances are stopped |
| 11 | Mismatch max and p99 were means of per-trajectory statistics | confirmed | Pooled over all sampled tokens of a step (also logs `n` and `p50`) |
| 12 | LoRA init (torch) and vLLM sampling were unseeded (task sampling was already seeded) | confirmed | Torch seed plus a per-request vLLM seed; the run tag is now `sha256(run name)[:6]` |
| 13 | Untrusted text could forge `<\|im_end\|>` turn boundaries | confirmed | Special-token literals in the issue and observations are neutralised with a word joiner (U+2060) |
| 14 | Per-token trajectories were written to NFS | confirmed | gzip'd to node-local `/var/tmp/swe/trajectories/<run>/` |

**Tests.** 26 pass on node08 (`make -C swe-bench test NODE=node08`). New since the phase-1 entry: the harness filter, special-token forging, oracle truncation, commit capture, and a no-network sandbox.

### S5 pool gate, re-run offline: pass

`scripts/pool_gate.py` now grades with `--network none` by default (`--network` restores the host network). All 8 tasks score gold = 1 and empty = 0 with every PASS_TO_PASS green, as in the first gate. The empty-patch grades run the tests: every task gives F2P 0/1 and P2P all green (e.g. astroid-946 P2P 155/155).

### S8 re-runs: `s8-basic-v2` and `s8-basic-v3b`

Same schedule as the first S8 run (10 steps × 2 tasks × G = 8, 7 policy + 1 oracle, protocol `backticks`, lr 1e-5, LoRA r 8 on q,k,v,o, seed 0, 20 turns, 16k context, 8,192 training tokens, 1,800 s episode wall clock, 120 s per command). Task sampling is seeded, so all three runs drew the same tasks per step.

| | `s8-basic-lr1e-5` | `s8-basic-v2` | `s8-basic-v3b` |
|---|---|---|---|
| Code | before the LANG fix | LANG fix, before the review | all review fixes (`0801504`) |
| Sampling | reference (top_p 0.95, top_k 20, rep. 1.05) | reference | plain |
| Network (rollouts, grading) | on | on | none |
| Mismatch statistic | mean of per-rollout stats | mean of per-rollout stats | pooled over tokens |
| Oracle resolved | 18/20 | 20/20 | 20/20 |
| Degenerate groups | 2 (pydicom-811, steps 8–9) | 0 | 0 |
| Rollout / grading errors | 0 | 0 | 0 |
| Policy resolved | 0/140 | 0/140 | 0/140 |
| Non-empty policy patches | 3 (2 applied) | 7 (6 applied) | 3 (2 applied; 1 became empty after the harness filter) |
| Exits: submitted / max_turns / format_errors / context / wall_clock | 70 / 41 / 21 / 8 / 0 | 86 / 40 / 8 / 6 / 0 | 70 / 52 / 17 / 0 / 1 |
| Submitted on turn 1 | 19 | 26 | 27 |
| Valid actions (turns) | 1,020 / 1,141 (89.4 %) | 1,024 / 1,071 (95.6 %) | 1,186 / 1,282 (92.5 %) |
| Valid-action rate per step | 0.69–0.95 | 0.83–1.00 | 0.74–0.99 |
| Trained / dropped overlong | 136 / 8 | 152 / 8 | 156 / 4 |
| J gain > 0 | 10/10 | 7/10 | 7/10 |
| J gain range (mean) | 4.9e-05 to 2.4e-03 (7.7e-04) | −1.3e-03 to 2.0e-03 (5.7e-04) | −6.7e-04 to 2.0e-03 (5.5e-04) |
| Oracle Δlogp > 0 | 14/18 | 14/20 | 12/20 |
| Served Δlogp vs base | 0.0175–0.0285 | 0.0212–0.0274 | 0.0196–0.0276 |
| Mismatch mean | 0.0047–0.0100 | 0.0045–0.0094 | 0.0018–0.0077 |
| Mismatch p99 | 0.059–0.098 | 0.056–0.091 | 0.058–0.114 |
| Mismatch max | 0.142 | 0.137 | 0.401 |
| Tokens with \|Δlogp\| > 0.5 | 0 % | 0 % | 0 % |
| Commands that hit the timeout | 0 | 1 | 15 |
| Step time | 26–101 s | 18–155 s | 18–46 s; step 1 1,818 s |
| After grading (diagnostics, update, sync) | 2.7–7.7 s | 4.1–5.9 s | 4.2–7.4 s |
| Run wall clock | 10 min | 8 min | 35 min |
| Status | superseded | superseded | **final** |

The mismatch rows are not directly comparable across the columns. The first two runs averaged per-rollout statistics, so their "max" is a mean of per-rollout maxima. The pooled max of v3b (0.401) is the largest single-token gap in a step, and the pooled mean weights long trajectories more.

**`s8-basic-v3b` per step.** Policy resolved 0/14 and degenerate 0/2 on every step.

| Step | Tasks | Oracle | Trained (dropped > 8192) | Tokens | Loss | Grad norm | J gain | Served Δlogp | Mismatch n / mean / p99 / max | Valid actions | Seconds (roll / grade / total) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | pydicom-1694, astroid-1268 | 2/2 | 15 (1) | 5,254 | 0.515 | 0.634 | 5.1e-04 | 0.0251 | 4,766 / 0.0018 / 0.058 / 0.189 | 88% | **1807.0** / 5.1 / **1818.0** |
| 2 | pydicom-1694, astroid-946 | 2/2 | 16 (0) | 1,955 | 0.446 | 0.634 | −2.9e-04 | 0.0234 | 1,525 / 0.0071 / 0.102 / 0.182 | 96% | 8.5 / 5.6 / 18.3 |
| 3 | sqlfluff-1625, astroid-1268 | 2/2 | 16 (0) | 4,324 | 0.475 | 0.596 | −6.7e-04 | 0.0276 | 3,845 / 0.0033 / 0.082 / 0.344 | 74% | 13.4 / 10.2 / 28.7 |
| 4 | astroid-1268, astroid-1614 | 2/2 | 16 (0) | 4,226 | 0.394 | 0.616 | 8.3e-04 | 0.0237 | 3,738 / 0.0066 / 0.107 / 0.401 | 92% | 13.7 / 5.4 / 24.6 |
| 5 | sqlfluff-1625, astroid-1614 | 2/2 | 16 (0) | 3,768 | 0.408 | 0.630 | 9.4e-04 | 0.0241 | 3,323 / 0.0052 / 0.101 / 0.294 | 77% | 14.4 / 10.1 / 29.3 |
| 6 | pydicom-1256, sqlfluff-1625 | 2/2 | 15 (1) | 3,384 | 0.461 | 0.653 | −7.3e-05 | 0.0220 | 2,952 / 0.0074 / 0.114 / 0.290 | 91% | 22.9 / 10.5 / 38.8 |
| 7 | astroid-1614, sqlfluff-1625 | 2/2 | 16 (0) | 2,691 | 0.398 | 0.622 | 7.9e-04 | 0.0196 | 2,246 / 0.0077 / 0.106 / 0.291 | 79% | 13.5 / 10.3 / 28.4 |
| 8 | pydicom-811, astroid-1268 | 2/2 | 16 (0) | 4,308 | 0.443 | 0.558 | 1.0e-03 | 0.0229 | 3,793 / 0.0051 / 0.107 / 0.238 | 99% | 27.1 / 5.2 / 39.0 |
| 9 | pydicom-811, pydicom-1694 | 2/2 | 16 (0) | 7,027 | 0.447 | 0.576 | 4.0e-04 | 0.0257 | 6,546 / 0.0036 / 0.084 / 0.232 | 87% | 16.0 / 4.3 / 27.7 |
| 10 | marshmallow-1359, sqlfluff-1625 | 2/2 | 14 (2) | 2,884 | 0.497 | 0.719 | 2.0e-03 | 0.0216 | 2,444 / 0.0027 / 0.082 / 0.160 | 84% | 31.1 / 10.4 / 45.8 |

**pydicom-811 at steps 8–9.** The oracle resolved it in both re-runs, so the LANG fix removed the only cause of degenerate groups in the first run.

**Harness filter in action.** In v3b, 2 policy patches had sections dropped: one only created `tests/test_error.py` and became empty (step 1, pydicom-1694); one lost its `setup.cfg` and `tox.ini` sections, and the rest applied (step 4, astroid-1614).

**J gain is not positive on every step.** J gain is the post-update change of the surrogate (1/N) Σ A_i · mean log p_i on the update batch. With one AdamW step at lr 1e-5 it is not guaranteed to rise on every step: v2 and v3b each had 3 tiny negatives. Oracle and policy trajectories also share tokens (e.g. the submit block) with advantages of opposite sign.

**Straggler finding.** Step 1 of v3b took 1,818 s, against 18–46 s for the other nine steps.
- One policy rollout on pydicom-1694 (idx 5) sent the same `pip install --upgrade pydicom -r …/requirements.txt` + `pytest /testbed` command on all 15 of its turns (the first with a relative requirements path).
- pip fails fast offline, but `pytest /testbed` runs the whole pydicom suite, which takes more than 120 s. Every command hit the 120 s timeout (rc 124), until the 30-min episode wall clock ended the episode at 1,806 s.
- The median policy episode took 2.3 s. The synchronous loop waited for the straggler before grading and training.
- **Fix now:** `configs/basic_run.yaml` sets `command_timeout_s: 60` and `wall_clock_s: 600` (the diagram: 30 min). This bounds one straggler at about 10 minutes but does not remove the wait. Async rollouts and straggler handling are on the roadmap (`docs/02-phase2-roadmap.md`).
- v3b itself ran with the old limits (1,800 s, 120 s), as its `config.json` records.

**Interpretation.** Unchanged from the first run. All positive advantage went to the off-policy oracle, and the policy resolved 0/140 in every run. The fixes make the reward and the sandbox trustworthy, and the loop is proven end to end offline; no learning from earned reward has been shown.

### Reproduce

```bash
cd /home/yashp/Agentic-RL
ssh node08 'cd /home/yashp/Agentic-RL/swe-bench && ../.venv/bin/python scripts/pool_gate.py configs/pool_dev.txt --json experiments/pool-gate.json'   # S5, offline by default
make -C swe-bench serve                      # tmux swe-serve, GPU0 :8001
ssh node08 "tmux new -d -s swe-train 'bash /home/yashp/Agentic-RL/swe-bench/scripts/21_train.sh configs/basic_run.yaml --run-name s8-basic-v3b'"   # S8 final
make -C swe-bench test NODE=node08           # 26 tests
.venv/bin/sarl summarize swe-bench/runs/s8-basic-v3b
```

The current `basic_run.yaml` uses the tighter limits (60 s, 600 s). To repeat v3b exactly, set `command_timeout_s: 120` and `wall_clock_s: 1800`.

### Open after this entry

- terminal-bench's A0/B1/B2 driver takes both node08 GPUs next.
- Then S9 (tool-call protocol, via SFT or a larger model) and S10 (Verified eval); see `docs/02-phase2-roadmap.md`.
- Async rollouts and straggler handling.

---

## 2026-09-24 — Phase 1: gates S0–S8 on node08

node08 had just come back from reprovisioning, so every node-local cache was rebuilt from scratch. All times are PDT; they come from logs and file mtimes.

**Partly superseded** by the entry above: the S8 numbers here are from the first run (`s8-basic-lr1e-5`, before the LANG and review fixes), and `experiments/pool-gate.json` now holds the offline re-run of S5. S0, S1, S4, S6 and S7 stand as written.

| Time | Event |
|---|---|
| ≈00:43 | S0: pinned model and datasets downloaded on the headnode (29 s, download only) |
| 00:56:28–00:57:45 | S6: vLLM 0.25.1 policy server up on node08 GPU0 :8001 |
| 01:02:32 | S7 attempt 1 (1 step): passed, but only the greedy text probe was logged as sync evidence |
| 01:03:42 | S7 attempt 2 (2 steps): crashed on an adapter name collision (fixed below) |
| 01:06:44–01:08:01 | S7 attempt 3 (2 steps): **pass** → `runs/gate-s7` |
| 01:08:26–01:18:29 | S8: 10 steps → `runs/s8-basic-lr1e-5` (superseded by `s8-basic-v3b`) |
| 01:20:54 | LANG fix in `env/apptainer.py` |
| 01:21:18 | S5 pool gate: 8/8 → `experiments/pool-gate.json` (network on; the file was overwritten by the offline re-run at 02:05) |
| 01:21:29 | S1 attempt 1: failed in 10 s (HF offline mode) |
| 01:23:24–01:24:48 | S1 attempt 2: **pass** → `runs/phase1/20260924-012324`, `experiments/phase1-reference.json` |

S2 and S3 have no separate record. The offline pieces are covered by the test suite, which passes on node08 (23 tests at the time, 26 after the review fixes; `make -C swe-bench test NODE=node08`):
- MockEnv parity with the reference;
- the protocol parser;
- the token-exact harness;
- the GRPO math against the upstream code;
- the dataset pins and disjointness.

### S0 — pinned assets: pass

| Asset | Revision | Size / count |
|---|---|---|
| `Qwen/Qwen2.5-Coder-0.5B-Instruct` | `ea3f2471cf1b1f0db85067f1ef93848e38e88c25` | 954 MB in `~/hf-cache` |
| `SWE-bench/SWE-bench`, split `dev` (training) | `c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2` | 225 tasks, 6 repos |
| `SWE-bench/SWE-bench_Verified`, split `test` (eval only) | `78f471bf655a3137b2e8a75af1501690ec009ec3` | 500 tasks, 12 repos |
| `princeton-nlp/SWE-bench_Verified` (the reference's copy) | `c104f840cc67f8b6eec6f759ebc8b2693d585d4a` | 500 tasks |

- Dev vs Verified: **0** shared repos, instance ids or base commits (`contamination_report`, `test_splits_and_disjointness`).
- **Offline loading.** `load_dataset(revision=…)` still calls the Hub, and huggingface_hub rejects partial snapshots offline. `sarl.tasks` reads the pinned parquet files straight from the snapshot directories.
- **Pool v1** (`configs/pool_dev.txt`): 8 dev tasks chosen by `select_pool` (single-file gold patch ≤ 700 characters, 1 F2P, ≥ 1 P2P), then by image size.

### S1 — reference as published, on node08 GPU1: pass (second attempt)

`scripts/phase1_reference.sh` runs the upstream script and the 44-cell notebook (via nbconvert, kernel `sarl`) unchanged, except for the notebook's two `!pip` lines. The task is Verified `astropy__astropy-12907`, so the results are tagged `contaminates_eval: true` and never reused.

- **Attempt 1 (01:21) failed.** Both the script and the notebook died in `load_dataset("princeton-nlp/SWE-bench_Verified")` with `ConnectionError … (OfflineModeIsEnabled)`, 10 s in.
  - The upstream code loads the model and dataset **without a revision** (Colab style), which offline mode cannot resolve.
  - **Fix:** the script unsets `HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE` and `TRANSFORMERS_OFFLINE` for this run only. Afterwards it checks what `main` resolved to against our pins (`revisions.txt`: both "revision ok").
  - The attempt-1 run dir no longer exists.
- **Attempt 2 (01:23): both parts completed** (torch 2.11.0+cu128, transformers 5.14.1, peft 0.19.1).

| | Script (`GROUP_SIZE = 6`, no oracle) | Notebook (G = 8: 7 model + oracle) |
|---|---|---|
| Rollouts that wrote a patch | 2/6 | 2/7 model + the oracle |
| Rewards (`reward_random`) | 0, 0, 0.637, 0, 0.27, 0 | 0, 0.637, 0, 0.27, 0, 0, 0, **oracle 0.041** |
| Mean / std reward | 0.151 / 0.239 | 0.119 / 0.214 |
| Degenerate group | no | no |
| Trainable parameters | 1,081,344 of 495,114,112 (0.2184 %) | same |
| Loss / grad norm | −0.04545 / 0.7800 | −0.05207 / 0.6067 |
| Δ mean log-prob after the step | all 6 negative (−0.0002 to −0.0077) | all 8 negative (−0.0002 to −0.0054) |
| Wall clock / peak RSS | 37.9 s / 3.0 GB | — (22 code cells, 0 errors) |

Observations:
- **The proxy reward penalised the correct fix.** The notebook's oracle drew 0.041 from `reward_random`, so its advantage was −0.361, below two model rollouts that only wrote something (0.637 and 0.27).
- **The 0.5B model reproduces the lecture's failure mode** (1:17–1:26). The §3.3.1 rollout ran `python setup.py install`, `python -m unittest tests` and `python -m pytest tests` against the mock and wrote no patch.
- **One step moved every log-prob down.** With one AdamW step at lr 1e-5, even the positive-advantage trajectories' mean log-probs went down slightly.
- **Script and notebook rollouts match.** The notebook's model rollouts 0–4 have the same supervised-token counts and pre-update log-probs as the script's rollouts 1–5. Both seed identically, and the notebook's §3.3.1 demo rollout takes the first sample.

### S4 — apptainer sandbox: pass

- `apptainer instance start --fakeroot --containall --writable-tmpfs <sif> <name>` works unprivileged on node08 (apptainer 1.5.2).
  - `instance start` has no `--pid` flag.
  - Inside: uid 0, `/testbed`, conda env `testbed`.
  - State persists across execs, at about 0.7 s per exec.
  - The in-container `timeout -k 5` kills `sleep 600`.
- The image's HEAD is a commit named "SWE-bench", so the patch is diffed against the HEAD pinned at sandbox start (after `git add -A`, `--binary`), which also captures anything the agent commits.
- The SIFs for the 8 pool tasks are about 1 GB each, in node08 `/tmp/swe-sif`.

### S5 — test reward: pass (after the LANG fix)

- Grader: `swebench==5.0.2` in the node-local venv `/var/tmp/swe/grader-venv`. It uses `make_test_spec(row).eval_script` and `get_eval_report`; `sarl.reward.grade` runs a fresh container with the apply chain and `eval.sh`.
- **Finding.**
  - `--containall` implies `--cleanenv`, which leaves `LANG=C` in the container. Python 3.6 testbeds then use an ASCII filesystem encoding.
  - As a result, pydicom-811's pass-to-pass test `testUTF8FileName` failed **even for the gold patch** (84/85 P2P).
  - `scripts/pool_gate.py` found this over the whole pool. The same bug had already hit S8 (below).
- **Fix:** `APPTAINERENV_LANG=C.UTF-8` and `APPTAINERENV_LC_ALL=C.UTF-8` in `env.apptainer.apptainer_env()`. `APPTAINERENV_*` variables survive `--cleanenv`.
- **Result** (`experiments/pool-gate.json`, network on): 8/8 tasks score gold = 1 and empty = 0 with every test green. About 5–13 s per grade. Superseded by the offline re-run in the entry above, with the same 8/8 result; the file now holds that re-run.

| Task | F2P | P2P |
|---|---|---|
| pylint-dev__astroid-946 | 1/1 | 155/155 |
| marshmallow-code__marshmallow-1359 | 1/1 | 76/76 |
| pydicom__pydicom-811 | 1/1 | 85/85 |
| pylint-dev__astroid-1614 | 1/1 | 9/9 |
| sqlfluff__sqlfluff-1625 | 1/1 | 64/64 |
| pydicom__pydicom-1256 | 1/1 | 22/22 |
| pydicom__pydicom-1694 | 1/1 | 26/26 |
| pylint-dev__astroid-1268 | 1/1 | 91/91 |

### S6 — serve + token-exact harness: pass; protocol finding

- **Server.** vLLM 0.25.1 serves the 0.5B model on GPU0 :8001 with runtime LoRA. Startup log: weights 0.96 GiB, CUDA graphs 1.03 GiB, KV cache 11.99 GiB = 1,047,440 tokens, engine init 44.8 s.
- **Endpoint.** `/v1/completions` on token-id prompts returns `token_ids`, including the `<|im_end|>` stop token, plus `token_logprobs`.
- **Token exactness.** The harness is token-exact on real rollouts: the incremental ids equal the one-shot render tokenisation.
- **Protocol finding.** Qwen2.5-Coder-0.5B produced **0 of 48** valid JSON tool calls over 16 real episodes, even with an example in the prompt.
  - With the reference's fenced-bash protocol (`backticks`), **45 of 49 turns (92 %)** were valid, but 14 of 16 episodes submitted immediately by copying the submit block.
  - This matches lecture 2:50: a model that does not know the action space needs SFT or a larger model first.
  - Consequence: `basic_run.yaml` uses `protocol: backticks`, and the one oracle rollout per group supplies the learning signal. The move back to the diagram's protocol is S9 (`docs/02-phase2-roadmap.md`).
- The probe counts come from the gate session; no probe artifact is kept in `runs/`.

### S7 — one GRPO step end to end: pass (`runs/gate-s7`)

Config: `basic_run.yaml` with `--steps 2`. G = 8 (7 policy + 1 oracle), 2 tasks per step.

| Step | Tasks | Policy resolved | Oracle | Degenerate | Trained (tokens) | Loss / grad norm | Served Δlogp vs base | Mismatch mean / p99 / max | Seconds (roll / grade / total) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | pydicom-1694, astroid-1268 | 0/14 | 2/2 | 0 | 16 (4,333) | 0.492 / 0.609 | 0.0172 | 0.0055 / 0.054 / 0.086 | 14.7 / 5.4 / 23.7 |
| 2 | pydicom-1694, astroid-946 | 0/14 | 2/2 | 0 | 16 (9,855) | 0.504 / 0.594 | 0.0223 | 0.0051 / 0.069 / 0.140 | 28.9 / 5.7 / 41.0 |

- Policy turns were 272/272 valid. Exits: 16 submitted, 12 `max_turns`. 2 of 28 policy patches were non-empty and 1 applied.
- The mismatch columns are means of per-rollout statistics; pooling over tokens came with the review fixes (entry above).
- **Fix 1: adapter name collision.** Attempt 1 loaded `policy-s1`. In attempt 2, vLLM rejected the same name: `400 … The lora adapter 'policy-s1' has already been loaded`. Names must be unique for the server's lifetime.
  - Names are now run-tagged (`policy-<tag>-s<step>`).
  - `LoraSync.unload_leftovers()` unloads earlier runs' adapters at start; attempt 3 logged `unloaded leftover adapters: ['policy-s1']`.
- **Fix 2: sync evidence.** Attempt 1's only sync evidence was a greedy text probe, and it did not change. `served_dlogp_vs_base` was added: vLLM's `prompt_logprobs` of a fixed sequence under the new adapter vs the base model.

### S8 — basic run: pass as plumbing; no policy learning signal (`runs/s8-basic-lr1e-5`, superseded)

**Superseded by `s8-basic-v3b`** (entry above). This first run predates the LANG fix and the review fixes: reference sampling, network on, and mismatch reported as means of per-rollout statistics.

Config: `configs/basic_run.yaml` as it was then.
- 10 steps × 2 tasks × G = 8 (7 policy + 1 oracle); protocol `backticks`; seed 0.
- lr 1e-5; LoRA r 8 / α 16 on q,k,v,o (1,081,344 parameters).
- Limits: 20 turns, 16k context, 8,192 training tokens.
- `objective_gain` was added to `train.py` for this run.

| Step | Tasks | Policy resolved | Oracle | Degenerate | Trained (dropped > 8192) | Tokens | Loss | Grad norm | J gain | Served Δlogp | Mismatch mean / p99 / max | Valid actions | Seconds (roll / grade / total) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | pydicom-1694, astroid-1268 | 0/14 | 2/2 | 0 | 16 (0) | 4,811 | 0.460 | 0.581 | 6.8e-04 | 0.0180 | 0.0080 / 0.091 / 0.131 | 94% | 15.4 / 5.3 / 26.1 |
| 2 | pydicom-1694, astroid-946 | 0/14 | 2/2 | 0 | 15 (1) | 3,437 | 0.521 | 0.643 | 4.9e-05 | 0.0238 | 0.0061 / 0.078 / 0.111 | 90% | 52.9 / 6.1 / 62.9 |
| 3 | sqlfluff-1625, astroid-1268 | 0/14 | 2/2 | 0 | 14 (2) | 4,572 | 0.591 | 0.659 | 1.9e-04 | 0.0228 | 0.0071 / 0.063 / 0.109 | 89% | 26.5 / 10.2 / 41.3 |
| 4 | astroid-1268, astroid-1614 | 0/14 | 2/2 | 0 | 15 (1) | 4,705 | 0.491 | 0.696 | 9.7e-04 | 0.0200 | 0.0100 / 0.098 / 0.125 | 77% | 71.6 / 5.3 / 81.8 |
| 5 | sqlfluff-1625, astroid-1614 | 0/14 | 2/2 | 0 | 14 (2) | 3,031 | 0.515 | 0.754 | 3.4e-04 | 0.0175 | 0.0088 / 0.075 / 0.109 | 82% | 86.5 / 10.3 / 101.3 |
| 6 | pydicom-1256, sqlfluff-1625 | 0/14 | 2/2 | 0 | 16 (0) | 5,296 | 0.562 | 0.595 | 3.7e-04 | 0.0225 | 0.0047 / 0.059 / 0.097 | 79% | 72.4 / 10.4 / 88.8 |
| 7 | astroid-1614, sqlfluff-1625 | 0/14 | 2/2 | 0 | 16 (0) | 8,870 | 0.464 | 0.605 | 1.3e-03 | 0.0233 | 0.0064 / 0.084 / 0.142 | 69% | 20.2 / 10.4 / 38.3 |
| 8 | pydicom-811, astroid-1268 | 0/14 | **1/2** | **1** | 7 (1) | 1,630 | 0.626 | 0.712 | 6.7e-04 | 0.0285 | 0.0067 / 0.065 / 0.103 | 90% | 48.9 / 5.3 / 57.0 |
| 9 | pydicom-811, pydicom-1694 | 0/14 | **1/2** | **1** | 8 (0) | 1,894 | 0.434 | 0.697 | 2.4e-03 | 0.0214 | 0.0064 / 0.076 / 0.141 | 95% | 39.9 / 4.6 / 47.2 |
| 10 | marshmallow-1359, sqlfluff-1625 | 0/14 | 2/2 | 0 | 15 (1) | 7,010 | 0.475 | 0.635 | 7.6e-04 | 0.0202 | 0.0070 / 0.085 / 0.136 | 71% | 30.9 / 10.4 / 48.3 |

**Policy reward.** It was 0.0 at every step: **0 of 140** policy rollouts resolved. That is expected for 0.5B.
- 3 of 140 policy patches were non-empty and 2 applied.
- Exits: 70 submitted (19 on turn 1), 41 `max_turns`, 21 `format_errors`, 8 `context`.
- 1,020 of 1,141 turns (89 %) were valid actions.

**Oracle.** 18 of 20 oracle rollouts resolved.

**Evidence the loop works:**
- **Update.** `objective_gain > 0` on **10/10** steps (4.9e-05 to 2.4e-03). The oracle trajectories' mean log-prob rose for 14 of 18.
- **Weight sync.** `served_dlogp_vs_base` 0.0175–0.0285 on every step, so vLLM serves πθ′. The greedy text probe never changed.
- **Train–inference mismatch.** Mean |Δlogp| 0.0047–0.0100, p99 0.059–0.098, max ≤ 0.142, and 0 % of tokens above 0.5.
- **Timing.** Step wall time 26–101 s (rollouts 15–87 s, grading 4.6–10.4 s). Everything after grading (diagnostics, update, sync) took 2.7–7.7 s per step. The run took 10 min.

**Degenerate groups at steps 8–9.** Both come from the LANG bug found at S5: pydicom-811's gold patch failed `testUTF8FileName` (P2P 84/85), so the oracle scored 0 and both groups were all-zero. They were skipped as designed. S8 ran before the fix; both re-runs (`s8-basic-v2`, `s8-basic-v3b`, entry above) had 0 degenerate groups.

**Interpretation.** All positive advantage went to the off-policy oracle. The update is therefore imitation of the gold-patch trajectories plus a push-down of the policy's own samples; it is not RL from earned reward. S8 proves the plumbing (serve → rollout → test reward → update → sync), not learning.

### Reproduce

The headnode may run only the fetch, the asset download and read-only summaries. Everything else runs on node08.

```bash
cd /home/yashp/Agentic-RL
make -C swe-bench fetch                      # S1 input: upstream @8cddfb60, sha256-verified, gitignored
make -C swe-bench assets                     # S0: pinned model + datasets into ~/hf-cache (headnode, download only)
make -C swe-bench sifs grader                # S4/S5 inputs on node08: /tmp/swe-sif, /var/tmp/swe/grader-venv
ssh node08 'cd /home/yashp/Agentic-RL/swe-bench && ../.venv/bin/python scripts/pool_gate.py configs/pool_dev.txt --json experiments/pool-gate.json'   # S5
ssh node08 '/home/yashp/Agentic-RL/.venv/bin/sarl episode marshmallow-code__marshmallow-1359 --policy oracle --grade'   # S4+S5 smoke test (tool_call protocol)
make -C swe-bench serve                      # S6: tmux swe-serve, GPU0 :8001
ssh node08 "tmux new -d -s swe-train 'bash /home/yashp/Agentic-RL/swe-bench/scripts/21_train.sh configs/basic_run.yaml --steps 2 --run-name gate-s7'"   # S7
ssh node08 "tmux new -d -s swe-train 'bash /home/yashp/Agentic-RL/swe-bench/scripts/21_train.sh configs/basic_run.yaml --run-name s8-basic-lr1e-5'"   # S8 (first run; today's config differs)
make -C swe-bench phase1                     # S1: upstream script + notebook on GPU1 -> experiments/phase1-reference.json
make -C swe-bench test NODE=node08           # 23 tests then, 26 now
.venv/bin/sarl summarize swe-bench/runs/s8-basic-lr1e-5   # per-step table (reads metrics.jsonl only)
```

`make -C swe-bench train` starts the same loop but cannot pass `--run-name`, so its run dir is named `<timestamp>-basic`. Check GPU use first (`make -C swe-bench status`); terminal-bench's driver needs both GPUs.

### Open after this entry

- Done (entry above): S8 re-run after the LANG fix, twice, with 0 degenerate groups both times.
- The next gates are S9 (tool-call protocol) and S10 (Verified eval); see `docs/02-phase2-roadmap.md`.
