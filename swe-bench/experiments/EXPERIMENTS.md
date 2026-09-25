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
- Steps 10–12 slowed to about 40 min and the trainer was OOM-killed at step 13 (the 32 GB tmpfs incident above).
- 21:24: resumed from step 12 with `--resume` (adapter hash verified, task draws replayed, AdamW moments reset).
- 01:37: finished.

**Speed:**
- Steps took 561–776 s at the median, against 740 s per 16-rollout step in v1. That is about 2× the rollout throughput.
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

EVAL_PENDING

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
