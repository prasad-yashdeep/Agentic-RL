# CLAUDE.md — swe-bench (Agentic-RL)

Project notes for `swe-bench/` (package `sarl`). The cluster map, standing rules and monorepo rules are in the root `CLAUDE.md`.

## What this is

RL for SWE tasks with a basic agentic harness loop. It starts from the Vizuara SWE-RL lecture and its repo:
- Lecture: YouTube `oNHzs7wi7TY` (2026-08-02), by Abhishek Goswami.
- Repo: `abgoswam/swe_in_prod_vizuara_01` @ `8cddfb60`.

The target is the repo's diagram (`reference/_upstream/swe_agent_env_interface.png`):
- the policy is served by vLLM;
- the harness owns the transcript and emits one bash action per turn;
- the environment is a sandboxed SWE-bench container;
- the reward is the tests run on the replayed patch;
- the outer loop is GRPO, with the new weights synced back into the server.

## Rules specific to this subproject

- **Upstream code is never committed.** It has no license. `reference/fetch.sh` fetches it at the pinned SHA and verifies it with sha256 into gitignored `reference/_upstream/`. `sarl` is our own implementation; tests import the fetched module only to cross-check behaviour.
- **Train on the SWE-bench *dev* split only.** Its 225 tasks come from marshmallow, pvlib, pydicom, astroid, pyvista and sqlfluff; repos, IDs and commits are all disjoint from Verified.
- **Verified is eval-only.** `sarl.tasks.contamination_report` must be clean for every pool.
- The phase-1 replica trains on Verified `astropy__astropy-12907`, as the lecture does "for illustrative purposes" (0:15). Its results are tagged `contaminates_eval` and never reused.
- **Keep large files on node08's local disks, never on `/home`** (the code refuses `/home` paths):
  - images: `/tmp/swe-sif` (16 GB partition);
  - apptainer cache and tmp, grader venv, grading workdirs, adapters: `/var/tmp/swe/…`.
- **The grader has its own venv:** `/var/tmp/swe/grader-venv` with `swebench==5.0.2`. Never install swebench into the shared venv, because its dependencies would re-resolve the CUDA stack.
- **GPU layout on node08:**
  - GPU0: vLLM policy server, tmux `swe-serve`, port 8001, runtime LoRA loading on;
  - GPU1: trainer, tmux `swe-train`.
  - The terminal-bench driver needs both GPUs, so the two projects take turns.
- **node04 (RTX 4060 Laptop, 8 GB) is the third Gemma server** (2026-09-25): tmux `swe-serve4`, port 8003,
  `SARL_PLE_OFFLOAD=1` keeps the 5.25 GiB per-layer embedding table in pinned CPU memory (`scripts/vllm_ple_offload.py`).
  node04 also holds the probe images (`/var/tmp/swe/sif-dev`) and runs the probe harness, sandboxes and grading, which
  keeps that CPU and RAM load off node08. Measured layout for probes: `--servers node08:8001 node08:8002 node04:8003
  --per-server 12 12 8` = ×1.78 rollout throughput over node08 alone at 8+8 (`experiments/ab-serving/`).
  Launch it exactly like this (without `SARL_SERVE_MORE` the Inductor compile tries to copy the 5.25 GiB table onto the
  8 GB card and start-up fails with a CUDA OOM, seen 2026-09-26):
  ```
  cd swe-bench && tmux new -d -s swe-serve4 "HF_HOME=/var/tmp/swe/hf-cache CUDA_HOME=/cm/shared/apps/cuda12.8/toolkit/12.8.1 \
    SARL_EAGER=0 SARL_PLE_OFFLOAD=1 SARL_SERVE_OOM_SCORE_ADJ=1000 SARL_SERVE_MORE='-cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY' \
    SARL_MAX_NUM_SEQS=8 SARL_MAX_BATCHED=2048 bash scripts/22_serve_gemma4.sh 0 8003 0.935 2>&1 | tee /var/tmp/swe/serve-gemma-node04.log"
  ```
  `OOM_SCORE_ADJ=1000` makes the server, not the sandboxes, the kernel's victim when node04's 14 GB run out; the engine
  then dies silently ("engine core exited unexpectedly", no traceback), as on 2026-09-26 04:56 during the SWE-smith
  screen. Probes fence the dead server and re-admit it once restarted.
- **node08 disk (`/var`, 40 GB; 15 GB free on 2026-09-26, check `df -h /var`):** `hf-cache` (bf16 Gemma, 15 GB, kept
  by user decision: the trainer reads it at every start), `sif-dev` (the 11 dev-pool images, 7.6 GB, so dev-pool runs
  such as `kd_minicoder_v1` need no rsync: `SARL_SIF_DIR=/var/tmp/swe/sif-dev`), `adapters` (0.9 GB; grpo-gemma-v1
  keeps only step-30). The SWE-smith images are on node04 only. grpo-gemma-v1 steps 1–29 are archived on node04
  `/var/tmp/swe/adapters-archive/grpo-gemma-v1`. New run configs set `keep_adapters: 2` so a run keeps only its newest
  two adapters (resume needs only the newest).

## Facts learned the hard way (2026-09-24)

- **Apptainer instances work unprivileged on node08.**
  - Start one with `instance start --fakeroot --containall --writable-tmpfs`. That subcommand has no `--pid` flag.
  - Inside you are uid 0 in `/testbed`; activate the env with `source /opt/miniconda3/bin/activate testbed`.
  - State persists across execs, and each exec takes about 0.7 s.
  - The image's HEAD is a commit named "SWE-bench", so patch = `git diff HEAD`.
- **Grading takes about 5 s per patch for the pool tasks.** Gold = 1 and empty = 0 on marshmallow-1359, with all 76 pass-to-pass tests green.
- **Qwen2.5-Coder-0.5B does not follow the diagram's JSON tool calls.** It produced 0 valid calls in 48 turns, even with an example in the prompt. With the reference's fenced-bash protocol (`protocol: backticks`) 92% of its actions were valid, but it mostly submits immediately.
  - This matches the lecture's 2:50 advice: warm up with SFT, or use a larger model, before RL on the tool-call protocol.
  - The oracle-in-group trick (one scripted gold-patch rollout per group) is what gives the 0.5B run non-degenerate groups.
- **Model choice after S9:**
  - Qwen2.5-Coder never emits `<tool_call>` tags at 0.5B, 1.5B or 7B-AWQ; every size writes the JSON inside a ```json markdown fence.
  - Qwen3-1.7B (non-thinking, `MODELS["qwen3-1.7b"]`) does: 97 % valid calls. But it resolved 0/140 in `s9-qwen3-1.7b-v1`, looping on `grep`.
  - The user wants a model that follows the protocol yet leaves room for RL to show a gain, so the target is a low but non-zero pool pass@8 (about 5–30 %), measured with `scripts/s9_probe.py --grade` before training.
- **Qwen3's chat template is not append-only.** It adds an empty `<think>\n\n</think>\n\n` to the last assistant turn. `ChatFormat` passes `enable_thinking=False` and cuts the delta at the reply's `<|im_end|>`; a test covers this.
- **vLLM 0.25.1 serves exactly what the harness needs.**
  - `/v1/completions` with token-ID prompts returns `token_ids`, including the `<|im_end|>` stop token, plus `token_logprobs`.
  - Runtime LoRA loading goes through `POST /v1/load_lora_adapter`. Adapter names must be unique per server lifetime, so names are run-tagged and leftovers are unloaded at run start.
- **Isolation defaults (after the adversarial review).**
  - Sandboxes and grading run with `--network none`, so they cannot reach the vLLM admin API, the cluster or the internet. pip and conda are set to fail fast offline (`PIP_NO_INDEX`, `CONDA_OFFLINE`).
  - Grading binds only a read-only input folder. The spec, log and report stay on the host, and patch sections touching the test harness are dropped.
  - Crashed or ungraded rollouts leave their group rather than counting as reward 0.
- **Stragglers.** The loop is synchronous, so one rollout that re-runs a slow test suite holds the whole step. A single rollout rerunning `pytest /testbed` 15 times cost 30 min in `s8-basic-v3b`. `basic_run.yaml` now uses 60 s commands and a 600 s episode clock.
- **Offline dataset loading:** `load_dataset(revision=…)` still calls the Hub, and huggingface_hub rejects partial snapshots offline. `sarl.tasks` therefore reads the pinned parquet files straight from the HF cache.

## SWE-smith as a second task source (2026-09-25)

- `sarl.smith` + `load_split("smith")`: the four dev-repo mirrors only (`SMITH_REPOS`, 4,662 tasks after dropping the
  `pr_` mirrors, which copy real dev issues byte for byte). MIT data and code; none of the 12 Verified repos has rows.
- A row's `patch` *introduces* the bug; `Task.patch` is its reverse (the gold), `Task.bug_patch` the original.
  `ApptainerEnv(setup=task.sandbox_setup)` applies the bug, deletes the FAIL_TO_PASS test files and leaves a
  one-commit repository (no remote, no reflog: the fix is not recoverable from git). The grader rebuilds the bug with
  the tests kept (`/sarl_in/setup.sh`) and runs a synthesized `pytest -rA` script over the F2P files, parsed by
  swebench's `parse_log_pytest`; PASS_TO_PASS is restricted to tests in those files, minus the offline-failing ones in
  `experiments/smith-p2p-exclusions.json` (pydicom's network-dependent tests).
- One image per repository (`swesmith.x86_64.<owner>_1776_<repo>.<sha8>.sif`, 1.3–1.6 GB); screens must run with
  `--keep-all` because `rolling_probe` deletes per task. Images live in node04 `/var/tmp/swe/sif-smith`.
- Pre-registered screen list `configs/screen_smith_v1.txt` (200 tasks, hashed 50 per repo); gate record
  `experiments/gate-smith-v1.json` (179 pass; pydicom re-gated after calibration).

## On-policy KD (`objective: kd`, 2026-09-26)

- **What:** a second training objective next to GRPO (`sarl.kd`, wired into `sarl.train`). The student generates
  and grades episodes exactly as in GRPO. A teacher server then scores every sampled token (vLLM `prompt_logprobs`).
  The per-token advantage `a_t = kd.coef·(log p_teacher − log p_sampler) + kd.reward_coef·A_group` replaces the group
  advantage in `LoRATrainer.step`, which accepts a float or a per-position sequence (a constant sequence gives the same
  update as the scalar). This is the sampled-token reverse-KL estimator of on-policy distillation, with discount 0.
  It is unbiased only at T = 1 with no top-p/top-k.
- **Config:** `objective: kd`, `teacher: {url, model: policy, model_key, sleep}` and
  `kd: {coef: 1.0, reward_coef: 0.0, workers: 4, timeout_s: 600}`. Validation requires a teacher and
  `oracle_per_group: 0`. GRPO configs are untouched (`objective` defaults to `grpo`).
- **Batch:** every valid policy rollout is trained, including degenerate groups (`A_group = 0` there). Groups are
  still computed and logged.
- **Teacher scoring:** the teacher scores each trajectory up to its last trained token, with `kd.workers` concurrent
  requests. A trajectory that still fails after one retry is dropped (`kd.dropped`). If every request fails, the run
  stops before its update and can be resumed.
- **Metrics:** `metrics.jsonl` gets a `kd` record per step and `seconds.teacher`. Read the `kd` record like this:
  - Every `rkl_*`, `mean_*_lp` and `msq_gap_*` value averages over the same `n_ok` tokens: trained tokens with both
    a sampler and a teacher log-prob (the rest have a_t = 0 and make up `frac_missing`).
  - `rkl_sampler` is the KD learning curve (fresh samples each step); it should fall across steps.
    `rkl_trainer_before − rkl_sampler` is the trainer-vs-sampler mismatch on those tokens.
  - `rkl_gain = rkl_trainer_before − rkl_trainer_after` > 0 means the update moved the student toward the teacher on
    this batch. The after-update value is importance-weighted (w = e^(lp_after − lp_before), capped at 10;
    `is_w_max` is the largest weight before the cap), because the tokens were sampled before the update. Without the
    weight the teacher term cancels and the "gain" would be positive for any update that lowers the sampled tokens'
    log-probs, even one that moves away from the teacher.
  - `objective_gain` (token level: Σ a_t·Δlp_t / T) > 0 means the update ascended its own surrogate;
    `msq_gap_before/after` is the squared per-token gap the update descends on these tokens.
- **Start-up checks:**
  1. Identical vocabulary and special-token ids between the student's tokenizer and `MODELS[teacher.model_key]`'s
     (offline, from the snapshot).
  2. The teacher serves `teacher.model` from that snapshot, and its `max_model_len` ≥
     max(limits.max_context_tokens, train.max_train_tokens + 1). vLLM needs prompt + 1 ≤ max_model_len.
  3. `preflight_servers` checks base-weight identity across the rollout servers only; its sleep check also covers the
     teacher when `teacher.sleep`. A sleeping teacher shares the trainer's GPU and naps with the sleep servers during
     the update (`sleepers_for`). A step with nothing to train leaves every server awake and the trainer on the CPU.
  4. After the trainer's CUDA context exists, one `train.max_train_tokens`-token teacher request (`teacher_smoke`)
     must return every log-prob. It is the first test of vLLM's chunked prompt-logprob path and of its logits memory,
     which lies outside the profiled budget. `servers.json` records its time and, for a sleeping teacher, the GPU
     memory still free after it.
- **Layout** (`configs/kd_minicoder_v1.yaml`, mini-coder-4b → mini-coder-1.7b, like-for-like with grpo-minicoder-v2):

  | where | serves | settings |
  |---|---|---|
  | node08 GPU0 | student :8001 | 0.88, 16 seqs |
  | node08 GPU1 | teacher :8002 | 0.84, 16 seqs, sleep mode, no LoRA; the trainer shares this GPU |
  | node04 | student :8003 | 0.92 (to verify), 8 seqs |

  Every server runs with `SARL_MAX_MODEL_LEN=25600`. At the default 32768, node04's 8 GB card cannot hold one
  full-length sequence next to the 1.7B weights, and `preflight_servers` requires every student server to use the
  same value. No request comes near it: the harness caps prompt + reply at 24576 tokens. Sandboxes run on node08 with
  `SARL_SIF_DIR=/var/tmp/swe/sif-dev`. vLLM builds the prompt-logprob logits in 1024-token chunks (0.3–0.6 GB) that
  its start-up profile never sees; at 0.84 they fit in the ~2.5 GiB left on GPU1. The config header has the exact
  commands (each tees a node-local log, `/var/tmp/swe/serve-kd-*.log`) and the memory arithmetic, taken from the vLLM
  start-up logs.

## Commands

```bash
make -C swe-bench help
make -C swe-bench test NODE=node08            # 80 tests (sandbox/grader/qwen markers skip cleanly elsewhere)
make -C swe-bench fetch | assets | sifs | grader | serve [MODEL=qwen3-1.7b] | train [CONFIG=…] | notebook | status
.venv/bin/sarl task <id> | pool | grade <id> --gold | episode <id> --policy vllm --grade | summarize runs/<run>
# training curves (run on node04, not the headnode): series JSON + SVG charts spliced into the explainer's section 12
python scripts/run_tracker.py runs/grpo-gemma-v1 --compare runs/grpo-minicoder-v2 \
    --out experiments/tracker-gemma-v1.json --splice docs/blog/swe-rl-explainer.html --stamp "refreshed <date>"
```

The explainer (`docs/blog/swe-rl-explainer.html`, Artifact "Teaching Gemma to Fix Real Bugs") keeps its charts between
`<!-- tracker:auto:start -->` and `<!-- tracker:auto:end -->`; everything else in section 12 is hand-written and names
the step it describes, so update the prose when the run moves on.
