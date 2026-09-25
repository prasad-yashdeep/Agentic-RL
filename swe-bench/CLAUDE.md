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

## Commands

```bash
make -C swe-bench help
make -C swe-bench test NODE=node08            # 27 tests (sandbox/grader/qwen markers skip cleanly elsewhere)
make -C swe-bench fetch | assets | sifs | grader | serve [MODEL=qwen3-1.7b] | train [CONFIG=…] | notebook | status
.venv/bin/sarl task <id> | pool | grade <id> --gold | episode <id> --policy vllm --grade | summarize runs/<run>
```
