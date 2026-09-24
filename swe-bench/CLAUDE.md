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
- **vLLM 0.25.1 serves exactly what the harness needs.**
  - `/v1/completions` with token-ID prompts returns `token_ids`, including the `<|im_end|>` stop token, plus `token_logprobs`.
  - Runtime LoRA loading goes through `POST /v1/load_lora_adapter`. Adapter names must be unique per server lifetime, so names are run-tagged and leftovers are unloaded at run start.
- **Offline dataset loading:** `load_dataset(revision=…)` still calls the Hub, and huggingface_hub rejects partial snapshots offline. `sarl.tasks` therefore reads the pinned parquet files straight from the HF cache.

## Commands

```bash
make -C swe-bench help
make -C swe-bench test NODE=node08            # 23 tests (sandbox/grader/qwen markers skip cleanly elsewhere)
make -C swe-bench fetch | assets | sifs | grader | serve | train | notebook | status
.venv/bin/sarl task <id> | pool | grade <id> --gold | episode <id> --policy vllm --grade | summarize runs/<run>
```
