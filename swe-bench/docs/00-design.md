# 00 — Design

How `sarl` implements the lecture's diagram (`reference/_upstream/swe_agent_env_interface.png`) on node08. The numbers come from the gate runs of 2026-09-24 (`experiments/EXPERIMENTS.md`); S8 numbers are from the final run `s8-basic-v3b`, made after the adversarial-review fixes (commit `0801504`), unless marked otherwise.

## 1. The diagram, mapped to code

| Diagram element | sarl | Notes |
|---|---|---|
| AGENT = harness + policy | `harness.run_episode(policy, env, task, fmt, limits, protocol=…)` | The policy is stateless (text/ids in, ids out). The harness is deterministic code and is never trained. |
| POLICY πθ, vLLM-generated, GPU | `policy.VLLMPolicy`; server `scripts/20_serve_policy.sh` | `/v1/completions` on token-id prompts. Returns `token_ids` (including `<\|im_end\|>`) and `token_logprobs`. Model `policy` on GPU0 :8001, with LoRA adapters selected by model name. |
| Harness (CPU) owns the transcript and the loop | `harness.run_episode`, `harness.ChatFormat`, `harness.Trajectory` | The transcript is kept as token ids with a mask and log-probs (§3). |
| ACTION = one bash tool call | `protocol.parse_reply(text, protocol)`, `protocol.BASH_TOOL` | Two protocols, `tool_call` and `backticks` (§2). |
| OBSERVATION = `<returncode>…</returncode><output>…</output>`, tool message, mask 0 | `protocol.format_observation`, `ChatFormat.tool_turn` / `user_turn` | Output over 10k characters is shown as head + tail with a warning (mini-swe-agent's rule). |
| Teal = policy tokens, mask 1 → gradient | `Trajectory.mask`, `Trajectory.logprobs` | Only sampled tokens have mask 1. Harness-inserted tokens have mask 0. |
| INNER LOOP exits: submit · 50 turns · 32k context · 30 min | `Limits` (defaults `max_turns=20`, `max_context_tokens=16384`, `wall_clock_s=1800`, `max_format_errors=3`; `basic_run.yaml` sets `wall_clock_s: 600`), `protocol.is_submit` | `exit_reason` ∈ `submitted`, `max_turns`, `context`, `wall_clock`, `format_errors`. Budgets are smaller than the diagram's (§7). |
| ENVIRONMENT = Docker container, `/testbed` | `env.apptainer.ApptainerEnv` (one apptainer instance of the task's SWE-bench image per rollout, `--network none`); `env.mock.MockEnv` in tests | There is no Docker on the cluster (§4). |
| `git diff > patch.txt`; `echo COMPLETE_TASK…` submits | `ApptainerEnv.patch()` collects the patch when the episode ends, diffed against the HEAD recorded at sandbox start; `protocol.SUBMIT` | The agent does not write `patch.txt`: the system prompt says changes are collected on submit. |
| REWARD, episode end only: checkout/clean, `git apply`, `run_tests.sh` → r ∈ [0,1] | `reward.grade.grade`, `reward/grader_main.py` (runs inside the swebench 5.0.2 venv) | Test-harness edits are filtered out, then a fresh offline container replays only the patch from a read-only input folder. The reward is binary (§5). |
| OUTER LOOP: trajectory + reward → grouped advantage (16 rollouts/task) | `train.run`, `grpo.group_advantages`, `grpo.is_degenerate` | G = 8 (7 policy + 1 scripted oracle) in `basic_run.yaml`. Crashed or ungraded rollouts are left out of their group. |
| GRPO update | `grpo.LoRATrainer.step` | Sequence-mean REINFORCE with a group baseline (the reference's loss). |
| πθ′ loaded back into vLLM | `train.LoraSync.load` → `POST /v1/load_lora_adapter`; `train.served_logprobs` | This is proven every step by `served_dlogp_vs_base` (§6). |
| (not in the diagram) oracle rollout | `policy.ScriptedPolicy` + `harness.oracle_replies` | This is the reference notebook's trick for keeping groups from being all-zero. Its data is off-policy. |
| (not in the diagram) tasks | `tasks.load_split`, `read_pool`, `select_pool`, `contamination_report` | Pinned parquet files are read straight from the HF cache. Training uses the dev split only. |

## 2. Action protocols

| | `tool_call` (the diagram) | `backticks` (the reference notebook) |
|---|---|---|
| Reply format | a short thought, then one `<tool_call>{"name": "bash", "arguments": {"command": …}}</tool_call>` (Qwen/Hermes form; the diagram's flat `{"name":"bash","command":…}` is also accepted) | one sentence of reasoning, then one fenced block: ` ```bash … ``` ` |
| Prompt | `SYSTEM_PROMPT` plus the tools block (`BASH_TOOL`) rendered by the chat template | `BACKTICKS_SYSTEM_PROMPT`, with no tools block |
| Parser | zero calls, more than one call, an unclosed tag, bad JSON, a non-object JSON value, an unknown tool or an empty command each become a format error | the first fenced bash block; if there is none, the reply gets the reference's `NO_COMMAND` message |
| Observation turn | a `tool` message (Qwen renders it as `<tool_response>` in a user turn) | a plain `user` turn, as in the reference |

A format error is fed back to the policy as a user turn (mask 0). Three in a row end the episode. A JSON value that decodes to a string or list is a format error, not a crash; the lecturer's `trial_10` parser bug killed a whole batch that way.

**Why `backticks` for the 0.5B model.** At S6, Qwen2.5-Coder-0.5B-Instruct produced 0 valid JSON tool calls in 48 turns over 16 real episodes, even with an example call in the system prompt. With the fenced-bash protocol, 45 of 49 turns (92 %) were valid, but 14 of 16 episodes submitted immediately, copying the submit block from the prompt. In the final S8 run, 1,186 of 1,282 policy turns (92.5 %) were valid actions, and 27 of 140 episodes submitted on turn 1 (first S8 run: 1,020 of 1,141, 89 %). The lecture says the same at 2:50: a model that does not know the action space needs SFT, or a larger model, before RL. `basic_run.yaml` therefore sets `protocol: backticks`. Moving back to the diagram's protocol is gate S9 (`02-phase2-roadmap.md`).

## 3. Token-exact harness

The trainer consumes the ids the policy actually sampled; it does not re-render text.
- Re-rendering a transcript with the chat template does not reproduce the sampled tokens. Templates re-serialise tool calls, merge consecutive same-role turns and drop `<think>`.
- The reference decodes with `skip_special_tokens=True`, calls `.strip()`, and re-tokenises for training. Its trained tokens are therefore not its sampled tokens.

```
ids:   [system(+tools) | user issue | asst reply₁ | obs₁ turn | asst reply₂ | obs₂ turn | …]
mask:  [       0       |     0      |   1 … 1    |   0 … 0   |   1 … 1    |   0 … 0   | …]
lp:    [     None      |    None    | vLLM lp …  |   None    | vLLM lp …  |   None    | …]
```

**Prompt.** The prompt is rendered once with `apply_chat_template(..., add_generation_prompt=True)` and tokenised without special tokens added.

**Replies.** Each reply is appended exactly as sampled, with mask 1 and vLLM's per-token log-probs. The reply normally ends with the sampled `<|im_end|>`. If generation stopped for length or on `<|endoftext|>`, the harness appends `<|im_end|>` itself, with mask 0 and log-prob None, because that token was never sampled. The scripted oracle follows the same rule: if its reply is cut at `max_new_tokens`, the executed text is decoded from the kept ids, so what runs is what is trained on.

**Untrusted text.** The issue and every observation pass through `ChatFormat.sanitize` before tokenisation. A literal special token such as `<|im_end|>` printed by a command would otherwise tokenise as the real turn delimiter and forge a turn boundary. A word joiner (U+2060) after `<|` keeps the text readable but plain.

**Turn deltas (`ChatFormat`).** The harness needs the exact token deltas for new tool and user turns. It derives them from the tokenizer's own template by rendering a probe conversation with and without one extra turn:
- It asserts that the template is append-only for that turn type.
- It cuts right after the previous assistant's `<|im_end|>`, which the policy sampled itself.
- It splits the new turn's text at the probe string into a prefix and a suffix.

Each observation is then encoded as `prefix + observation + suffix` and appended with mask 0.

**Test.** `tests/test_swe_harness.py::test_scripted_episode_is_token_exact` runs for both protocols. It checks that the incrementally built ids equal a one-shot tokenisation of the rendered conversation, and that mask 1 covers exactly the assistant replies. At S6, the same property held on real vLLM rollouts: the incremental ids matched the one-shot render tokenisation.

**Context budget.** The budget is checked before each generation (`max_context_tokens − len(ids) ≥ min_reply_tokens`). One long observation can therefore push a trajectory past the budget. The longest policy trajectory had 18,473 tokens in the first S8 run and 14,657 in the final one.

## 4. Environment: apptainer instances of the SWE-bench images

There is no Docker or podman on the cluster. Apptainer 1.5.2 runs unprivileged from `~/apptainer-env`.

| Aspect | Choice | Why |
|---|---|---|
| Image | `apptainer pull docker://swebench/sweb.eval.x86_64.<id with __→_1776_>:latest` into `/tmp/swe-sif/<id>.sif` (`env.apptainer.ensure_sif`: locked, atomic rename) | These are the official SWE-bench eval images, about 1 GB each. The caches (`/var/tmp/swe/apptainer-{cache,tmp}`) stay node-local. |
| Start | `apptainer instance start --fakeroot --containall --writable-tmpfs --net --network none <sif> sarl-<run tag>-<name>-<8hex>`, then `git rev-parse HEAD` is recorded as the diff base | Inside, the agent is uid 0 in `/testbed`, and the conda env `testbed` is activated per exec. `instance start` has no `--pid` flag; instances always get their own PID namespace. |
| Why an instance | A per-command `apptainer exec` would start a new container and lose the tmpfs overlay | In an instance, file edits persist across execs for the instance's lifetime, and each exec takes about 0.7 s. The overlay lives in RAM, so no per-rollout disk is needed. mini-swe-agent's singularity backend unpacks a sandbox directory per run, which would not fit in the 16 GB `/tmp`. |
| One command | `timeout -k 5 T bash -c CMD > /tmp/.sarl_out 2>&1 < /dev/null; rc=$?; cat /tmp/.sarl_out; exit $rc`, with a host timeout of T + 60 s. T = `command_timeout_s`, 60 s in `basic_run.yaml` (120 s in the S8 runs). | The output is captured to a file inside the container rather than a host pipe. A backgrounded child that holds stdout open would otherwise stop the pipe from reaching EOF and hang the call; Harbor needed a patch for this in terminal-bench. At S4 the timeout killed `sleep 600`. Return codes 124 and 137 are reported as `timeout`. |
| Output cap | 200k characters in the env (head + tail); 10k in the observation (head + tail + warning) | This keeps huge `find` or `cat` output out of the context. |
| Patch | `git -c core.fileMode=false add -A && git -c core.fileMode=false diff --cached --binary <HEAD recorded at start>`, read through its own exec: stdout only, never truncated; a git failure raises | The image's HEAD is a commit named "SWE-bench" on top of `base_commit`. This diff includes new files and binary changes and ignores mode changes. Pinning the base means an agent that commits cannot hide its changes, and a lost patch is an error rather than a silent empty patch (reward 0). |
| Locale | `APPTAINERENV_LANG=C.UTF-8`, `APPTAINERENV_LC_ALL=C.UTF-8` | `--containall` implies `--cleanenv`, which leaves `LANG=C`. Python 3.6 testbeds then use an ASCII filesystem encoding, and pydicom-811's pass-to-pass test `testUTF8FileName` failed even with the gold patch. `APPTAINERENV_*` variables survive `--cleanenv`. Found by `scripts/pool_gate.py`, after the first S8 run (see EXPERIMENTS). |
| Network | `--net --network none` for rollouts and grading (`env.network: false`, `grading.network: false`, the defaults). Offline, `APPTAINERENV_PIP_NO_INDEX=1`, `APPTAINERENV_PIP_DISABLE_PIP_VERSION_CHECK=1` and `APPTAINERENV_CONDA_OFFLINE=true` make pip and conda fail fast. | A sandbox on the host network could reach the vLLM LoRA admin API on localhost :8001, the cluster, and the internet (including upstream fix commits). `--network none` leaves a private loopback only and works unprivileged. Without the fast-fail variables, a policy looping on `pip install` waited through minutes of retries per command and stalled a step. |
| Cleanup | Instance names carry the run prefix `sarl-<run tag>-`; `stop_all(prefix)` runs at start and at exit (`atexit`); an instance that fails to start is stopped | A crashed trainer or a half-started instance does not leak sandboxes into the next run. |
| Guards | `refuse_on_headnode`; `paths.refuse_on_home` for the SIF, cache, grading, adapter and trajectory directories | Standing rule: no compute on the headnode, and no large files on the NFS `/home`. |

## 5. Reward

`reward.grade.grade(task, patch, sif=…)`, once per rollout, after the episode:

1. **Harness filter.** `filter_patch` drops every file section that touches the test harness (`conftest.py`, `pytest.ini`, `tox.ini`, `setup.cfg`, `noxfile.py`, test files and `test/`/`tests/` directories) and reports the paths as `filtered_paths`. `eval.sh` re-checks-out the task's own test files anyway. In the final S8 run, 2 policy patches were filtered: one touched only `tests/test_error.py` and became empty; one lost its `setup.cfg` and `tox.ini` sections.
2. **Empty patch.** It scores 0 without running anything (`run_empty=True` forces a real run). The pool gate uses that to prove empty = 0 with the tests actually run: every pool task gives F2P 0/1 with all P2P green, e.g. astroid-946 P2P 155/155.
3. **Workdir.** A node-local workdir `/var/tmp/swe/grade/<id>-<8hex>/` has two folders:
   - `in/` holds `patch.diff` and `eval.sh`. It is the only thing bound into the container, read-only.
   - `host/` holds `instance.json` (the v5 fields `make_test_spec` needs), `test_output.txt` and `report.json`. The container never sees it.
4. **Eval script.** `grader_main.py spec` runs inside the grader venv (`/var/tmp/swe/grader-venv`, `swebench==5.0.2`) and writes `make_test_spec(row).eval_script` to `in/eval.sh`.
5. **Fresh container.** A fresh, non-instance container is started from the same SIF (`apptainer exec --fakeroot --containall --writable-tmpfs --net --network none -B <wd>/in:/sarl_in:ro`). In it:
   - The apply chain tries `git apply --verbose`, then `git apply --verbose --reject`, then `patch --batch --fuzz=5 -p1`. Between attempts it runs `git checkout -- . && git clean -fd`. If all three fail, it prints `>>>>> Patch Apply Failed`.
   - Then `bash eval.sh` applies the held-out `test_patch` and runs the tests.
   - The log is the container's stdout, captured by the host into `host/test_output.txt` and capped at 2 MB (head + tail). A wall-clock timeout appends `>>>>> Tests Timed Out`.
6. **Report.** `grader_main.py report` runs `get_eval_report(spec, pred, log, include_tests_status=True)` on the host.
7. **Result.** `resolved` means every FAIL_TO_PASS test passes and every PASS_TO_PASS test still passes. Reward = 1.0 if resolved, otherwise 0.0. `GradeResult` also records `applied`, F2P/P2P counts, `timed_out`, `infra_failure`, `filtered_paths`, seconds and a 3,000-character log tail. The workdir is deleted afterwards (`keep=True` keeps it).

**Why the split workdir.** Before the review fix, the whole workdir was bound writable. Code under test is policy-written and runs as fakeroot uid 0, so it could have rewritten `instance.json` (the FAIL_TO_PASS/PASS_TO_PASS lists) or the log and forged reward 1.

**Fresh container, not the rollout's instance.** This is stronger than the diagram's "checkout + clean": it also discards anything the agent changed outside the repo (installed packages, `/tmp`, the conda env). The tree is wiped, and only the patch is replayed.

**Separate venv.** swebench's dependency set would re-resolve the shared venv's CUDA stack, so `grader_main.py` must not import `sarl`. The cost is two subprocess calls per grade.

**Cost.** In the first pool gate, gold and empty patches graded in about 5–13 s each. In the final S8 run, the grading phase took 4.3–10.5 s per step, 8 grades run concurrently.

## 6. Training loop (`train.run`)

Per step, with `basic_run.yaml`:

1. **Tasks.** Sample `tasks_per_step = 2` tasks from the pool (`random.Random(seed)`). `torch.manual_seed(seed)` fixes the LoRA init.
2. **Rollouts.** Run `group_size = 8` rollouts per task, 16 at a time (`concurrency.rollouts`), each in its own `ApptainerEnv`.
   - Policy rollouts sample from the adapter currently served (`sync.current`), with plain sampling and a per-request vLLM seed (`seed · 1,000,003 + step · 1,009 + idx`).
   - The last `oracle_per_group = 1` rollout per group is `ScriptedPolicy(oracle_replies(task, protocol))`. It writes the gold patch with a heredoc, runs `git apply`, then submits, all in the same action format the policy is asked to use.
   - The step waits for its slowest rollout (the loop is synchronous). In the final S8 run, one episode held step 1 for 1,806 s while the median episode took 2.3 s.
3. **Grading.** Every candidate patch is graded (8 concurrent).
4. **Advantages.** A group keeps only valid members: a crashed rollout, a failed grade or an infra failure is left out rather than counted as reward 0, and a group with fewer than 2 valid members is skipped (logged as `invalid` and counted as degenerate). Per group, `A_i = (r_i − mean r) / (std r + 1e-4)`, with the biased std (the reference's formula; `tests/test_swe_grpo.py` checks it against the upstream code). With 7 zeros and one 1, the oracle gets A = +2.645 and each policy rollout gets −0.378.
   - **Degenerate groups** (all rewards equal) are skipped (`advantage.skip_degenerate: true`) and counted. The lecture throws them out (2:00); the reference code keeps them as a no-op (A = 0). With the 1/N loss normalisation, skipping also changes the step size: N counts informative trajectories only. The first S8 run had 2 (the LANG bug); the two later runs had 0.
   - **Over-long trajectories.** Trajectories longer than `max_train_tokens = 8192` are dropped from the update *after* the advantages are computed, so their reward still sets the group baseline (DAPO-style overlong filtering). They are counted per step as `dropped_overlong` (4 in the final S8 run).
5. **Diagnostics before the update.** The trainer's log-probs of the sampled tokens are compared with vLLM's (`mismatch_stats`), pooled over every sampled token of the step's policy rollouts (up to `max_train_tokens`). The first two S8 runs logged the mean of per-rollout statistics instead, which understates the max and p99. The mean log-prob of every batch trajectory is also recorded.
6. **Update.** `LoRATrainer.step` computes `L = −(1/N) Σ_i A_i · mean_{t: mask=1} log πθ(a_t | s_t)` and backpropagates it per trajectory.
   - It then clips the grad norm to 1.0 and takes one AdamW step (lr 1e-5). The optimizer persists across steps. The step is skipped if the loss or grad norm is not finite.
   - LoRA: r = 8, α = 16, dropout 0 on `q,k,v,o` (1,081,344 trainable parameters). The base is bf16 with gradient checkpointing; the adapter is kept in fp32 (lecture 1:47: fp16 Adam second moments underflow).
7. **Chunked log-softmax.** `token_logprobs` runs the transformer body once and gathers the hidden states only at supervised positions (mask 1, shifted by one). It then applies `lm_head` and `log_softmax` in chunks of 1,024 positions, each chunk checkpointed. Full logits for a 16k-token trajectory would be 16k × 152k floats. `test_trainer_logprobs_match_full_softmax` checks this against the unchunked computation.
8. **Objective check.** After the update, `J = (1/N) Σ A_i · mean log π` is recomputed on the same batch, and `objective_gain = J_after − J_before` is logged. It is not guaranteed to be positive on every step: the AdamW update is preconditioned and carries momentum from earlier batches, so one step at lr 1e-5 need not ascend this batch's J, and oracle and policy trajectories share tokens (e.g. the submit block) with advantages of opposite sign. Final S8 run: positive on 7/10 steps, mean +5.5e-04, the three negatives ≥ −6.7e-04. `oracle_dlogp` and `policy_dlogp_mean` are logged separately.
9. **Weight sync.**
   - `save_pretrained` writes the adapter to `/var/tmp/swe/adapters/<run>/step-N`.
   - `POST /v1/load_lora_adapter {"lora_name": "policy-<run_tag>-s<N>", "lora_path": …}`, then unload the previous adapter. The server runs `--max-loras 2` and `--max-lora-rank 16`, with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`.
   - The next step samples from the new name.
   - **Unique names.** Adapter names must be unique for the server's lifetime. The name carries a run tag (the first 6 hex digits of `sha256(run_dir.name)`, also used as the sandbox prefix), and `LoraSync.unload_leftovers()` removes older runs' adapters at start. Before this fix, a second run crashed with `400 … 'policy-s1' has already been loaded`.
10. **Proof that vLLM serves πθ′.** `served_dlogp_vs_base` is the mean |log p| difference over a fixed probe sequence (a user prompt plus ` ```bash\nls\n```<|im_end|> `), measured by vLLM's `prompt_logprobs` under the new adapter and under the base model. A greedy text probe (`probe_changed`) is also logged. It never changed in S7 or any S8 run, because the updates are too small to flip a greedy decode, so the log-prob metric is the evidence.
11. **Records.**
    - `metrics.jsonl` gets one record per step.
    - `rollouts.jsonl` gets one record per rollout, without tokens.
    - `/var/tmp/swe/trajectories/<run>/step-NNN/<task>__<kind><i>.json.gz` on node08 holds the full ids, mask, log-probs and messages, gzip'd and node-local (NFS is about 92 % full). `summary.json` records the trajectory and adapter paths.
    - `log.txt` is the human-readable log.

**What the metrics mean.**

| Metric | Definition | Final S8 run (`s8-basic-v3b`, 10 steps) |
|---|---|---|
| `objective_gain` | J_after − J_before on the update batch | −6.7e-04 to 2.0e-03, positive on 7/10 steps, mean +5.5e-04 |
| `served_dlogp_vs_base` | mean \|log p_served(adapter) − log p_served(base)\| on the probe sequence | 0.0196–0.0276 |
| `mismatch.n` / `mean` / `p50` / `p99` / `max` / `frac_gt_0.5` | \|log p_trainer − log p_vLLM\| on sampled policy tokens (lecture 2:31), pooled over all of a step's tokens | n 1,525–6,546 tokens; mean 0.0018–0.0077; p99 0.058–0.114; max 0.401; 0 % > 0.5 |
| `oracle_dlogp` | per-oracle-trajectory change in mean log-prob | positive for 12 of 20 |
| `policy_valid_action_rate` | parsed actions / turns (mean over policy rollouts) | 74–99 % per step |
| `dropped_overlong` | trajectories over `max_train_tokens`, dropped after advantages | 4 in 10 steps |

## 7. Hyper-parameters (`configs/basic_run.yaml`)

| Group | Value | Diagram / reference |
|---|---|---|
| Model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` @ `ea3f2471`, served as `policy` at `http://localhost:8001` | reference: same model, unpinned |
| Pool | `configs/pool_dev.txt`, split `dev` (8 tasks) | reference: Verified astropy-12907 |
| Schedule | 10 steps × 2 tasks × G = 8; `oracle_per_group: 1`; `seed: 0` (torch seed and per-request vLLM seeds) | diagram: 16 per task; notebook: G = 8 (7 + oracle) |
| Protocol | `backticks` | diagram: `tool_call` |
| Sampling | plain: temperature 1.0, top_p 1.0, top_k −1, repetition_penalty 1.0. `policy.REFERENCE_SAMPLING` keeps the reference's settings (used by the first two S8 runs). | reference: T 1.0, top_p 0.95, plus Qwen's generation_config (top_k 20, rep. 1.05) applied implicitly by HF `generate` |
| Limits | 20 turns; 16,384 context tokens; 1,024 new tokens per reply; 600 s wall clock; 60 s per command; 10,000 observation characters; 3 consecutive format errors. All three S8 runs used 1,800 s and 120 s. | diagram: 50 turns, 32k, 30 min. Reference: 4 turns, 200 new tokens, 800-character observations |
| Train | lr 1e-5; LoRA r 8, α 16 on q,k,v,o; grad clip 1.0; `max_train_tokens` 8192; chunk 1024 | reference: same LoRA, lr and clip; `build_masked` truncates at 3,072 tokens |
| Advantage | eps 1e-4; `skip_degenerate: true` | reference: eps 1e-4, degenerate kept |
| Concurrency | 16 rollouts, 8 grades | — |
| Grading / env | timeout 1,800 s; network off (`--network none`) for both | — |

Serve flags (`scripts/20_serve_policy.sh`): `--max-model-len 32768 --gpu-memory-utilization 0.85 --enable-lora --max-lora-rank 16 --max-loras 2 --generation-config vllm --enable-prefix-caching`, plus `VLLM_USE_FLASHINFER_SAMPLER=0` (node08 has no nvcc) and `HF_HUB_OFFLINE=1`. The model path is the pinned local snapshot.

## 8. node08 layout

| Resource | Use |
|---|---|
| GPU0 (A4000 16 GB) | vLLM policy server in tmux `swe-serve`, port 8001. Measured at startup: weights 0.96 GiB, CUDA graphs 1.03 GiB, KV cache 11.99 GiB = 1,047,440 tokens (31.97× concurrency at 32,768 tokens). |
| GPU1 (A4000 16 GB) | trainer in tmux `swe-train` (`SARL_TRAIN_GPU`, default 1); also the phase-1 reference run (`SARL_PHASE1_GPU`, default 1) |
| `/tmp/swe-sif` (16 GB `/tmp` partition) | one SIF per pool task, about 1 GB each (8 for the pool) |
| `/var/tmp/swe` (39 GB `/var/tmp` partition, shared with terminal-bench's `/var/tmp/tarl`) | `apptainer-cache/`, `apptainer-tmp/`, `grader-venv/`, `uv-cache/`, `grade/<id>-<8hex>/{in,host}/` (one per grade, deleted after grading), `adapters/<run>/step-N/`, `trajectories/<run>/step-NNN/*.json.gz` |
| NFS `~/hf-cache` | pinned model (954 MB) and the three dataset snapshots (about 11 MB of parquet) |
| NFS `swe-bench/runs/` (gitignored) | run directories: config, metrics, rollouts, log and summary (856 KB for `s8-basic-v3b`). Runs before the review fixes (gate-s7 and the first two S8 runs) also kept their per-token trajectories here (8–10 MB per S8 run). |
| `swe-bench/experiments/runs/<run>/` | copies of the run records (config, metrics, log, rollouts, summary) for gate-s7, the three S8 runs and phase1-reference |

The two subprojects share node08. terminal-bench's driver uses GPU0 (vLLM :8000) and GPU1 (trainer), so the swe-bench serve and train sessions cannot run at the same time as it. The projects take turns; terminal-bench's A0/B1/B2 driver is next.

## 9. Decisions and trade-offs

| Decision | Alternative | Why | Cost |
|---|---|---|---|
| Apptainer instance per rollout, `--writable-tmpfs` | Docker (unavailable); an exec per command; a sandbox directory per run | Unprivileged, state persists, about 0.7 s per exec, no per-rollout disk | RAM-backed overlay; relies on `--fakeroot` working on the node |
| Output captured to a file inside the container | Host pipes | No hang when a command backgrounds a child | No streaming output; output arrives only after the command ends |
| Fresh container for grading | `git checkout/clean` inside the rollout container | Side effects outside the repo cannot leak into the reward | One extra container start per grade |
| Grading binds only a read-only input folder; spec, log and report stay on the host | Bind the whole workdir writable (the first version) | Code under test runs as fakeroot uid 0 and could otherwise rewrite the F2P/P2P lists or the log and forge reward 1 | None measurable |
| Drop patch sections that touch the test harness (`filter_patch`) | Grade the patch as written | A policy could edit `conftest.py`, `pytest.ini`, `tox.ini`, `setup.cfg` or tests to make tests pass or be skipped | A real fix that must touch those files loses that part; dropped paths are reported per grade |
| `--network none` for rollouts and grading; pip and conda fail fast | Host network (the first version) | The sandbox cannot reach the vLLM LoRA admin API (:8001), the cluster, or the internet (upstream fix commits) | The agent cannot install packages; the pool gate had to re-verify gold = 1 / empty = 0 offline (8/8) |
| swebench 5.0.2 in a node-local venv | Installing into the shared venv | The CUDA stack stays pinned; official log parsers and eval scripts | Two subprocesses per grade; the venv must be rebuilt after a node reprovision |
| Token-id transcript | Re-render text for training (the reference) | The tokens trained are the tokens sampled, and mask and log-probs line up | Needs chat-template deltas; `ChatFormat` asserts the template is append-only |
| `backticks` protocol for the 0.5B model | the diagram's JSON `tool_call` | 0/48 valid JSON calls versus 92 % valid fenced blocks | Deviates from the diagram until S9 |
| One scripted gold-patch oracle per group | A pure policy group | Without it every group is degenerate (policy reward was 0/140 in every S8 run) | Off-policy data with no importance correction. The update amounts to imitation of the oracle plus pushing down the policy's own samples. |
| Skip degenerate groups | Keep them as a no-op (the reference) | Matches the lecture (2:00) and saves backward passes | Changes the 1/N scale between steps |
| Leave crashed rollouts, failed grades and infra failures out of their group; skip groups with < 2 valid members | Count them as reward 0 (the first version) | An infrastructure failure is not evidence about the policy, and a 0 would feed a wrong baseline | Groups can be smaller than G; none were in the final S8 run (0 errors) |
| Drop over-long trajectories after the advantages (DAPO-style overlong filtering) | Drop them before the advantages, or truncate them (the reference) | Their reward still informs the group baseline, and no trajectory is trained cut off mid-episode | The trained members' advantages no longer sum to zero |
| Sequence-mean REINFORCE, one step per batch | Token-level clipped surrogate with ratios | Parity with the reference; simplest correct on-policy loss | Each trajectory gets equal weight regardless of length; no ratio or clip, so no safe reuse of stale or off-policy data |
| LoRA r=8 synced through vLLM runtime loading | Full fine-tune with an NCCL weight broadcast | Small adapter, loaded in seconds, no server restart, trainer and server on separate GPUs | 1.08M trainable parameters; one adapter write and load per step |
| 20 turns, 16k context, 8,192 training tokens, G = 8 | The diagram's 50 turns, 32k, G = 16 | Fits one 16 GB trainer GPU and keeps most steps at 18–155 s across the S8 runs | First S8 run: 8 of 140 policy episodes hit the context limit and 8 trajectories were dropped as longer than 8,192 tokens. Final run: 0 context exits, 4 dropped. |
| Synchronous loop with a 600 s episode wall clock and 60 s per command | The diagram's 30 min; async rollouts | Simple, strictly on-policy; the limits bound how long one straggler can hold a step | In the final S8 run (30 min, 120 s), one rollout ran `pytest /testbed` 15 times and step 1 took 1,818 s against 18–46 s for the others |
| Train on SWE-bench dev only | Train on Verified (the lecture's illustration) | Dev and Verified share 0 repos, ids and base commits, so Verified stays a clean eval set | Dev has only 225 tasks from 6 repos |
| Read pinned parquet files from the HF cache | `load_dataset(revision=…)` | Offline `load_dataset` still calls the Hub, and huggingface_hub rejects partial snapshots offline | The code depends on the HF cache's directory layout |
| Plain sampling (T 1, no top-p, top-k or repetition penalty) with raw vLLM log-probs | The reference's top_p 0.95, top_k 20, rep. 1.05 (`REFERENCE_SAMPLING`, used by the first two S8 runs) | The loss treats rollouts as samples of π, and the mismatch metric compares raw log-probs, so any sampling distortion would be invisible to it | More low-probability tokens reach the transcript. Valid-action rate per step was 0.74–0.99 in the final run against 0.83–1.0 with reference sampling (`s8-basic-v2`); the runs also differ in network setting, so this is not a controlled comparison. |
