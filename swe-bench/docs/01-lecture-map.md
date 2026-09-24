# 01 — Lecture map

Where each part of the Vizuara SWE-RL session and its reference code lands in `sarl`, and where `sarl` deliberately differs.

**Lecture:** "Build an RL agent for Software Engineering Tasks from Scratch | SWE-RL", Vizuara, YouTube `oNHzs7wi7TY`.
- Published 2026-08-02, runtime 2:58:43, no chapter markers.
- Speaker: Abhishek Goswami, Microsoft AI Frontiers Lab.

**Code:** `abgoswam/swe_in_prod_vizuara_01` at `8cddfb60c93cc6b2c7d9dc047939084858f30f76` (2026-07-31). Files: `utils.py`, `swe_grpo_one_step.py`, a 44-cell notebook (22 code cells), and `swe_agent_env_interface.png`. See `reference/UPSTREAM.md`.

## 1. Timestamp map

The timestamps come from YouTube's **auto-generated (ASR) English transcript**, fetched through the innertube `player` API with the iOS client (3,448 segments, about 23.1k words). They were checked by reading the transcript around each entry.
- ASR spellings to expect: "sui/sweep bench" = SWE-bench, "quen" = Qwen, "world" is probably verl.
- Two corrections to earlier notes: LoRA comes in around 1:47, not 1:53; model rollouts start around 1:17.

| Time | Content | In sarl |
|---|---|---|
| 0:06–0:08 | Agenda: the notebook (60–75 min), then production training dynamics, reward curves and debugging | — |
| 0:09–0:14 | `temp.py` Fibonacci bug; Claude Code proposes a diff | — |
| 0:14–0:15 | "technically we should not use the eval data set for training but for illustrative purposes that is what we will do" | Train on the dev split only. The phase-1 replica on Verified astropy-12907 is tagged `contaminates_eval` (`tasks.py`, `phase1_summary.py`). |
| 0:18–0:25 | SWE-bench Verified fields via astropy-12907 (`separable.py`, `= 1` should be `= right`); FAIL_TO_PASS 2, PASS_TO_PASS 13 | `tasks.Task`, `sarl task <id> --full`. `test_reference_task_fields` checks 2 / 13. |
| 0:30–0:37 | The diagram: agent = harness + policy; Docker container = environment; action = tool call; observation = return code + output | `docs/00-design.md` §1 |
| 0:34–0:35 | Exits: max turns, submit sentinel, context limit (32k/128k), wall clock (about 30 min) | `harness.Limits`, `protocol.is_submit` |
| 0:35–0:36 | Replay the patch in a clean container, run the tests; reward 0/1; an LLM judge is the alternative | `reward.grade` (fresh container, binary) |
| 0:37–0:39 | On-policy outer loop: rollouts with frozen weights, one GRPO step, push to vLLM. Async RL is described as more advanced. | `train.run` (synchronous), `LoraSync`. Async is in the roadmap. |
| 0:39–0:46 | Inference uses only the inner loop; reward and GRPO exist only in training | `sarl episode` (no update) vs `sarl train` |
| 0:47–0:48 | Each of the 16 rollouts gets its own Docker container; git worktrees as an alternative | one `ApptainerEnv` instance per rollout |
| 0:49–0:50 | GRPO over PPO (no value network); DPO excluded | `grpo.py` |
| 0:51–0:53 | Execution rewards beat LLM judges for code | test-based reward |
| 0:53–0:55 | Partial credit exists in practice; with binary rewards all-zero groups waste compute; this session uses 0/1 only | binary now; partial credit is in the roadmap |
| 0:58–1:05 | State = the files at `base_commit`; ls/grep leave it unchanged, `cat >` changes it; patch = final state − initial state | `MockEnv.patch`, `ApptainerEnv.patch` (`git diff --cached --binary` against the HEAD pinned at sandbox start) |
| 1:05–1:12 | Bash as the only tool, with a command allowlist; mini-swe-agent cited; the codebase is not loaded into the context | one bash tool; **no allowlist** (difference 11 in §3) |
| 1:12 | Test files are kept out of the agent's container; only the verifier sees them | `test_patch` is applied only by `eval.sh` in the grading container |
| 1:14–1:16 | The model is stateless; the harness is deterministic code; harness work alone is worth 10–30 points | `policy.*` is stateless; `harness.py` is never trained |
| 1:17–1:26 | Qwen2.5-Coder-0.5B rollout loops on blocked `setup.py install` / `pip install`; empty patch, reward 0 | Seen again in S1 (`python setup.py install`), and in S8: 12 of 140 policy episodes started with `pip` in the first run, 6 of 140 in the final run (`s8-basic-v3b`), where one episode repeated `pip install …` + `pytest /testbed` until the 30-min wall clock |
| 1:26–1:30 | Oracle rollout: cat, heredoc fix, pytest, ls | `harness.oracle_replies` (heredoc of the gold patch, `git apply`, submit) |
| 1:31–1:39 | Loss only on assistant tokens; tool responses masked | `Trajectory.mask` |
| 1:40–1:41 | `reward_random`, because a real verifier is hard to run in a notebook | replaced by real tests |
| 1:41–1:45 | G = 8 (7 model rollouts + 1 oracle); a fresh environment per `run_agent` | `group_size: 8`, `oracle_per_group: 1` |
| 1:46–1:51 | Advantages; LoRA used "purely because I was hitting some issue … FP16", with the adapter kept in fp32 | `grpo.group_advantages`, `LoRATrainer` |
| 1:51–1:55 | `build_masked` (labels −100), off-by-one shift, log_softmax, masked mean log-prob | `LoRATrainer.token_logprobs` (chunked, supervised positions only) |
| 1:56–1:59 | Plain REINFORCE with a group baseline, one AdamW step, log-probs before and after; then "push to vLLM" | `LoRATrainer.step`, `objective_gain`, `oracle_dlogp`, `LoraSync.load` |
| 2:00–2:02 | The oracle exists so a group is never all zero; all-0 and all-1 groups are "degenerate" and must be thrown out | `skip_degenerate: true`, `n_degenerate_groups` |
| 2:02 | A phase two is planned | `docs/02-phase2-roadmap.md` |
| 2:25–2:30 | Reward curves, Qwen3-14B vs Qwen3.5-9B; a crash at step 139 and restart; track train and validation reward | `metrics.jsonl`; validation is S10 |
| 2:31–2:34 | \|logprob(vLLM) − logprob(trainer)\| plot; "Diagnosing Training Inference Mismatch in LLM RL" | `grpo.mismatch_stats`, logged every step |
| 2:35–2:42 | A real trace (numpy structured dtype): one bash tool, `<think>` kept, huge `find` outputs, the verifier replays the patch; a wrong patch scored 0.05 | 10k-character observation cap; the harness keeps sampled tokens as-is |
| 2:47–2:49 | Partial credit after timeouts: did the agent touch the right files or regions? | roadmap item 7 |
| 2:49 | "For this specific project we are using a fork of verl"; also prime-rl and SkyRL | roadmap (shared core) |
| 2:50–2:51 | Do SFT first if the base model does not know the action space | the S6 protocol finding; roadmap S9 |
| 2:54 | Goswami is in the ECHO acknowledgements (MSR) | — |

## 2. Reference notebook and script → sarl

| Notebook section (cells) | What it does | sarl |
|---|---|---|
| Title, setup (0–1) | Phase-1 and Phase-2 goals; `!pip install` | The cluster copy drops only the two `!pip` lines and sets the kernel to `sarl` (`reference/make_cluster_nb.py`). |
| §1 Load one instance (2–12) | Verified astropy-12907: `problem_statement`, `test_patch`, FAIL_TO_PASS, PASS_TO_PASS | `tasks.load_split("verified_princeton")`, `Task` (`patch` and `test_patch` hidden from the agent), `sarl task` |
| The whole picture (13) | The diagram; "a Python dict, a fixed 4 turns, and 6 rollouts" | `docs/00-design.md` |
| §2 The environment (14–20) | State = files; read-only vs writing actions; new state − initial state = patch | `env.mock.MockEnv` (same semantics; `test_matches_reference_semantics`). The real version is `env.apptainer.ApptainerEnv`. |
| §3.1 The model (22–23) | Load Qwen2.5-Coder-0.5B-Instruct in fp16 | `policy.VLLMPolicy` (serving), `policy.HFPolicy` (in-process), `LoRATrainer` (bf16) |
| §3.2 The harness (24–26) | `run_agent`: prepare → call → parse → execute → append, 4 turns | `harness.run_episode`. `protocol.parse_reply(…, "backticks")` = `first_bash_block`, and `NO_COMMAND` is kept verbatim. |
| §3.3.1 Rollout with the model (27–30) | `show_rollout` (who wrote each turn) | `Trajectory.turns`, `rollouts.jsonl`, `trajectories/` |
| §3.3.2 Oracle (31–33) | A scripted sampler replays a known fix | `policy.ScriptedPolicy` + `harness.oracle_replies` |
| §4 Reward (34–35) | `reward_random` | `reward.grade` |
| §5 Sample a group (36–37) | 7 model rollouts + the oracle | `train.run` with `group_size: 8`, `oracle_per_group: 1` |
| §6 One GRPO step (38–43) | Advantages; LoRA; `build_masked`; `seq_logprob`; AdamW; log-probs before and after | `grpo.group_advantages`, `LoRATrainer.token_logprobs` / `.step`, `objective_gain`, `oracle_dlogp` |
| (lecture only) push to vLLM | — | `train.LoraSync`, `served_dlogp_vs_base` |

`swe_grpo_one_step.py` is the same pipeline without the walkthrough (`load_task`, `load_policy`, `run_agent`, `reward_random`, `add_lora`, `build_masked`, `seq_logprob`, `grpo_step`) with `GROUP_SIZE = 6` and no oracle. Both ran on node08 as gate S1, unmodified except the notebook's two `!pip` lines (`experiments/phase1-reference.json`).
- The first attempt failed: upstream loads the model and dataset without a revision, which HF offline mode cannot resolve.
- The second attempt ran online, as published, and recorded that `main` resolved to our pinned commits (`experiments/runs/phase1-reference/revisions.txt`).
- Script: 38 s, 6 rollouts, 2 wrote patches, rewards [0, 0, 0.637, 0, 0.27, 0], mean 0.151, std 0.239, loss −0.045, grad norm 0.78, 1,081,344 trainable parameters.
- Notebook: 22 code cells, 0 errors; 8 rollouts with rewards [0, 0.637, 0, 0.27, 0, 0, 0, 0.041 (oracle)], mean 0.119, loss −0.052, grad norm 0.607.

## 3. Differences between the transcript, the reference code and sarl

| # | Transcript / reference | What sarl does |
|---|---|---|
| 1 | **Degenerate groups.** The lecture (2:00) says to throw them out. The reference code keeps them: it prints "DEGENERATE GROUP", every A = 0, and it still steps. | Skips them and counts them. The first S8 run skipped 2 of 20 groups (the LANG bug made the oracle fail on pydicom-811); the two later runs skipped 0. Crashed rollouts, failed grades and infra failures are also left out of a group instead of counting as reward 0, and a group with fewer than 2 valid members is skipped. |
| 2 | **Group size.** The notebook text says G = 6 (cell 36, and "6 rollouts" in cell 13), but cell 37 runs G = 8 (7 + oracle). The script uses `GROUP_SIZE = 6` with no oracle. The lecture (1:41) says G = 8; the diagram says 16. | `group_size: 8` with `oracle_per_group: 1`, both configurable |
| 3 | **Reply length.** `utils.generate` uses `max_new_tokens=200`. The only write action the prompt teaches is a full-file heredoc, and that does not fit in 200 tokens, so model rollouts rarely write: 2 of 6 script rollouts and 2 of 7 notebook policy rollouts produced a patch. | `max_new_tokens: 1024`. Any edit command works (`sed -i`, python, heredoc). |
| 4 | **Trained ≠ sampled tokens.** The reference decodes with `skip_special_tokens=True` and `.strip()`, then re-renders and re-tokenises for training (`build_masked`). | Trains on the sampled ids (including `<\|im_end\|>`) with vLLM's log-probs. The equivalence is tested. |
| 5 | **Leaky proxy reward.** `reward_random` gives 0 for no patch and U(0,1) for any patch; notebook §4 calls it leaky. In S1 the notebook's oracle (the correct fix) drew 0.041, so its advantage was −0.361, below two model rollouts that merely wrote something (0.637 and 0.27). | Real SWE-bench tests in a fresh container, binary |
| 6 | **Training task.** The reference trains on a Verified task, astropy-12907; the lecture says so at 0:15. | Trains on the SWE-bench dev split, which shares 0 repos, ids and commits with Verified. The replica is tagged `contaminates_eval` and never reused. |
| 7 | **Exits.** The reference always runs 4 turns ("our ONLY stopping condition") with no submit. The diagram has submit, 50 turns, 32k and 30 min. | Submit, 20 turns, a 16k context, a 600 s wall clock in `basic_run.yaml` (the `Limits` default and the three S8 runs used 30 min), and 3 consecutive format errors |
| 8 | **Observation size.** The reference truncates observations to 800 characters (and the issue to 1,500). | 10k characters head + tail with a warning (mini-swe-agent); issue up to 6,000 characters |
| 9 | **Observation role.** The reference uses a user turn; the diagram uses a tool message. | `tool` for `tool_call`, `user` for `backticks` |
| 10 | **Localization leak.** The reference `SYSTEM` prompt names the target file (`cat`, `grep` and a heredoc on `astropy/modeling/separable.py`), while §1.1 says the agent is never told which file to open. | The prompts name no file |
| 11 | **Allowlist.** The lecture has a command allowlist (1:05–1:12; blocked `pip`/`setup.py` at 1:17). The reference `MockEnv` implements only ls, cat, grep, a canned pytest and heredoc writes. | Real bash with no allowlist, but `--network none` for rollouts and grading (pip and conda fail fast). A fresh grading container keeps installs out of the reward. |
| 12 | **dtype.** The reference loads the model in fp16 and relies on the fp32 LoRA adapter (1:46–1:51). | bf16 trainer base with the adapter in fp32; vLLM serves bf16 |
| 13 | **Truncation vs dropping.** `build_masked` truncates at 3,072 tokens. | Trajectories over 8,192 tokens are dropped after the advantages are computed (DAPO-style overlong filtering; counted as `dropped_overlong`), never truncated mid-trajectory |
| 14 | **Weight push.** "Push to vLLM" (1:56–1:59) is described but not implemented. | Runtime LoRA load every step, verified by `served_dlogp_vs_base` |
| 15 | **Mismatch plot.** The \|Δlogp\| plot (2:31) is shown as production telemetry. | Measured every step, pooled over all sampled policy tokens (final S8 run: mean 0.0018–0.0077, p99 ≤ 0.114, max 0.401) |
| 16 | **Partial credit** (0:53, 2:47) | Binary for now (roadmap item 7) |
| 17 | **SFT before RL** for a model that does not know the action space (2:50) | Confirmed at S6 (0/48 JSON tool calls); this is roadmap S9 |
| 18 | **Sampling.** The reference samples with top_p 0.95, and HF `generate` also applies Qwen's generation_config (top_k 20, repetition penalty 1.05) implicitly, while its loss scores the raw π. | Plain sampling by default (T 1, no top-p, top-k or penalty), so the sampled tokens come from the π the loss and the mismatch metric use. `policy.REFERENCE_SAMPLING` keeps the reference's settings; the first two S8 runs used them. |

## 4. Licensing

The upstream repository has **no license file**, so its code is not ours to redistribute.
- `reference/fetch.sh` downloads the five files at the pinned SHA into `reference/_upstream/` and verifies them against `reference/UPSTREAM.sha256`. The directory is gitignored.
- `make_cluster_nb.py` writes a cluster copy that differs only by the two `!pip` lines and the kernelspec. It records both edits in `metadata.agentic_rl`, and `test_swe_upstream.py` checks that nothing else changed.
- `sarl` is our own implementation of the diagram, not a port. Tests import the fetched module only to cross-check behaviour (advantage formula, MockEnv semantics, checksums), and those tests skip when `_upstream/` is absent.
- Phase-1 run directories (`runs/phase1/…`) contain copies of the upstream files. `runs/` is gitignored. `experiments/runs/phase1-reference/` keeps only the script's output (`run.log`), its stderr and timing (`script-time.log`) and `revisions.txt`; no upstream code.
- `experiments/phase1-reference.json` is committed. It keeps output tails, which quote short fragments of the upstream prompt and notebook output. Review it before pushing anywhere public.
- The remote (`prasad-yashdeep/Agentic-RL`) is private.
