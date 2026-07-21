# Experiment Log

Template — copy per run:

```
## <date> <run-id> — <one-line purpose>
- Arm: A0..A4          Commit: <sha>
- Model/ckpt:          Harness:          Split:
- Config: configs/<file>
- Result: pass@1 = x/N (± CI over k seeds)
- Wall-clock:          GPU-hours:        Teacher $:
- Notes / anomalies:
```

---

## 2026-07-21 day-1 — M0 plumbing bring-up (no eval run yet)

**Done:**
- Env: uv venv on node08, `pip install -e .` with pinned stack; harbor + hf CLIs via `uv tool install`.
- Apptainer 1.5.2 user-space via micromamba/conda-forge (`~/apptainer-env`) — the official
  unprivileged installer fails on this distro (zstd rpm payload vs old cpio). Alpine OCI exec verified.
- FP8 model `prithivMLmods/gemma-4-E4B-it-FP8` (~13 GB) downloaded to `~/hf-cache`.
- G0 first attempt (job `2026-07-21__03-07-46`, local clone `/tmp/tb2`, `-n 2`): **0/3 — all errored**,
  but every failure was infra, not task logic. `.sif` cache warmed as a side effect (411 MB, 3 images).

**Failures + fixes (all reproduced, all fixed in-repo):**
1. `harbor run -t <task>` → ValidationError; `-d terminal-bench@2.1` → not in registry.
   Fix: local clone + `-p /tmp/tb2` + `-i <task>`; script now takes `TARL_TB_PATH`.
2. vLLM 0.25.1 default install pulls **cu13 torch** → "NVIDIA driver too old (12080)" at engine init
   (node08 driver 570.133 caps CUDA at 12.8). Fix: `uv pip install --torch-backend=cu128`
   (in `_day1_setup.sh` + one-shot `_cu128_fix.sh`). First fix attempt silently died: tmux session
   launched without `~/.local/bin` on PATH → `uv: command not found` — scripts now set PATH themselves.
3. `--limit-mm-per-prompt` needs JSON syntax in vLLM 0.25 (`'{"image": 0, "audio": 0}'`) — fixed in `01_serve.sh`.
4. **Host `libfakeroot.so` (GLIBC_2.38) injected into containers with older glibc → every exec dies**
   (`break-filter-js-from-html` = Debian 12/glibc 2.36; the Ubuntu 24.04/glibc 2.39 images were fine).
   Harbor hardcodes `--fakeroot`; with no subuid mapping apptainer wraps with the *host* fakeroot
   command and preloads its libfakeroot into the container.
   - First fix attempt `APPTAINER_IGNORE_FAKEROOT_COMMAND=1` was WRONG: it flipped the failure —
     without fakeroot, Harbor's bootstrap can't dpkg-install python3/tmux into images missing them
     ("FATAL: cannot install /usr/bin/python3") and its `su` calls die on `setgroups` (single-uid ns).
   - Real fix: **user-space compat fakeroot** — Ubuntu focal fakeroot 1.24 debs extracted to
     `~/fakeroot-compat` (libfakeroot needs only GLIBC_2.14), patched wrapper first on PATH.
     Apptainer resolves `fakeroot-sysv` via PATH, so the compat bin shadows `/usr/bin/fakeroot-sysv`.
     Verified: exec + `su` OK in all 3 G0 images. Wired into `scripts/env.sh`.
5. Remaining two G0 trials timed out at 900 s (agent + verifier) under CPU contention from three
   concurrent first-time sif conversions. Retry: warm cache, `-n 1`, `--timeout-multiplier 2.0`.
6. cu128 torch fix done (`torch 2.11.0+cu128`, cuda available) — but PyPI vllm 0.25.1 wheel links
   `libcudart.so.13`. Fix: GitHub release wheel `vllm-0.25.1+cu129` (CUDA 12.x minor-version
   compatible with the 12.8 driver) on the cu128 torch stack — `scripts/_vllm_cu12_fix.sh`.

**M0.5 DONE (04:12): vLLM serve up + thinking verified.** Three more failures on the way there:
7. humming (FP8 repack JIT) died `CUDA_ERROR_INVALID_IMAGE`: orphaned CUDA-13 pip packages
   (nvidia-cuda-nvrtc 13.3, nvcc 13.2, 19 more) from the original cu13 install were feeding its JIT.
   Removed all bare/cu13 nvidia-* packages; reinstalled the 4 cu12 twins they clobbered (shared paths);
   pinned nvcc-cu12 12.8.93.
8. OOM at FULL cudagraph capture (14.5/15.6 GiB): `--gpu-memory-utilization 0.87` + piecewise-only graphs.
9. flashinfer JIT sampler needs nvcc (none on node08): `VLLM_USE_FLASHINFER_SAMPLER=0` → native sampler.
Verification: `/v1/models` healthy; chat probe with `chat_template_kwargs={"enable_thinking": true}`
→ `message.reasoning` populated (1.6k chars), answer in `content`, finish=stop. Trivial questions may
skip the thought channel even with thinking on (probe used a multi-step problem). Kwarg omitted → no
reasoning field + "thinking process" text leaks into content (that's the thinking-OFF arm behavior).

**Gate G1 (QLoRA trainability) — training half PASSED (04:35).** Findings, all fixed in-script:
10. `prepare_model_for_kbit_training` upcasts every non-quantized param to fp32; E4B's unquantized
    embeddings (+PLE) are ~5 GiB bf16 → 10.5 GiB fp32 alloc OOMs the A4000. Skip the helper.
11. Bare-name `target_modules` also match the vision/audio towers, which wrap projections in
    `Gemma4ClippableLinear` (PEFT-incompatible). Meta-device inspection: language_model = 258 plain
    nn.Linear ✓; vision tower = 112, audio tower = 36 Clippable ✗. → regex-scope LoRA to
    `.*language_model.*(q|k|v|o|gate|up|down)_proj$`. **Design fact for the real trainer config.**
    Result: 34.9M trainable (0.44%), loss 9.49→7.97 in one step, grad norm 20.6, zero base-grad leak.
12. **Merge-fidelity insight:** a one-step LoRA ΔW is smaller than the bf16 ulp of the base weights —
    bf16 `merge_and_unload` rounds most of it away (measured max |Δlogprob| = 0.656 vs unmerged).
    Not a merge bug: fidelity must be asserted in fp32 (<1e-4), and bf16/FP8 fold-in noise becomes an
    explicit tolerance term in the M2.6 `align_check` (trainer-vs-vLLM logprob comparison).

**⛩ GATE G1 PASSED (04:41).** fp32 merge check: max |Δlogprob| = 4.2e-5 → SMOKE_MERGE_OK, G1_SMOKE_DONE.
Gemma-4 E4B is trainable with QLoRA on one A4000 and the merge path is numerically sound.

**⛩ GATE G0 PASSED (05:15) — after finding the real container blocker.**
13. **squashfuse statx hang (the cause of EVERY verifier timeout in attempts 1–3):** rust binaries
    (uv/uvx — the toolchain every TB2 verifier bootstraps with) block forever in
    statx(AT_EMPTY_PATH) against the squashfuse-mounted rootfs; daemon sits idle in
    fuse_dev_do_read, request never answered (captured live from /proc/<pid>/task/*/syscall:
    syscall 332, wchan=request_wait_answer). curl/apt/dpkg use older syscalls → worked, which
    disguised the container as healthy. A/B proof: same `uv python install` — squashfuse ∞ hang
    vs extracted sandbox dir 2.09s. Fix: `scripts/shims/singularity` rewrites exec-type calls to
    cached sandbox extractions (build/pull untouched); wired via PATH in env.sh.
    Also raised `sessiondir max size` 64→8192 MB (ruled out as the cause, kept as headroom).
    ⚠ M1 note: sandboxes are 2–4× sif size — /tmp (16 GB) needs cache eviction before the
    89-task baseline; move image caches to node05 /local or add LRU cleanup.
    Task-set note: adaptive-rejection-sampler (R-from-source oracle, ~30 min+) and
    break-filter-js-from-html (selenium/chrome verifier) excluded from smoke → docker-fidelity lane.
**G0 attempt 4 (job 2026-07-21__05-12-48): 3/3 reward=1, mean 1.000, zero exceptions.**
Per-trial wall: fix-git 30s · large-scale-text-editing 84s · regex-log 41s (vs 1800–2400s timeouts before).

**✅ M0.6 PASSED (05:24) — Terminus 2 × our endpoint (job 2026-07-21__05-18-40).**
First live model-in-the-loop run: 3 tasks, ~5 min total, zero exceptions, rewards 0.0 (expected —
untrained policy). Transcript inspection: JSON actions parsed every turn; real commands executed in
tmux (fix-git: model ran `git status`/`git branch` and reasoned over their actual outputs across
8 steps); regex-log went 11 agent turns incl. Terminus 2's completion-confirmation protocol;
42.9k prompt / 7.0k completion tokens through the FP8 endpoint; recording.cast panes captured.
CLI gotcha: model goes in `-m openai/gemma-4-E4B-it` (harbor passes model_name itself; `--ak
model_name=` collides), api_base via `--ak`.
⚠ M1 TODO: this ran thinking-OFF — the thinking-ON baseline arm needs `chat_template_kwargs:
{"enable_thinking": true}` plumbed through harbor/litellm (extra_body) to the endpoint.

**🏁 M0 (plumbing + hard gates) COMPLETE — day 1.** All gates green: G0 containers, G1 trainability,
serve verified, agent-loop verified. Plan budgeted ~5 engineer-days; actual: one long day, 13
documented failure→fix pairs.

**Gate G0 root-cause chain (~05:00):** attempt 3 (compat fakeroot, warm cache, serial, 2x timeouts)
still lost all trials to phase timeouts. Two independent hang bugs found and fixed:
13. **harbor exec-server pipe-EOF hang**: `server.py exec_command()` uses `process.communicate()`,
    which waits for stdout EOF, not process exit — any background child (apt-postinst daemons,
    asciinema installed by this oracle, task servers) inherits the pipe and holds the request open
    until the phase timeout kills the trial. Signature matched all attempt-3 failures (agent 1800s +
    2 verifiers 1800s, logs truncated mid-command); interactive `apt-get install tmux` in the same
    image takes 3 s. Patch: file-based capture + `wait()` — `scripts/03_patch_harbor.sh` (idempotent,
    re-run after any `uv tool upgrade harbor`) + reference diff `scripts/patches/`.
14. **squashfuse statx hang**: rust binaries (uv/uvx — every TB2 verifier bootstraps uv) block in
    `statx(AT_EMPTY_PATH)` against a squashfuse-mounted rootfs; the same container from an extracted
    sandbox dir works. Fix: transparent `singularity` shim substituting cached sandbox dirs on
    exec/run/shell/instance (`scripts/shims/singularity`, commit a321fd1; wired first on PATH in
    env.sh). G0 attempt 4 (job 05-12-48) runs with both fixes active.

**Heavyweight oracle confirmation (job 05-27-15, 31 min):** `break-filter-js-from-html` **reward=1
PASSES under singularity** (earlier "docker-lane-only" call disproven — selenium/chrome works with
the sandbox shim). `adaptive-rejection-sampler` oracle still AgentTimeoutError at 1800 s with output
truncated at the identical byte (3,837) as attempt 3 — oracle-setup apt (python3+asciinema) stalls
after printing the package list; install commands are properly `-y`/noninteractive, so the block is
deeper (suspect: apt network fetch inside the sandbox-substituted container on this image).
⚠ Parked as **M1 baseline risk**: Terminus trials also run the asciinema setup, so images lacking
python3/asciinema could hit the same stall → keep `--max-retries` in baseline config and debug with
this repro if the failure rate is non-trivial. Not gate-relevant (G0 passed; oracle-only evidence).

**Night-shift incidents during A0-ON (all fixed forward, run never stopped):**
14. Orphaned apptainer runtime parents (TERM-immune after trial teardown) pinned every sandbox as
    "in-use" → janitor starved → /tmp hit 1G. `_reaper.sh` (PPID-1 + >5 min → SIGKILL) each watch
    tick; janitor emergency tier. (commit 17e50a6)
15. Big ML images (mteb/hf/gpt2) can't convert in 16G /tmp → 6 conversion RuntimeErrors. Image
    caches migrated live to /var/tmp/tarl (39G partition) with symlinks at the old baked paths;
    janitor now measures the cache partition. Remaining attempts of those tasks recover. (cb7ae2b)
16. qemu-* images ship python 3.9; generic get-pip.py refuses <3.10 → deterministic bootstrap
    failure. Harbor patch 2: versioned get-pip URL. (8aa7d82)
17. Unknown model name → litellm context-limit fallback = 1M; Terminus overruns the served 32k
    window → vLLM 400 "max_tokens must be at least 1, got 0" (1 trial). Harbor patch 3 +
    HARBOR_FALLBACK_CONTEXT_LIMIT=32768 in env.sh; applies to OFF arm and future runs — ON-arm
    occurrences remain as errored trials (documented caveat). (08399b0)
