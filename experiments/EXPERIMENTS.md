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
