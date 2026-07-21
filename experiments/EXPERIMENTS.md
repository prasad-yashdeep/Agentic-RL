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

**In flight at close of entry:** G0 attempt 3 with compat fakeroot (tmux `g0`), vllm +cu129 install
then serve relaunch (tmux `fixvllm` → `serve`); watchers armed on all.
