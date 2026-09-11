# Source me. Single source of truth for runtime environment on node08.
export NCCL_P2P_DISABLE=1
export HF_HOME="$HOME/hf-cache"
export HARBOR_FALLBACK_CONTEXT_LIMIT=32768  # match vLLM --max-model-len (patch 3, 03_patch_harbor.sh)
export APPTAINER_CACHEDIR=/var/tmp/tarl/apptainer-cache  # 39G partition; /tmp symlinks here
export TARL_SIF_CACHE=/var/tmp/tarl/sif-cache  # 39G partition; /tmp/sif-cache symlinks here
export TARL_ET_SIF=/var/tmp/tarl/et-sif  # Endless Terminals task images (tarl.data.endless), ~50 MB each
# PATH order matters, leftmost wins:
#  1. shims/          — singularity shim: rewrites exec to extracted sandbox dirs
#                       (squashfuse statx hang breaks uv/uvx in every TB2 verifier)
#  2. fakeroot-compat — focal fakeroot 1.24 (max GLIBC_2.14): host libfakeroot
#                       needs GLIBC_2.38 and kills execs in older-glibc images;
#                       apptainer resolves fakeroot-sysv from PATH
#  3. apptainer-env   — the real apptainer/singularity binaries
export PATH="$HOME/terminal-agent-rl/scripts/shims:$HOME/fakeroot-compat/bin:$HOME/apptainer-env/bin:$HOME/.local/bin:$PATH"
# Secrets (XAI_API_KEY etc.) — never in git:
[ -f "$HOME/.config/tarl/env" ] && . "$HOME/.config/tarl/env"
# Project venv, if present:
[ -f "$HOME/terminal-agent-rl/.venv/bin/activate" ] && . "$HOME/terminal-agent-rl/.venv/bin/activate"
