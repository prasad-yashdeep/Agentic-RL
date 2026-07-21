# Source me. Single source of truth for runtime environment on node08.
export NCCL_P2P_DISABLE=1
export HF_HOME="$HOME/hf-cache"
export APPTAINER_CACHEDIR=/tmp/apptainer-cache
export TARL_SIF_CACHE=/tmp/sif-cache
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
