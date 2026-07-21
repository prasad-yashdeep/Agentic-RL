# Source me. Single source of truth for runtime environment on node08.
export NCCL_P2P_DISABLE=1
export HF_HOME="$HOME/hf-cache"
export APPTAINER_CACHEDIR=/tmp/apptainer-cache
export TARL_SIF_CACHE=/tmp/sif-cache
# Host libfakeroot.so needs GLIBC_2.38 → breaks any container with older glibc.
# Skipping fakeroot entirely breaks Harbor's dpkg bootstrap + su instead. Fix:
# focal fakeroot 1.24 (max GLIBC_2.14) extracted user-space; apptainer resolves
# fakeroot-sysv from PATH, so this dir must precede /usr/bin.
export PATH="$HOME/fakeroot-compat/bin:$PATH"
export PATH="$HOME/apptainer/bin:$HOME/apptainer-env/bin:$HOME/.local/bin:$PATH"
# Secrets (XAI_API_KEY etc.) — never in git:
[ -f "$HOME/.config/tarl/env" ] && . "$HOME/.config/tarl/env"
# Project venv, if present:
[ -f "$HOME/terminal-agent-rl/.venv/bin/activate" ] && . "$HOME/terminal-agent-rl/.venv/bin/activate"
