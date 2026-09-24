#!/usr/bin/env bash
# One-shot: reinstall torch stack as cu128 (driver 570.133 caps CUDA at 12.8;
# vllm 0.25 pulls cu13 wheels by default → "NVIDIA driver too old" at engine init).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/terminal-agent-rl"
{
    . .venv/bin/activate \
    && uv pip install --reinstall --torch-backend=cu128 vllm \
    && uv pip freeze > configs/lockfile.txt \
    && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())" \
    && echo CU128_FIX_DONE
} 2>&1 | tee "$HOME/terminal-agent-rl/cu128-fix.log"
