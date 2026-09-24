"""Training: advantages, clipped surrogate, curriculum, cold-start simulator, run loop.

Import-light on purpose: `tarl.training.advantages` / `curriculum` / `coldstart` / `run_config` /
`metrics` / `rollouts` never import torch; `loss` imports torch (fine on CPU); `model` is
GPU-guarded and imports peft/bitsandbytes lazily.
"""
