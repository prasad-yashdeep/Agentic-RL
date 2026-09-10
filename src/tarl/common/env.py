"""Where-am-I helpers: host role and GPU availability (see CLAUDE.md, docs/01-infra.md)."""

from __future__ import annotations

import os
import socket

HEADNODE_PREFIX = "bcm11-headnode"
GPU_HOST = "node08"


class GPUUnavailable(RuntimeError):
    """A CUDA-only code path was entered on a machine without a usable GPU."""


def hostname() -> str:
    return socket.gethostname()


def is_headnode() -> bool:
    """True on the BCM head node unless TARL_ALLOW_HEADNODE=1 (tests)."""
    if os.environ.get("TARL_ALLOW_HEADNODE") == "1":
        return False
    return hostname().startswith(HEADNODE_PREFIX)


def has_cuda() -> bool:
    try:
        import torch
    except Exception:  # torch missing or broken install
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def require_cuda(what: str) -> None:
    if not has_cuda():
        raise GPUUnavailable(
            f"{what} needs a CUDA GPU ({GPU_HOST}); host={hostname()} has none "
            "(all GPU nodes were down on 2026-09-09 — check `sinfo -N -l`)"
        )


def refuse_on_headnode(what: str) -> None:
    if is_headnode():
        raise RuntimeError(
            f"{what} must not run on the head node (standing rule: no compute on "
            f"{HEADNODE_PREFIX}); run it via `ssh node03` / `ssh node08`"
        )
