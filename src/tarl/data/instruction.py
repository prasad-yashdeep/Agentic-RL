"""Pull the task instruction out of a Terminus-2 trajectory.

The first (and normally only) `user` step of an ATIF trajectory is the whole Terminus 2 system
prompt followed by the literal marker ``Task Description:`` and the initial screen. Nothing else
on disk holds the instruction (the TB2 clone lived on node08's /tmp), so this is the parser.
"""

from __future__ import annotations

from typing import Any

TASK_MARK = "Task Description:\n"
STATE_MARK = "\n\nCurrent terminal state:"


def _first_user_message(traj_or_steps: Any) -> str | None:
    if traj_or_steps is None:
        return None
    if hasattr(traj_or_steps, "trajectory"):  # Episode
        traj_or_steps = traj_or_steps.trajectory
    if isinstance(traj_or_steps, dict):
        steps = traj_or_steps.get("steps") or []
    else:
        steps = list(traj_or_steps)
    if not steps:
        return None
    first = steps[0]
    if first.get("source") not in (None, "user"):
        return None
    return str(first.get("message") or "")


def split_system_and_task(message: str) -> tuple[str, str | None, str | None]:
    """(system_prompt, instruction, terminal_state). instruction is None if markers are missing."""
    if TASK_MARK not in message:
        return message, None, None
    system, rest = message.split(TASK_MARK, 1)
    if STATE_MARK in rest:
        instruction, state = rest.split(STATE_MARK, 1)
    else:
        instruction, state = rest, None
    return system.rstrip(), instruction.strip(), (state.strip() if state is not None else None)


def task_instruction(traj_or_episode: Any) -> str | None:
    """The human task statement given to the agent, or None when it cannot be located."""
    msg = _first_user_message(traj_or_episode)
    if msg is None:
        return None
    return split_system_and_task(msg)[1]
