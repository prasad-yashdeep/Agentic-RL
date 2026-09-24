"""Environments: the sandbox the harness executes actions in. `state` = the files; `action` = one bash command;
`observation` = (returncode, output). The candidate patch is the diff between initial and final state."""

from sarl.env.base import CommandResult, Env
from sarl.env.mock import MockEnv

__all__ = ["CommandResult", "Env", "MockEnv"]
