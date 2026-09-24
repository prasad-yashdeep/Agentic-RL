"""The environment interface every sandbox implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CommandResult:
    returncode: int | None
    output: str
    timed_out: bool = False
    seconds: float = 0.0


class Env(Protocol):
    def start(self) -> None: ...
    def run(self, command: str, timeout: float | None = None) -> CommandResult: ...
    def patch(self) -> str: ...
    def stop(self) -> None: ...
