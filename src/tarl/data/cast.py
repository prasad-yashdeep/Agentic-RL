"""asciicast v2 (agent/recording.cast) and tmux pane (agent/terminus_2.pane) readers.

The cast is the ground truth of what was typed into the terminal: every ``"i"`` event is a
keystroke string sent by the harness (the first one is always the harness' own ``clear``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_ANSI = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"   # CSI sequences
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC sequences
    r"|\x1b[()][A-Z0-9]"           # charset selection
    r"|\x1b[=>78]"                 # keypad / save-restore cursor
    r"|\x1b[DME]"                  # index / reverse index
)


@dataclass
class Cast:
    header: dict
    events: list[tuple[float, str, str]] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.events[-1][0] if self.events else 0.0


def parse_cast(path: str | Path) -> Cast:
    """Tolerant asciicast v2 parser (a truncated last line is ignored)."""
    header: dict = {}
    events: list[tuple[float, str, str]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if i == 0 and isinstance(obj, dict):
                header = obj
                continue
            if isinstance(obj, list) and len(obj) == 3:
                events.append((float(obj[0]), str(obj[1]), str(obj[2])))
    return Cast(header=header, events=events)


def input_events(cast: Cast, *, skip_harness_clear: bool = True) -> list[tuple[float, str]]:
    ev = [(t, data) for t, kind, data in cast.events if kind == "i"]
    if skip_harness_clear and ev and ev[0][1].strip() == "clear":
        ev = ev[1:]
    return ev


def keystrokes(cast: Cast) -> list[str]:
    """Just the typed strings, harness `clear` removed."""
    return [data for _, data in input_events(cast)]


def output_between(cast: Cast, t0: float, t1: float) -> str:
    return "".join(data for t, kind, data in cast.events if kind == "o" and t0 <= t < t1)


def strip_ansi(text: str) -> str:
    text = _ANSI.sub("", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def load_pane(path: str | Path) -> str:
    return strip_ansi(Path(path).read_text(encoding="utf-8", errors="replace"))
