"""Parse watch.log (scripts/watch.sh heartbeat: one line per minute on node08).

Tick format:  ``MM-DD HH:MM:SS serve=ok|DOWN tmp_free=<N>G jobs=<n>``  (local time, PDT).
Other lines (``synced ...``, ``reaped ...``, ``ALARM ...``) are kept as events, not ticks.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

TICK = re.compile(r"^(\d\d)-(\d\d) (\d\d:\d\d:\d\d) serve=(ok|DOWN) tmp_free=(\d+(?:\.\d+)?)([GMKT]?) jobs=(\d+)\s*$")
EVENT = re.compile(r"^(\d\d)-(\d\d) (\d\d:\d\d:\d\d) (ALARM .*|.*)$")
LOCAL_TZ = ZoneInfo("America/Los_Angeles")
_UNIT = {"": 1.0, "G": 1.0, "M": 1 / 1024, "K": 1 / 1024 / 1024, "T": 1024.0}


def parse(path: str | Path, *, year: int = 2026):
    """DataFrame(ts_local, ts_utc, serve_ok, tmp_free_gb, jobs) of heartbeat ticks."""
    import pandas as pd

    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = TICK.match(line.rstrip("\n"))
            if not m:
                continue
            mo, d, hms, serve, val, unit, jobs = m.groups()
            ts_local = datetime.strptime(f"{year}-{mo}-{d} {hms}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=LOCAL_TZ)
            rows.append(
                {
                    "ts_local": ts_local,
                    "ts_utc": ts_local.astimezone(ZoneInfo("UTC")),
                    "serve_ok": serve == "ok",
                    "tmp_free_gb": float(val) * _UNIT.get(unit, 1.0),
                    "jobs": int(jobs),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df


def events(path: str | Path, *, year: int = 2026):
    """DataFrame(ts_local, kind, text) of non-tick lines (ALARM / synced / reaped ...)."""
    import pandas as pd

    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if TICK.match(line):
                continue
            m = EVENT.match(line)
            if not m:
                continue
            mo, d, hms, text = m.groups()
            ts_local = datetime.strptime(f"{year}-{mo}-{d} {hms}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=LOCAL_TZ)
            kind = "alarm" if text.startswith("ALARM") else text.split(" ", 1)[0]
            rows.append({"ts_local": ts_local, "kind": kind, "text": text})
    return pd.DataFrame(rows)
