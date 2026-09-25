#!/usr/bin/env python3
"""Build the Gemma training pool from every Gemma probe/screen record: a task enters when it was mixed (solved sometimes,
not always) in at least one record and never always-solved. Prints, for the training host, which images are missing.

    python scripts/gemma_pool.py experiments/probe-gemma-pool.json experiments/probe-gemma-pool-v2.json \
        experiments/probe-gemma-pool-v3.json experiments/screen-gemma-v1.json --out configs/pool_gemma_v1.txt \
        --sif-dir /var/tmp/swe/sif-dev --host node08
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("records", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--sif-dir", default="/var/tmp/swe/sif-dev")
    p.add_argument("--host", default=None, help="check image presence on this host (ssh)")
    p.add_argument("--min-solved", type=int, default=1)
    a = p.parse_args(argv)

    tally: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for f in a.records:
        d = json.loads(Path(f).read_text())
        for iid, t in d["tasks"].items():
            eps = (t.get("arms") or {}).get("gemma", {}).get("episodes", [])
            if len(eps) < 4:
                continue
            tally[iid].append((Path(f).stem, sum(1 for e in eps if e.get("resolved")), len(eps)))
    keep, always, never = [], [], []
    for iid, runs in sorted(tally.items()):
        solved = sum(r for _, r, _ in runs)
        n = sum(k for _, _, k in runs)
        mixed_any = any(0 < r < k for _, r, k in runs)
        if solved == n:
            always.append(iid)
        elif solved >= a.min_solved and mixed_any:
            keep.append((iid, solved, n, runs))
        else:
            never.append(iid)
    lines = [f"# Gemma 4 E4B training pool: tasks solved sometimes (never always) across {len(a.records)} probe/screen records",
             "# (" + ", ".join(Path(f).stem for f in a.records) + "); pooled solved/episodes per task in the comment.",
             f"# Built by scripts/gemma_pool.py; {len(keep)} tasks; {len(always)} always-solved excluded ({', '.join(always) or 'none'}); {len(never)} never solved.", ""]
    for iid, solved, n, runs in keep:
        lines.append(f"{iid}  # {solved}/{n}: " + ", ".join(f"{name.replace('probe-gemma-', 'p').replace('screen-gemma-', 's')} {r}/{k}" for name, r, k in runs))
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}: {len(keep)} tasks, pooled {sum(s for _, s, _, _ in keep)}/{sum(n for _, _, n, _ in keep)} solved")
    for iid, solved, n, runs in keep:
        print(f"  {iid:40} {solved:2}/{n:<3} {runs}")
    if a.host:
        names = [iid for iid, *_ in keep]
        cmd = "for f in " + " ".join(names) + f"; do [ -e {a.sif_dir}/$f.sif ] && echo \"$f present\" || echo \"$f MISSING\"; done"
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", a.host, cmd], capture_output=True, text=True, timeout=60).stdout
        missing = [l.split()[0] for l in out.splitlines() if l.endswith("MISSING")]
        print(f"{a.host}: {len(names) - len(missing)} images present, {len(missing)} missing: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
