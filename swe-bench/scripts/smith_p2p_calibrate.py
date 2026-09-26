#!/usr/bin/env python3
"""Calibrate PASS_TO_PASS for SWE-smith tasks whose gold patch fails the offline gate only on pass-to-pass tests: grade
the gold patch again with the report kept, record the P2P tests that fail under gold, and write them as exclusions the
loader subtracts (`experiments/smith-p2p-exclusions.json`). Such tests cannot pass offline whatever the agent does, so
they carry no information about the patch. Run where the images and the grader venv are (node04).

    SARL_SANDBOX_VMEM_GB=4 python scripts/smith_p2p_calibrate.py experiments/gate-smith-v1.json --sif-dir /var/tmp/swe/sif-smith \
        --out experiments/smith-p2p-exclusions.json --workers 4
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sarl.env.apptainer import sif_for
from sarl.reward import grade
from sarl.tasks import get_task


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("gate_json")
    p.add_argument("--sif-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    gate = json.loads(Path(a.gate_json).read_text())["tasks"]
    todo = [iid for iid, r in gate.items() if (r.get("gate") or {}).get("ok") is False and (r["gate"].get("gold_f2p") or "0/1").split("/")[0] == (r["gate"].get("gold_f2p") or "0/1").split("/")[1]]
    print(f"{len(todo)} gate failures with all F2P passing under gold", flush=True)
    excl = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}

    def one(iid):
        t = get_task(iid, ("smith",))
        g = grade(t, t.patch, sif=sif_for(t, Path(a.sif_dir)))
        ts = g.report.get("tests_status", {})
        f2p_fail = ts.get("FAIL_TO_PASS", {}).get("failure", [])
        p2p_fail = ts.get("PASS_TO_PASS", {}).get("failure", [])
        return iid, g.resolved, f2p_fail, p2p_fail, g.seconds

    with ThreadPoolExecutor(a.workers) as ex:
        for iid, resolved, f2p_fail, p2p_fail, secs in ex.map(one, todo):
            if f2p_fail:
                print(f"  {iid}: F2P still failing under gold {f2p_fail[:3]} -> not calibrated", flush=True)
                continue
            excl[iid] = sorted(set(excl.get(iid, [])) | set(p2p_fail))
            print(f"  {iid}: {len(p2p_fail)} P2P tests excluded ({secs:.0f} s)", flush=True)
    Path(a.out).write_text(json.dumps(excl, indent=1, sort_keys=True))
    print(f"wrote {a.out}: {len(excl)} tasks, {sum(len(v) for v in excl.values())} tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
