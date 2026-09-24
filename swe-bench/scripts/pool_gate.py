#!/usr/bin/env python3
"""Gate S5 for a pool: the gold patch must score 1 and the empty patch 0 under OUR sandbox (apptainer), else the task
is unusable (its gold can never be rewarded). Run on node08: python scripts/pool_gate.py [pool] [--json out]."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from sarl.env.apptainer import sif_for
from sarl.reward import grade
from sarl.tasks import read_pool


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("pool", nargs="?", default="configs/pool_dev.txt")
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    pool = read_pool(a.pool)

    def g(t):
        return t, grade(t, t.patch, sif=sif_for(t)), grade(t, "", sif=sif_for(t), run_empty=True)

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(g, pool))
    out, bad = [], 0
    for t, gold, empty in res:
        ok = gold.reward == 1.0 and empty.reward == 0.0
        bad += not ok
        p2p_fail = gold.report.get("tests_status", {}).get("PASS_TO_PASS", {}).get("failure", [])
        f2p_fail = gold.report.get("tests_status", {}).get("FAIL_TO_PASS", {}).get("failure", [])
        print(f"{'ok  ' if ok else 'FAIL'} {t.instance_id:38} gold={gold.reward} f2p {gold.f2p_passed}/{gold.f2p_total} "
              f"p2p {gold.p2p_passed}/{gold.p2p_total} ({gold.seconds:.0f}s) | empty={empty.reward}")
        for name in f2p_fail + p2p_fail:
            print(f"       failing under apptainer: {name}")
        out.append({"instance_id": t.instance_id, "ok": ok, "gold_reward": gold.reward, "empty_reward": empty.reward,
                    "f2p": [gold.f2p_passed, gold.f2p_total], "p2p": [gold.p2p_passed, gold.p2p_total],
                    "failing_tests": f2p_fail + p2p_fail, "gold_workdir": gold.workdir})
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1))
    print(f"{len(pool) - bad}/{len(pool)} tasks pass the gate")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
