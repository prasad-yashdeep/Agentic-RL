#!/usr/bin/env python3
"""Runs inside the node-local grader venv (swebench==5.0.2), never inside the shared project venv. Standalone:
it must not import sarl.

  grader_main.py spec   --instance inst.json --out eval.sh
  grader_main.py report --instance inst.json --patch patch.diff --log test_output.txt --out report.json
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("spec"); s.add_argument("--instance", required=True); s.add_argument("--out", required=True)
    r = sp.add_parser("report"); r.add_argument("--instance", required=True); r.add_argument("--patch", required=True)
    r.add_argument("--log", required=True); r.add_argument("--out", required=True)
    a = p.parse_args(argv)

    from swebench.harness.grading import get_eval_report
    from swebench.harness.utils import make_test_spec

    inst = json.load(open(a.instance))
    spec = make_test_spec(inst)
    if a.cmd == "spec":
        open(a.out, "w").write(spec.eval_script)
        return 0
    patch = open(a.patch).read()
    pred = {"instance_id": inst["instance_id"], "model_name_or_path": "sarl", "model_patch": patch}
    rep = get_eval_report(spec, pred, a.log, include_tests_status=True)[inst["instance_id"]]
    json.dump(rep, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
