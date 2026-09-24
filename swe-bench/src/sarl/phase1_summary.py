"""Summarise a phase-1 reference run (scripts/phase1_reference.sh) into experiments/phase1-reference.json.

Parses the upstream script's stdout and the executed notebook's cell outputs; keeps the raw text of the tables the
reference prints (rewards/advantages/sup_tokens, logp before/after) so nothing is lost to a brittle parser.
"""

from __future__ import annotations

import json
import platform
import re
import socket
import sys
from pathlib import Path

from sarl.paths import EXPERIMENTS, UPSTREAM_SHA

ROLL = re.compile(r"rollout (\d+): (\d+) commands, patch (yes|no), reward ([0-9.]+)")
MEAN = re.compile(r"mean reward ([0-9.\-]+)\s+std ([0-9.\-]+)")
LOSS = re.compile(r"loss ([+\-0-9.e]+)\s+grad_norm ([0-9.e+\-]+)")
TRAIN = re.compile(r"trainable params: ([\d,]+) \|\| all params: ([\d,]+)")


def _parse(text: str) -> dict:
    d: dict = {"rollouts": [{"i": int(a), "commands": int(b), "patch": c == "yes", "reward": float(r)} for a, b, c, r in ROLL.findall(text)]}
    if m := MEAN.search(text):
        d["mean_reward"], d["std_reward"] = float(m.group(1)), float(m.group(2))
    if m := LOSS.search(text):
        d["loss"], d["grad_norm"] = float(m.group(1)), float(m.group(2))
    if m := TRAIN.search(text):
        d["trainable_params"] = int(m.group(1).replace(",", ""))
    d["degenerate_group"] = "DEGENERATE GROUP" in text
    return d


def _nb_text(path: Path) -> tuple[str, list[dict]]:
    nb = json.loads(path.read_text())
    out, cells = [], []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        txt = ""
        for o in c.get("outputs", []):
            if "text" in o:
                txt += "".join(o["text"])
            elif "data" in o and "text/plain" in o["data"]:
                txt += "".join(o["data"]["text/plain"]) + "\n"
            elif o.get("output_type") == "error":
                txt += f"ERROR {o.get('ename')}: {o.get('evalue')}\n"
        out.append(txt)
        cells.append({"cell": i, "chars": len(txt), "error": any(o.get("output_type") == "error" for o in c.get("outputs", []))})
    return "\n".join(out), cells


def main(run_dir: str) -> int:
    run = Path(run_dir)
    rec: dict = {"run_dir": str(run), "host": socket.gethostname(), "python": platform.python_version(), "upstream_sha": UPSTREAM_SHA,
                 "task": "astropy__astropy-12907", "contaminates_eval": True,
                 "note": "the reference trains on a SWE-bench Verified task for illustration (lecture 0:15); nothing from this run is reused"}
    try:
        import torch
        import transformers
        import peft

        rec["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                           "cuda": torch.cuda.is_available()}
    except Exception:
        pass
    s = run / "script" / "run.log"
    if s.exists():
        text = s.read_text()
        rec["script"] = {**_parse(text), "tail": text[-4000:]}
        t = run / "script" / "time.log"
        if t.exists():
            m = re.search(r"Elapsed \(wall clock\) time.*?: ([\d:.]+)", t.read_text())
            rec["script"]["wall_clock"] = m.group(1) if m else None
    nb = run / "nb" / "executed.ipynb"
    if nb.exists():
        text, cells = _nb_text(nb)
        rec["notebook"] = {**_parse(text), "n_code_cells": len(cells), "cells_with_errors": [c["cell"] for c in cells if c["error"]],
                           "outputs_tail": text[-6000:]}
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    out = EXPERIMENTS / "phase1-reference.json"
    out.write_text(json.dumps(rec, indent=1))
    print(f"wrote {out}: script rollouts {len(rec.get('script', {}).get('rollouts', []))}, notebook rollouts {len(rec.get('notebook', {}).get('rollouts', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
