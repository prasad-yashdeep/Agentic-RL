#!/usr/bin/env python3
"""Cluster copy of the upstream notebook: identical except cell 1 drops its two `!pip` lines (installing into the
shared venv would re-resolve the pinned CUDA stack) and the kernelspec points at the shared-venv kernel `sarl`.
The declared edits are recorded in metadata.agentic_rl; tests/test_swe_upstream.py checks nothing else changed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "_upstream" / "swe_grpo_one_step.ipynb"
DST = HERE / "_upstream" / "swe_grpo_one_step.cluster.ipynb"
SHA = "8cddfb60c93cc6b2c7d9dc047939084858f30f76"


def main() -> None:
    nb = json.loads(SRC.read_text())
    cell = nb["cells"][1]
    before = list(cell["source"])
    cell["source"] = [ln for ln in before if not ln.lstrip().startswith("!pip")]
    dropped = [ln.rstrip("\n") for ln in before if ln.lstrip().startswith("!pip")]
    nb["metadata"]["kernelspec"] = {"name": "sarl", "display_name": "sarl (Agentic-RL .venv)", "language": "python"}
    nb["metadata"]["agentic_rl"] = {
        "upstream": f"https://github.com/abgoswam/swe_in_prod_vizuara_01/blob/{SHA}/swe_grpo_one_step.ipynb",
        "upstream_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
        "edits": {"cell_1_dropped_lines": dropped, "kernelspec": "sarl"},
    }
    DST.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {DST.name}: dropped {len(dropped)} line(s) from cell 1")


if __name__ == "__main__":
    main()
