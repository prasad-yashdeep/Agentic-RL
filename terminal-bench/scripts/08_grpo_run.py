#!/usr/bin/env python3
"""GRPO run with per-step JSONL metrics (the forward path; scripts/06_grpo_micro.py is the frozen
G4 artifact). Arms: --lambda-judge 0 -> B1 (verifier-only), 0.5 -> B2 (judge-shaped).

    # today, CPU (node03): prove the plumbing by replaying G4 through the library
    python scripts/08_grpo_run.py --dry-run --replay-jobs jobs-g4 --g4-log g4.log --judge replay \
        --lambda-judge 0.5 --tasks fix-git regex-log --G 4 --steps 5
    # same without the Gemma tokenizer (headnode-safe smoke)
    python scripts/08_grpo_run.py --dry-run --replay-jobs jobs-g4 --g4-log g4.log --judge replay --no-tokenize
    # future real runs on node08
    python scripts/08_grpo_run.py --tasks <train tasks> --G 8 --steps 50 --lr 1e-6 --sync-every 5 --lambda-judge 0   # B1
    python scripts/08_grpo_run.py --tasks <train tasks> --G 8 --steps 50 --lr 1e-6 --sync-every 5 \
        --lambda-judge 0.5 --judge live --yes-spend --cost-cap-usd 50 --collect-rollout-details              # B2

Outputs runs/<ts>-<arm>/{config.json, manifest.json, metrics.jsonl, rollouts.jsonl, judge-audit.jsonl, summary.json, log.txt}.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tarl.training.loop import run  # noqa: E402
from tarl.training.run_config import parse_args  # noqa: E402


def main(argv=None) -> int:
    cfg = parse_args(argv)
    run_dir = run(cfg)
    print(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
