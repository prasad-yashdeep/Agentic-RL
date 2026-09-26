#!/usr/bin/env python3
"""Pre-registered SWE-smith screening list: cheap single-file tasks from our four repos, hashed per repo, with the
contamination report against Verified and dev saved next to it.

    python scripts/smith_pool.py --n 50 --salt smith-v1 --out configs/screen_smith_v1.txt --guard experiments/smith-v1-guard.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sarl.paths import DATASETS
from sarl.smith import strategy
from sarl.tasks import contamination_report, hashed_subset, load_split, write_subset


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50, help="tasks per repo")
    p.add_argument("--salt", default="smith-v1")
    p.add_argument("--max-patch-chars", type=int, default=2000)
    p.add_argument("--max-f2p", type=int, default=5)
    p.add_argument("--out", required=True)
    p.add_argument("--guard", required=True)
    a = p.parse_args(argv)

    tasks = load_split("smith")  # SMITH_REPOS only; pr_ mirrors already dropped by the loader
    ok = [t for t in tasks if t.problem_statement.strip() and t.pass_to_pass and t.n_patch_files == 1
          and 1 <= len(t.fail_to_pass) <= a.max_f2p and len(t.bug_patch) <= a.max_patch_chars
          and not any(" " in x for x in t.fail_to_pass + t.pass_to_pass)]
    by_repo = Counter(t.repo for t in ok)
    chosen = []
    for repo in sorted(by_repo):
        chosen += hashed_subset([t for t in ok if t.repo == repo], a.salt, a.n)
    digest = write_subset(a.out, chosen, salt=a.salt, source=f"{DATASETS['smith'][0]}@{DATASETS['smith'][1][:8]} ({len(by_repo)} repos)",
                          note=f"filter: non-empty issue, single-file bug patch <= {a.max_patch_chars} chars, 1-{a.max_f2p} F2P, P2P in the F2P files; no pr_ mirrors")
    guard = {"n_loaded": len(tasks), "n_candidates": len(ok), "per_repo_candidates": dict(by_repo), "n_chosen": len(chosen),
             "sha256_ids": digest, "strategies": dict(Counter(strategy(t.instance_id) for t in chosen).most_common()),
             "vs_verified": contamination_report(chosen, load_split("verified")),
             "vs_dev": contamination_report(chosen, load_split("dev"))}
    Path(a.guard).write_text(json.dumps(guard, indent=1))
    print(f"loaded {len(tasks)}, candidates {len(ok)} {dict(by_repo)}, chosen {len(chosen)} -> {a.out} (sha256 {digest[:12]})")
    print("strategies:", guard["strategies"])
    # against dev, sharing a repository is the point (these are mirrors of the dev repos); ids, commits and issue text must not
    dev_hits = guard["vs_dev"]["hits"]
    guard["vs_dev"]["clean_except_repo"] = not any(dev_hits[k] for k in ("instance_id", "base_commit", "problem_statement"))
    Path(a.guard).write_text(json.dumps(guard, indent=1))
    print("clean vs verified:", guard["vs_verified"]["clean"], "| vs dev (ids/commits/issue text):", guard["vs_dev"]["clean_except_repo"],
          {k: len(v) for k, v in dev_hits.items()})
    return 0 if guard["vs_verified"]["clean"] and guard["vs_dev"]["clean_except_repo"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
