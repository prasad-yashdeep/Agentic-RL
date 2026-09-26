#!/usr/bin/env python3
"""Split a training pool into a train part and a pre-registered in-distribution validation part, hashed per repository,
and run the contamination guard on both.

Why: the Verified evaluation starts from a 7 % base rate for Gemma, so its 95 % CI (about ±4 pp) cannot see a small gain.
Tasks the model solves *sometimes* (the mixed tasks of a screen) start near 40 %, where a paired base-vs-trained
comparison is far more sensitive. Holding some of them out before training gives that measurement without training on it.
Both arms must be re-run with fresh seeds; the screen's own base numbers were used to select the tasks and are biased.

    python scripts/holdout_split.py configs/pool_smith_all_v1.txt --split smith --salt smith-v1-val --per-repo 5 \
        --train configs/pool_smith_v1.txt --val configs/val_smith_v1.txt --guard experiments/pool-smith-v1-guard.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from sarl.tasks import contamination_report, load_split, read_pool


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("pool")
    p.add_argument("--split", default="smith")
    p.add_argument("--salt", required=True)
    p.add_argument("--per-repo", type=int, required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--guard", required=True)
    a = p.parse_args(argv)

    src = Path(a.pool).read_text().splitlines()
    comment = {ln.split("#", 1)[0].strip(): ln.split("#", 1)[1].strip() for ln in src if "#" in ln and ln.split("#", 1)[0].strip()}
    tasks = read_pool(a.pool, a.split)
    by_repo = defaultdict(list)
    for t in tasks:
        by_repo[t.upstream_repo].append(t)
    key = lambda t: hashlib.sha256(f"{a.salt}:{t.instance_id}".encode()).hexdigest()
    val = [t for r in sorted(by_repo) for t in sorted(by_repo[r], key=key)[: a.per_repo]]
    vids = {t.instance_id for t in val}
    train = [t for t in tasks if t.instance_id not in vids]

    def write(path, ts, what):
        head = [f"# {what}: {len(ts)} tasks from {a.pool}, hashed per repository (salt {a.salt!r}, {a.per_repo} per repo held out);",
                "# written by scripts/holdout_split.py before any training on this pool.", ""]
        Path(path).write_text("\n".join(head + [f"{t.instance_id}  # {comment.get(t.instance_id, '')}".rstrip(" #") for t in ts]) + "\n")
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    h_train = write(a.train, train, "SWE-smith GRPO training pool")
    h_val = write(a.val, val, "SWE-smith in-distribution validation set (never trained on)")
    guard = {
        "pool": a.pool, "salt": a.salt, "per_repo": a.per_repo,
        "train": {"file": a.train, "n": len(train), "sha256": h_train},
        "val": {"file": a.val, "n": len(val), "sha256": h_val},
        "train_vs_val": contamination_report(train, val),
        "train_vs_verified": contamination_report(train, load_split("verified")),
    }
    Path(a.guard).write_text(json.dumps(guard, indent=1, default=str))
    print(f"train {len(train)} -> {a.train} ({h_train[:12]}), val {len(val)} -> {a.val} ({h_val[:12]})")
    for k in ("train_vs_val", "train_vs_verified"):
        r = guard[k]
        print(k, "clean" if r["clean"] else "hits:", {kk: len(vv) for kk, vv in r["hits"].items()})
    # train vs val shares repositories and base commits by design (in-distribution); ids and issue text must not overlap
    if guard["train_vs_val"]["hits"]["instance_id"] or guard["train_vs_val"]["hits"]["problem_statement"]:
        raise SystemExit("train and val share an instance id or an issue text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
