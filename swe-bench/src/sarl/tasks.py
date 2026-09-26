"""SWE-bench tasks: loading the pinned datasets offline, the Task record, pools and the contamination guard.

Training draws from the SWE-bench **dev** split (225 tasks from marshmallow, pvlib, pydicom, astroid, pyvista and
sqlfluff) and, since 2026-09-25, from SWE-smith's mirrors of the same four repos (`sarl.smith`); both are disjoint in
repos, instance ids and base commits from SWE-bench **Verified** (500 tasks from 12 other repos), which stays
evaluation-only. The reference lecture trains on a Verified task "for illustrative
purposes" (0:15); we reproduce that only in the phase-1 replica, tagged `contaminates_eval`.
"""

from __future__ import annotations

import glob
import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from sarl.paths import DATASETS, HF_CACHE, SMITH_REPOS
from sarl.smith import (base_sha8, eval_script_for, is_pr_mirror, is_smith_row, reverse_unidiff, setup_script, test_files,
                        upstream_repo)

VERIFIED_REPOS = frozenset({
    "astropy/astropy", "django/django", "matplotlib/matplotlib", "mwaskom/seaborn", "pallets/flask", "psf/requests",
    "pydata/xarray", "pylint-dev/pylint", "pytest-dev/pytest", "scikit-learn/scikit-learn", "sphinx-doc/sphinx",
    "sympy/sympy",
})


def _as_list(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return list(json.loads(x))
    return [str(v) for v in list(x)]


@dataclass(frozen=True)
class Task:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    image: str  # docker ref, e.g. swebench/sweb.eval.x86_64.<id with __ → _1776_>:latest
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    eval_script: str = ""
    log_parser: str = ""
    eval_type: str = ""
    version: str = ""
    split: str = ""
    # held out from the agent: the gold fix and the grader's test patch
    patch: str = field(default="", repr=False)
    test_patch: str = field(default="", repr=False)
    # SWE-smith only: the diff that creates the bug (gold = its reverse) and the test files the agent must not see
    bug_patch: str = field(default="", repr=False)
    hidden_files: tuple[str, ...] = ()

    @property
    def is_smith(self) -> bool:
        return bool(self.bug_patch)

    @property
    def docker_ref(self) -> str:
        return "docker://" + self.image

    @property
    def sif_name(self) -> str:
        if self.is_smith:  # one image per repository: key the file by the image, not the instance
            return self.image.rsplit("/", 1)[-1].split(":", 1)[0] + ".sif"
        return f"{self.instance_id}.sif"

    @property
    def upstream_repo(self) -> str:
        return upstream_repo(self.repo)

    @property
    def sandbox_setup(self) -> str | None:
        """Bash run in a fresh sandbox before the agent starts (None for SWE-bench tasks: the image is the task state)."""
        return setup_script(self.bug_patch, self.hidden_files, keep_tests=False) if self.is_smith else None

    def grader_setup(self, patch_path: str) -> str | None:
        return setup_script(self.bug_patch, self.hidden_files, keep_tests=True, patch_path=patch_path) if self.is_smith else None

    @property
    def n_patch_files(self) -> int:
        return self.patch.count("diff --git")

    @property
    def contaminates_eval(self) -> bool:
        return self.upstream_repo in VERIFIED_REPOS

    def grader_row(self) -> dict:
        """The dict swebench.harness.utils.make_test_spec() needs (v5 field names)."""
        return {
            "instance_id": self.instance_id, "image": self.image, "repo": self.repo, "version": self.version,
            "FAIL_TO_PASS": list(self.fail_to_pass), "PASS_TO_PASS": list(self.pass_to_pass),
            "log_parser": self.log_parser, "eval_type": self.eval_type, "eval_script": self.eval_script,
        }

    def to_json(self) -> dict:
        d = asdict(self)
        d["fail_to_pass"], d["pass_to_pass"] = list(self.fail_to_pass), list(self.pass_to_pass)
        return d


def image_for(instance_id: str) -> str:
    """SWE-bench's Docker Hub naming (swebench/image_builder): `__` becomes `_1776_`."""
    return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest"


def snapshot_dir(repo: str, revision: str, cache: Path = HF_CACHE) -> Path:
    """The pinned snapshot folder in the HF cache. Read directly: offline `load_dataset(revision=…)` still calls the Hub,
    and huggingface_hub refuses partial snapshots offline (we download only the splits we use)."""
    d = Path(cache) / "hub" / f"datasets--{repo.replace('/', '--')}" / "snapshots" / revision
    if not d.exists():
        raise FileNotFoundError(f"{repo}@{revision[:8]} not in {cache}; run `make -C swe-bench assets` on the headnode")
    return d


def _smith_row_to_task(r: dict, split: str) -> Task:
    """A SWE-smith row: bug patch -> gold by reversal; F2P test files hidden from the agent; PASS_TO_PASS restricted to
    tests in those files (what the synthesized eval script runs)."""
    bug = r["patch"] if r["patch"].endswith("\n") else r["patch"] + "\n"
    f2p = tuple(_as_list(r.get("FAIL_TO_PASS")))
    files = test_files(f2p)
    p2p = tuple(t for t in _as_list(r.get("PASS_TO_PASS")) if t.split("::", 1)[0] in files)
    image = r["image_name"] if ":" in r["image_name"] else r["image_name"] + ":latest"
    return Task(
        instance_id=r["instance_id"], repo=r["repo"], base_commit=base_sha8(r["repo"]),
        problem_statement=r.get("problem_statement") or "", image=image, fail_to_pass=f2p, pass_to_pass=p2p,
        eval_script=eval_script_for(files), log_parser="parse_log_pytest", eval_type="pass_and_fail", version="smith",
        split=split, patch=reverse_unidiff(bug), test_patch="", bug_patch=bug, hidden_files=files,
    )


def _row_to_task(r: dict, split: str) -> Task:
    if is_smith_row(r):
        return _smith_row_to_task(r, split)
    image = r.get("image") or image_for(r["instance_id"])
    return Task(
        instance_id=r["instance_id"], repo=r["repo"], base_commit=r["base_commit"],
        problem_statement=r["problem_statement"], image=image,
        fail_to_pass=tuple(_as_list(r.get("FAIL_TO_PASS"))), pass_to_pass=tuple(_as_list(r.get("PASS_TO_PASS"))),
        eval_script=r.get("eval_script") or "", log_parser=r.get("log_parser") or "", eval_type=r.get("eval_type") or "",
        version=str(r.get("version") or ""), split=split, patch=r.get("patch") or "", test_patch=r.get("test_patch") or "",
    )


@lru_cache(maxsize=8)
def load_split(name: str) -> tuple[Task, ...]:
    """name ∈ DATASETS: 'dev' (training), 'verified' (eval), 'verified_princeton' (the reference's copy)."""
    import pandas as pd

    repo, rev, split = DATASETS[name]
    files = sorted(glob.glob(str(snapshot_dir(repo, rev) / "data" / f"{split}-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no {split} parquet in {repo}@{rev[:8]}")
    if name == "smith":  # 59k rows / 4 GB of test lists: read only our repos' rows, and never the PR mirrors
        import pyarrow.parquet as pq

        tables = [pq.read_table(f, filters=[("repo", "in", list(SMITH_REPOS))]) for f in files]
        rows = [r for t in tables for r in t.to_pylist() if not is_pr_mirror(r["instance_id"])]
    else:
        rows = pd.concat([pd.read_parquet(f) for f in files]).to_dict("records")
    return tuple(_row_to_task(r, name) for r in rows)


def get_task(instance_id: str, splits: Sequence[str] = ("dev", "verified")) -> Task:
    for s in splits:
        for t in load_split(s):
            if t.instance_id == instance_id:
                return t
    raise KeyError(f"{instance_id} not in {list(splits)}")


def read_pool(path: str | Path, split: str = "dev") -> list[Task]:
    """A pool file lists instance ids, one per line; `#` starts a comment."""
    ids = [ln.split("#", 1)[0].strip() for ln in Path(path).read_text().splitlines()]
    ids = [i for i in ids if i]
    by_id = {t.instance_id: t for t in load_split(split)}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"pool ids not in split {split}: {missing}")
    return [by_id[i] for i in ids]


def select_pool(tasks: Iterable[Task], n: int, *, max_patch_chars: int = 700, max_f2p: int = 3,
                exclude: Iterable[str] = ()) -> list[Task]:
    """Smallest single-file fixes with few fail-to-pass tests: the cheapest tasks for a 0.5B policy to touch."""
    ex = set(exclude)
    cands = [t for t in tasks if t.instance_id not in ex and not t.contaminates_eval and t.n_patch_files == 1
             and len(t.patch) <= max_patch_chars and 1 <= len(t.fail_to_pass) <= max_f2p and t.pass_to_pass]
    cands.sort(key=lambda t: (len(t.patch), len(t.fail_to_pass), t.instance_id))
    return cands[:n]


def contamination_report(train: Iterable[Task], evaluation: Iterable[Task]) -> dict:
    """Structural leakage guard: shared repository (SWE-smith mirrors mapped back to owner/repo), instance id, base
    commit, or an identical problem statement (SWE-smith's `pr_` mirrors copy real issues). A semantic near-duplicate
    check is not implemented."""
    train, evaluation = list(train), list(evaluation)
    ev_repos = {t.upstream_repo for t in evaluation} | set(VERIFIED_REPOS)
    ev_ids = {t.instance_id for t in evaluation}
    ev_commits = {t.base_commit for t in evaluation}
    ev_text = {t.problem_statement.strip() for t in evaluation if t.problem_statement.strip()}
    hits = {
        "repo": sorted({t.instance_id for t in train if t.upstream_repo in ev_repos}),
        "instance_id": sorted({t.instance_id for t in train if t.instance_id in ev_ids}),
        "base_commit": sorted({t.instance_id for t in train if t.base_commit and t.base_commit in ev_commits}),
        "problem_statement": sorted({t.instance_id for t in train if t.problem_statement.strip() in ev_text}),
    }
    return {"n_train": len(train), "n_eval": len(evaluation), "hits": hits, "clean": not any(hits.values())}


def hashed_subset(tasks: Iterable[Task], salt: str, n: int, exclude: Iterable[str] = ()) -> list[Task]:
    """A fixed, pre-registered eval subset: order by sha256(salt + instance_id) and take the first n. Changing the
    subset means a new salt (never edit a subset once results exist)."""
    import hashlib

    ex = set(exclude)
    ranked = sorted((t for t in tasks if t.instance_id not in ex),
                    key=lambda t: hashlib.sha256((salt + t.instance_id).encode()).hexdigest())
    return ranked[:n]


def write_subset(path: str | Path, tasks: Sequence[Task], *, salt: str, source: str, excluded: Sequence[str] = (), note: str = "") -> str:
    """Write an id list with a provenance header; returns the sha256 of the id list (one id per line, newline-terminated)."""
    import hashlib

    ids = [t.instance_id for t in tasks]
    body = "\n".join(ids) + "\n"
    digest = hashlib.sha256(body.encode()).hexdigest()
    header = [f"# pre-registered eval subset — {source}", f"# salt: {salt}   n: {len(ids)}   excluded: {', '.join(excluded) or '-'}",
              f"# sha256(ids): {digest}", "# never edit once results exist; a new subset needs a new salt"]
    if note:
        header.append(f"# {note}")
    Path(path).write_text("\n".join(header) + "\n" + body)
    return digest
