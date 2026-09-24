"""SWE-bench tasks: loading the pinned datasets offline, the Task record, pools and the contamination guard.

Training draws only from the SWE-bench **dev** split (225 tasks from marshmallow, pvlib, pydicom, astroid, pyvista and
sqlfluff), whose repos, instance ids and base commits are disjoint from SWE-bench **Verified** (500 tasks from 12
other repos), which stays evaluation-only. The reference lecture trains on a Verified task "for illustrative
purposes" (0:15); we reproduce that only in the phase-1 replica, tagged `contaminates_eval`.
"""

from __future__ import annotations

import glob
import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from sarl.paths import DATASETS, HF_CACHE

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

    @property
    def docker_ref(self) -> str:
        return "docker://" + self.image

    @property
    def sif_name(self) -> str:
        return f"{self.instance_id}.sif"

    @property
    def n_patch_files(self) -> int:
        return self.patch.count("diff --git")

    @property
    def contaminates_eval(self) -> bool:
        return self.repo in VERIFIED_REPOS

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


def _row_to_task(r: dict, split: str) -> Task:
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
    """Structural leakage guard (the semantic MiniLM check lives in scripts/pool_guard.py)."""
    train, evaluation = list(train), list(evaluation)
    ev_repos = {t.repo for t in evaluation} | set(VERIFIED_REPOS)
    ev_ids = {t.instance_id for t in evaluation}
    ev_commits = {t.base_commit for t in evaluation}
    hits = {
        "repo": sorted({t.instance_id for t in train if t.repo in ev_repos}),
        "instance_id": sorted({t.instance_id for t in train if t.instance_id in ev_ids}),
        "base_commit": sorted({t.instance_id for t in train if t.base_commit in ev_commits}),
    }
    return {"n_train": len(train), "n_eval": len(evaluation), "hits": hits, "clean": not any(hits.values())}
