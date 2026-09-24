"""Test-based reward for one candidate patch (the diagram's REWARD box, episode end only).

    filter the patch (no edits to the test harness)  →  fresh container from the task's SIF
    →  apply chain (git apply / --reject / patch --fuzz=5, resetting between)
    →  bash eval.sh (swebench TestSpec.eval_script: install, checkout+apply test_patch, run the tests)
    →  swebench get_eval_report on the HOST  →  resolved ∈ {True, False}  →  reward 1.0 / 0.0

Integrity (the code under test is policy-written and runs as fakeroot uid 0):
  - only an input folder (patch.diff, eval.sh) is bound into the container, READ-ONLY; the test spec
    (instance.json), the log and the report live in a host-only folder the container never sees — otherwise a patch
    could rewrite FAIL_TO_PASS/PASS_TO_PASS or forge the log and score 1 (review finding, 2026-09-24)
  - the log is the container's stdout, captured by the host
  - patch sections that touch the test harness (conftest.py, pytest.ini, tox.ini, setup.cfg, test files/dirs) are
    dropped and reported; eval.sh re-checks-out and patches the task's own test files anyway
  - network off by default (`--network none`); scripts/pool_gate.py checks gold=1 / empty=0 in that setting

"Tree wiped — only the patch is replayed": a fresh container also discards anything the agent changed outside the
repo (installed packages, /tmp, the conda env).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sarl.env.apptainer import APPTAINER, SANDBOX_VMEM_KB, apptainer_env, refuse_on_headnode
from sarl.paths import GRADER_VENV, REPO, SCRATCH, refuse_on_home

GRADER_MAIN = REPO / "src" / "sarl" / "reward" / "grader_main.py"
APPLY_FAIL = ">>>>> Patch Apply Failed"
TIMEOUT_MARK = ">>>>> Tests Timed Out"
MAX_LOG_BYTES = 2_000_000

APPLY_CHAIN = r"""
cd /testbed
ulimit -v %d 2>/dev/null
ok=0
for c in "git apply --verbose" "git apply --verbose --reject" "patch --batch --fuzz=5 -p1 -i"; do
    if $c /sarl_in/patch.diff; then ok=1; echo ">>>>> Applied Patch ($c)"; break; fi
    git checkout -- . >/dev/null 2>&1; git clean -fd >/dev/null 2>&1
done
if [ "$ok" != 1 ]; then echo "%s"; exit 1; fi
bash /sarl_in/eval.sh
""" % (SANDBOX_VMEM_KB, APPLY_FAIL)

_HARNESS_FILE = re.compile(r"(^|/)(conftest\.py|pytest\.ini|tox\.ini|setup\.cfg|noxfile\.py)$|(^|/)tests?/|(^|/)test_[^/]*\.py$|_test\.py$")
_DIFF_SPLIT = re.compile(r"(?m)^(?=diff --git )")


def filter_patch(patch: str) -> tuple[str, list[str]]:
    """Drop file sections that touch the test harness. Returns (kept patch, dropped paths)."""
    if not patch.strip():
        return patch, []
    kept, dropped = [], []
    for sec in _DIFF_SPLIT.split(patch):
        if not sec.strip():
            continue
        m = re.match(r"diff --git a/(\S+) b/(\S+)", sec)
        paths = [m.group(1), m.group(2)] if m else []
        if paths and any(_HARNESS_FILE.search(p) for p in paths):
            dropped.append(paths[-1])
        else:
            kept.append(sec)
    return "".join(kept), dropped


@dataclass
class GradeResult:
    instance_id: str
    reward: float
    resolved: bool
    applied: bool
    empty_patch: bool
    f2p_passed: int = 0
    f2p_total: int = 0
    p2p_passed: int = 0
    p2p_total: int = 0
    timed_out: bool = False
    infra_failure: bool = False
    filtered_paths: list = field(default_factory=list)
    seconds: float = 0.0
    workdir: str = ""
    log_tail: str = ""
    report: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


def _grader(*args: str, timeout: int = 300) -> None:
    py = GRADER_VENV / "bin" / "python"
    if not py.exists():
        raise FileNotFoundError(f"grader venv missing at {GRADER_VENV}; run `make -C swe-bench grader`")
    r = subprocess.run([str(py), str(GRADER_MAIN), *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"grader {args[0]} failed: {r.stderr[-800:]}")


def _cap_log(log: Path) -> None:
    """Keep head+tail of huge logs (a patch can print without bound); the markers sit near the end."""
    n = log.stat().st_size
    if n <= MAX_LOG_BYTES:
        return
    with open(log, "rb") as f:
        head = f.read(MAX_LOG_BYTES // 4)
        f.seek(n - 3 * MAX_LOG_BYTES // 4)
        tail = f.read()
    log.write_bytes(head + b"\n[... log truncated by sarl ...]\n" + tail)


def grade(task, patch: str, *, sif: str | Path, workdir_root: str | Path = SCRATCH / "grade", timeout_s: int = 1800,
          run_empty: bool = False, network: bool = False, keep: bool = False, filter_harness: bool = True) -> GradeResult:
    refuse_on_headnode("grading")
    root = refuse_on_home(Path(workdir_root), "grading workdirs")
    wd = root / f"{task.instance_id}-{uuid.uuid4().hex[:8]}"
    inp, host = wd / "in", wd / "host"
    inp.mkdir(parents=True, exist_ok=True)
    host.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dropped: list[str] = []
    if filter_harness:
        patch, dropped = filter_patch(patch)
    empty = not patch.strip()
    try:
        if empty and not run_empty:
            return GradeResult(task.instance_id, 0.0, False, False, True, f2p_total=len(task.fail_to_pass),
                               p2p_total=len(task.pass_to_pass), filtered_paths=dropped,
                               report={"note": "empty patch, not run"})
        (host / "instance.json").write_text(json.dumps(task.grader_row()))
        (inp / "patch.diff").write_text(patch if patch.endswith("\n") or not patch else patch + "\n")
        _grader("spec", "--instance", str(host / "instance.json"), "--out", str(inp / "eval.sh"))
        net = [] if network else ["--net", "--network", "none"]
        cmd = [str(APPTAINER), "exec", "--fakeroot", "--containall", "--writable-tmpfs", *net, "--pwd", "/testbed",
               "-B", f"{inp}:/sarl_in:ro", str(sif), "bash", "-c", APPLY_CHAIN]
        log = host / "test_output.txt"
        timed_out = False
        with open(log, "wb") as fh:
            try:
                subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout_s, env=apptainer_env(offline=not network))
            except subprocess.TimeoutExpired:
                timed_out = True
        if timed_out:
            with open(log, "a") as fh:
                fh.write(f"\n{TIMEOUT_MARK}\n")
        _cap_log(log)
        (host / "patch.diff").write_text((inp / "patch.diff").read_text())
        _grader("report", "--instance", str(host / "instance.json"), "--patch", str(host / "patch.diff"),
                "--log", str(log), "--out", str(host / "report.json"))
        rep = json.loads((host / "report.json").read_text())
        ts = rep.get("tests_status", {})
        f2p, p2p = ts.get("FAIL_TO_PASS", {}), ts.get("PASS_TO_PASS", {})
        resolved = bool(rep.get("resolved"))
        return GradeResult(
            instance_id=task.instance_id, reward=1.0 if resolved else 0.0, resolved=resolved,
            applied=bool(rep.get("patch_successfully_applied")), empty_patch=empty,
            f2p_passed=len(f2p.get("success", [])), f2p_total=len(task.fail_to_pass),
            p2p_passed=len(p2p.get("success", [])), p2p_total=len(task.pass_to_pass),
            timed_out=timed_out, infra_failure=bool(rep.get("infra_failure")), filtered_paths=dropped,
            seconds=time.time() - t0, workdir=str(wd) if keep else "", log_tail=log.read_text(errors="replace")[-3000:],
            report=rep,
        )
    finally:
        if not keep:
            shutil.rmtree(wd, ignore_errors=True)
