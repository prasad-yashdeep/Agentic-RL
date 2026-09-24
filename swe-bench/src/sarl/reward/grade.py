"""Test-based reward for one candidate patch (the diagram's REWARD box, episode end only).

    fresh container from the task's SIF  →  apply chain (git apply / --reject / patch --fuzz=5, resetting between)
    →  bash eval.sh (swebench TestSpec.eval_script: install, checkout+apply test_patch, run the tests)
    →  swebench get_eval_report  →  resolved ∈ {True, False}  →  reward 1.0 / 0.0

"Tree wiped — only the patch is replayed": using a fresh container (not the rollout's instance) also discards
anything the agent changed outside the repo (installed packages, /tmp, the conda env).
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sarl.env.apptainer import APPTAINER, apptainer_env, refuse_on_headnode
from sarl.paths import GRADER_VENV, REPO, SCRATCH, refuse_on_home

GRADER_MAIN = REPO / "src" / "sarl" / "reward" / "grader_main.py"
APPLY_FAIL = ">>>>> Patch Apply Failed"
TIMEOUT_MARK = ">>>>> Tests Timed Out"

APPLY_CHAIN = r"""
cd /testbed
ok=0
for c in "git apply --verbose" "git apply --verbose --reject" "patch --batch --fuzz=5 -p1 -i"; do
    if $c /sarl_run/patch.diff; then ok=1; echo ">>>>> Applied Patch ($c)"; break; fi
    git checkout -- . >/dev/null 2>&1; git clean -fd >/dev/null 2>&1
done
if [ "$ok" != 1 ]; then echo "%s"; exit 1; fi
bash /sarl_run/eval.sh
""" % APPLY_FAIL


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
    seconds: float = 0.0
    workdir: str = ""
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


def grade(task, patch: str, *, sif: str | Path, workdir_root: str | Path = SCRATCH / "grade", timeout_s: int = 1800,
          run_empty: bool = False, network: bool = True) -> GradeResult:
    refuse_on_headnode("grading")
    root = refuse_on_home(Path(workdir_root), "grading workdirs")
    wd = root / f"{task.instance_id}-{uuid.uuid4().hex[:8]}"
    wd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    empty = not patch.strip()
    if empty and not run_empty:
        return GradeResult(task.instance_id, 0.0, False, False, True, f2p_total=len(task.fail_to_pass),
                           p2p_total=len(task.pass_to_pass), seconds=0.0, workdir=str(wd), report={"note": "empty patch, not run"})
    (wd / "instance.json").write_text(json.dumps(task.grader_row()))
    (wd / "patch.diff").write_text(patch if patch.endswith("\n") or not patch else patch + "\n")
    _grader("spec", "--instance", str(wd / "instance.json"), "--out", str(wd / "eval.sh"))
    net = [] if network else ["--net", "--network", "none"]
    cmd = [str(APPTAINER), "exec", "--fakeroot", "--containall", "--writable-tmpfs", *net, "--pwd", "/testbed",
           "-B", f"{wd}:/sarl_run", str(sif), "bash", "-c", APPLY_CHAIN]
    log = wd / "test_output.txt"
    timed_out = False
    with open(log, "wb") as fh:
        try:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout_s, env=apptainer_env())
        except subprocess.TimeoutExpired:
            timed_out = True
    if timed_out:
        with open(log, "a") as fh:
            fh.write(f"\n{TIMEOUT_MARK}\n")
    _grader("report", "--instance", str(wd / "instance.json"), "--patch", str(wd / "patch.diff"),
            "--log", str(log), "--out", str(wd / "report.json"))
    rep = json.loads((wd / "report.json").read_text())
    ts = rep.get("tests_status", {})
    f2p, p2p = ts.get("FAIL_TO_PASS", {}), ts.get("PASS_TO_PASS", {})
    resolved = bool(rep.get("resolved"))
    return GradeResult(
        instance_id=task.instance_id, reward=1.0 if resolved else 0.0, resolved=resolved,
        applied=bool(rep.get("patch_successfully_applied")), empty_patch=empty,
        f2p_passed=len(f2p.get("success", [])), f2p_total=len(task.fail_to_pass),
        p2p_passed=len(p2p.get("success", [])), p2p_total=len(task.pass_to_pass),
        timed_out=timed_out, infra_failure=bool(rep.get("infra_failure")), seconds=time.time() - t0,
        workdir=str(wd), report=rep,
    )


def eval_command_preview(task) -> str:
    """Human-readable preview of what grading runs (for docs/notebook)."""
    return f"apptainer exec --fakeroot --containall --writable-tmpfs -B <wd>:/sarl_run {shlex.quote(task.sif_name)} bash -c '<apply chain>; bash /sarl_run/eval.sh'"
