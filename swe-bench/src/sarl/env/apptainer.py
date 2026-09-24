"""A real SWE-bench sandbox without Docker: one unprivileged apptainer *instance* per rollout.

Validated on node08 (2026-09-24, apptainer 1.5.2, marshmallow-1359 image):
  instance start --fakeroot --containall --writable-tmpfs <sif> <name>   (0.7 s per exec afterwards)
  - uid 0 inside (fakeroot, root-mapped userns), cwd /testbed, conda env `testbed` via /opt/miniconda3/bin/activate
  - file edits persist across execs for the instance's lifetime (RAM-backed tmpfs overlay)
  - `timeout -k 5 T bash -c CMD` kills runaway commands; backgrounded children don't hang the call because the
    command's output goes to a file inside the container, not to the host pipe (the pipe-EOF hang Harbor needed a
    patch for)
  - `instance start` has no --pid flag (instances always get their own PID namespace)
The image's HEAD is a commit named "SWE-bench" on top of base_commit, so the candidate patch is `git diff HEAD`.
"""

from __future__ import annotations

import fcntl
import os
import re
import shlex
import socket
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from sarl.env.base import CommandResult
from sarl.paths import APPTAINER, APPTAINER_CACHEDIR, APPTAINER_TMPDIR, FAKEROOT_COMPAT, SIF_DIR, refuse_on_home

ACTIVATE = "source /opt/miniconda3/bin/activate testbed"


def apptainer_env(offline: bool = False) -> dict[str, str]:
    env = dict(os.environ)
    if offline:
        # with --network none, pip/conda retry for minutes before giving up (a 0.5B policy loops on `pip install`,
        # which stalled a whole GRPO step on 2026-09-24): make them fail fast instead
        env["APPTAINERENV_PIP_NO_INDEX"] = "1"
        env["APPTAINERENV_PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        env["APPTAINERENV_CONDA_OFFLINE"] = "true"
    env["PATH"] = f"{FAKEROOT_COMPAT}:{APPTAINER.parent}:" + env.get("PATH", "")
    env["APPTAINER_CACHEDIR"] = str(APPTAINER_CACHEDIR)
    env["APPTAINER_TMPDIR"] = str(APPTAINER_TMPDIR)
    # --containall implies --cleanenv, which leaves LANG=C inside. Python 3.6 testbeds then use an ASCII filesystem
    # encoding and SWE-bench tests that touch non-ASCII paths fail even for the gold patch (pydicom-811
    # testUTF8FileName, found by scripts/pool_gate.py). APPTAINERENV_* survives --cleanenv.
    env["APPTAINERENV_LANG"] = "C.UTF-8"
    env["APPTAINERENV_LC_ALL"] = "C.UTF-8"
    return env


def refuse_on_headnode(what: str) -> None:
    if socket.gethostname().startswith("bcm11-headnode"):
        raise RuntimeError(f"{what} must not run on the head node (standing rule); use node08")


@contextmanager
def _lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def ensure_sif(docker_ref: str, sif_path: Path, *, timeout_s: int = 1800) -> Path:
    """`apptainer pull` once per image (locked, atomic). Caches and the SIF stay node-local."""
    sif_path = Path(sif_path)
    if sif_path.exists():
        return sif_path
    refuse_on_headnode("apptainer pull")
    refuse_on_home(sif_path.parent, "SIF images")
    for d in (sif_path.parent, APPTAINER_CACHEDIR, APPTAINER_TMPDIR):
        refuse_on_home(d, "apptainer cache/tmp")
        Path(d).mkdir(parents=True, exist_ok=True)
    with _lock(sif_path):
        if sif_path.exists():
            return sif_path
        tmp = sif_path.with_name(sif_path.name + f".tmp.{os.getpid()}")
        r = subprocess.run([str(APPTAINER), "pull", "--force", str(tmp), docker_ref], capture_output=True, text=True,
                           timeout=timeout_s, env=apptainer_env())
        if r.returncode != 0 or not tmp.exists():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"apptainer pull {docker_ref} failed ({r.returncode}): {r.stderr[-600:]}")
        tmp.rename(sif_path)
    return sif_path


def sif_for(task, sif_dir: Path = SIF_DIR) -> Path:
    return Path(sif_dir) / task.sif_name


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]", "-", s)[:40].strip("-")


class ApptainerEnv:
    """One persistent sandbox. Use as a context manager: `with ApptainerEnv(sif) as env: env.run("ls")`."""

    def __init__(self, sif: str | Path, *, name: str | None = None, prefix: str = "sarl-", workdir: str = "/testbed",
                 command_timeout: float = 120, network: bool = False, max_output_chars: int = 200_000,
                 activate: str = ACTIVATE):
        # network=False by default: a sandbox that shares the host network can reach the vLLM LoRA admin API on
        # localhost, the cluster, and the internet (upstream fix commits = a solution lookup). --network none gives
        # it a private loopback only (unprivileged-capable).
        self.sif = Path(sif)
        self.name = name or f"{prefix}{_safe_name(self.sif.stem)}-{uuid.uuid4().hex[:8]}"
        self.base_sha: str | None = None
        self.workdir = workdir
        self.command_timeout = command_timeout
        self.network = network
        self.max_output_chars = max_output_chars
        self.activate = activate
        self.calls: list[str] = []
        self._env = apptainer_env(offline=not network)
        self._started = False

    # lifecycle -------------------------------------------------------------------------------------------
    def start(self) -> None:
        refuse_on_headnode("an apptainer sandbox")
        if not self.sif.exists():
            raise FileNotFoundError(f"image not pulled: {self.sif}")
        net = [] if self.network else ["--net", "--network", "none"]
        cmd = [str(APPTAINER), "instance", "start", "--fakeroot", "--containall", "--writable-tmpfs", *net, str(self.sif), self.name]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=self._env)
        except subprocess.TimeoutExpired:
            self._force_stop()
            raise RuntimeError(f"apptainer instance start timed out ({self.name})")
        if r.returncode != 0 or "started successfully" not in (r.stdout + r.stderr):
            self._force_stop()  # a half-started instance must not leak
            raise RuntimeError(f"apptainer instance start failed ({r.returncode}): {(r.stderr or r.stdout)[-600:]}")
        self._started = True
        # pin the diff base now: an agent that commits would otherwise move HEAD and hide its own changes
        p = self._exec("git rev-parse HEAD", host_timeout=60)
        self.base_sha = p.stdout.decode().strip() if p.returncode == 0 else None
        if not self.base_sha:
            self.stop()
            raise RuntimeError(f"could not read the repository HEAD in {self.name}")

    def _force_stop(self) -> None:
        subprocess.run([str(APPTAINER), "instance", "stop", self.name], capture_output=True, text=True, timeout=120, env=self._env)

    def stop(self) -> None:
        if self._started:
            try:
                self._force_stop()
            except subprocess.TimeoutExpired:
                pass  # train.run() sweeps leftovers by prefix at start and end
            self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    # actions ---------------------------------------------------------------------------------------------
    def _exec(self, script: str, host_timeout: float) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(APPTAINER), "exec", "--pwd", self.workdir, f"instance://{self.name}", "bash", "-c", script],
            capture_output=True, timeout=host_timeout, env=self._env,
        )

    def run(self, command: str, timeout: float | None = None) -> CommandResult:
        """Run one bash command in a fresh subshell inside the persistent sandbox (conda `testbed`, cwd /testbed)."""
        if not self._started:
            raise RuntimeError("sandbox not started")
        self.calls.append(command)
        t = float(timeout or self.command_timeout)
        script = (f"{self.activate} >/dev/null 2>&1; cd {shlex.quote(self.workdir)}; "
                  f"timeout -k 5 {int(t)} bash -c {shlex.quote(command)} > /tmp/.sarl_out 2>&1 < /dev/null; "
                  "rc=$?; cat /tmp/.sarl_out; exit $rc")
        t0 = time.time()
        try:
            p = self._exec(script, host_timeout=t + 60)
        except subprocess.TimeoutExpired:
            return CommandResult(None, "[sarl] the sandbox did not return; command abandoned", True, time.time() - t0)
        out = p.stdout.decode("utf-8", errors="replace")
        if len(out) > self.max_output_chars:
            out = out[: self.max_output_chars // 2] + "\n[... output truncated by the harness ...]\n" + out[-self.max_output_chars // 2:]
        timed_out = p.returncode in (124, 137)
        return CommandResult(p.returncode, out, timed_out, time.time() - t0)

    def patch(self) -> str:
        """The candidate patch: every change (incl. new and committed files) relative to the HEAD recorded at start.
        Read through its own exec: stdout only (stderr discarded), never truncated. Raises if git fails, so a lost
        patch is an error, not a silent empty patch (= reward 0)."""
        if not self._started:
            raise RuntimeError("sandbox not started")
        script = (f"cd {shlex.quote(self.workdir)} && git -c core.fileMode=false add -A >/dev/null 2>&1 && "
                  f"git -c core.fileMode=false diff --cached --binary {self.base_sha} 2>/dev/null")
        p = self._exec(script, host_timeout=120)
        if p.returncode != 0:
            raise RuntimeError(f"git diff failed in {self.name} (rc {p.returncode})")
        return p.stdout.decode("utf-8", errors="replace")


def list_instances(prefix: str = "sarl-") -> list[str]:
    r = subprocess.run([str(APPTAINER), "instance", "list"], capture_output=True, text=True, timeout=60, env=apptainer_env())
    names = []
    for line in r.stdout.splitlines()[1:]:
        parts = line.split()
        if parts and parts[0].startswith(prefix):
            names.append(parts[0])
    return names


def stop_all(prefix: str = "sarl-") -> int:
    """Stop leftover sandboxes (e.g. after a crashed run)."""
    n = 0
    for name in list_instances(prefix):
        subprocess.run([str(APPTAINER), "instance", "stop", name], capture_output=True, timeout=120, env=apptainer_env())
        n += 1
    return n
