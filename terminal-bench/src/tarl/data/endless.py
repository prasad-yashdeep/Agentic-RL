"""Endless Terminals (arXiv:2601.16443) as the RL training corpus — Terminal-Bench stays eval-only.

The HF snapshot `obiwan96/endless-terminals` is 2,492 Harbor-format task dirs::

    task_<id>/task.toml, instruction.md,
              environment/{Dockerfile, container.def, task.json, test_initial_state.py},
              tests/{test.sh, test_final_state.py}, solution/solve.sh (+ *_summary.json model traces, skipped)

Two things our singularity backend needs that the snapshot lacks: a `docker_image` in task.toml and an
image to run. `container.def` is `Bootstrap: localimage / From: ./ubuntu_22.04.sif` plus a %post script,
so the pipeline is::

    download (shared NFS)  ->  convert  (task.toml += docker_image = <ET_SIF>/<task>.sif; tiny, NFS)
                           ->  build_sif (apptainer build --fakeroot, node-local, lazily per sampled task)

`leakage_guard` cross-checks the chosen training instructions against the TB2 instructions before a run.
"""

from __future__ import annotations

import fcntl
import json
import os
import random
import re
import shutil
import subprocess
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from tarl.common.env import refuse_on_headnode
from tarl.common.paths import ET_HARBOR, ET_ROOT, ET_SIF, ET_SPLITS, HOME

HF_DATASET = "obiwan96/endless-terminals"
BASE_SIF_NAME = "ubuntu_22.04.sif"
BASE_DOCKER = "docker://ubuntu:22.04"
SKIP_PATTERNS = ("*_summary.json",)  # 354 MB of model traces we never read


# --------------------------------------------------------------------------- snapshot
def download(root: Path = ET_ROOT, *, max_workers: int = 16) -> Path:
    """Snapshot the dataset (without the model-trace summaries) into `root`. Network + disk only."""
    from huggingface_hub import snapshot_download

    p = snapshot_download(
        HF_DATASET, repo_type="dataset", local_dir=str(root), ignore_patterns=list(SKIP_PATTERNS), max_workers=max_workers
    )
    return Path(p)


def list_tasks(root: Path = ET_ROOT) -> list[str]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("task_") and (p / "task.toml").exists())


@dataclass(frozen=True)
class ETTask:
    name: str
    path: Path
    instruction: str
    difficulty: str
    category: str
    tags: tuple[str, ...]
    agent_timeout_s: float
    verifier_timeout_s: float

    def _read(self, rel: str) -> str:
        p = self.path / rel
        return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""

    @property
    def container_def(self) -> str:
        return self._read("environment/container.def")

    @property
    def dockerfile(self) -> str:
        return self._read("environment/Dockerfile")

    @property
    def test_final(self) -> str:
        return self._read("tests/test_final_state.py")

    @property
    def test_sh(self) -> str:
        return self._read("tests/test.sh")

    @property
    def solve_sh(self) -> str:
        return self._read("solution/solve.sh")

    @property
    def n_tests(self) -> int:
        return len(re.findall(r"^\s*def test_", self.test_final, flags=re.M))


def read_task(root: Path, name: str) -> ETTask:
    path = Path(root) / name
    toml_path = path / "task.toml"
    if not toml_path.exists():
        raise FileNotFoundError(f"not an Endless Terminals task dir: {path}")
    meta = tomllib.loads(toml_path.read_text())
    md = meta.get("metadata") or {}
    return ETTask(
        name=name,
        path=path,
        instruction=(path / "instruction.md").read_text(encoding="utf-8", errors="replace") if (path / "instruction.md").exists() else "",
        difficulty=str(md.get("difficulty", "")),
        category=str(md.get("category", "")),
        tags=tuple(str(t) for t in md.get("tags", []) or []),
        agent_timeout_s=float((meta.get("agent") or {}).get("timeout_sec", 0.0) or 0.0),
        verifier_timeout_s=float((meta.get("verifier") or {}).get("timeout_sec", 0.0) or 0.0),
    )


def task_table(root: Path = ET_ROOT, names: Sequence[str] | None = None):
    """One row per task: difficulty, category, tags, instruction size, #tests, canary hits, %post shape."""
    import pandas as pd

    from tarl.data.canary import scan_text

    rows = []
    for n in names or list_tasks(root):
        t = read_task(root, n)
        post = t.container_def
        rows.append(
            {
                "task": n,
                "difficulty": t.difficulty,
                "category": t.category,
                "tags": ",".join(t.tags),
                "instruction_chars": len(t.instruction),
                "n_tests": t.n_tests,
                "canary_hits": len(scan_text(t.instruction)) + len(scan_text(t.test_final)),
                "apt_lines": len(re.findall(r"apt-get\s+install", post)),
                "uses_pip": bool(re.search(r"\bpip3?\s+install", post)),
                "localimage": is_localimage(post),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- splits
def split(names: Sequence[str], *, dev_frac: float = 0.1, seed: int = 0) -> dict[str, list[str]]:
    """Deterministic train/dev split (dev = model selection; TB2 is touched only for milestone evals)."""
    order = sorted(set(names))
    rng = random.Random(seed)
    rng.shuffle(order)
    n_dev = int(round(len(order) * dev_frac))
    return {"dev": sorted(order[:n_dev]), "train": sorted(order[n_dev:])}


def write_splits(splits: dict[str, list[str]], path: Path = ET_SPLITS, **meta) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {"n_train": len(splits["train"]), "n_dev": len(splits["dev"]), **meta}, **splits}
    path.write_text(json.dumps(payload, indent=1))
    return path


def read_splits(path: Path = ET_SPLITS) -> dict:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- conversion
def sif_path(name: str, sif_dir: Path = ET_SIF) -> Path:
    return Path(sif_dir) / f"{name}.sif"


_FROM_RE = re.compile(r"^(From:\s*)(\S+)", re.M)


def is_localimage(def_text: str) -> bool:
    """`Bootstrap: localimage` (the snapshot spells it both `Bootstrap:` and `BootStrap:`)."""
    return bool(re.search(r"^bootstrap:\s*localimage", def_text, re.M | re.I))


def render_def(def_text: str, base_sif: Path) -> str:
    """Point a `Bootstrap: localimage` definition at the absolute base image; leave other bootstraps alone."""
    if is_localimage(def_text):
        return _FROM_RE.sub(lambda m: f"{m.group(1)}{Path(base_sif)}", def_text, count=1)
    return def_text


def _with_docker_image(toml_text: str, image: str) -> str:
    """Set `docker_image` under [environment] (insert, replace, or append the section). Idempotent."""
    line = f'docker_image = "{image}"'
    if re.search(r"^\s*docker_image\s*=", toml_text, re.M):
        return re.sub(r"^\s*docker_image\s*=.*$", line, toml_text, count=1, flags=re.M)
    if re.search(r"^\[environment\]\s*$", toml_text, re.M):
        return re.sub(r"^\[environment\]\s*$", "[environment]\n" + line, toml_text, count=1, flags=re.M)
    return toml_text.rstrip("\n") + "\n\n[environment]\n" + line + "\n"


def convert_task(name: str, root: Path = ET_ROOT, out_root: Path = ET_HARBOR, sif_dir: Path = ET_SIF) -> Path:
    """Copy one task into the Harbor-run tree with `docker_image` pointing at its (future) .sif."""
    src, dst = Path(root) / name, Path(out_root) / name
    if not (src / "task.toml").exists():
        raise FileNotFoundError(f"not an Endless Terminals task dir: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*SKIP_PATTERNS, ".cache", "__pycache__"))
    toml_path = dst / "task.toml"
    toml_path.write_text(_with_docker_image(toml_path.read_text(), str(sif_path(name, sif_dir))))
    return dst


def convert(names: Iterable[str], root: Path = ET_ROOT, out_root: Path = ET_HARBOR, sif_dir: Path = ET_SIF) -> list[Path]:
    return [convert_task(n, root, out_root, sif_dir) for n in names]


def converted_tasks(out_root: Path = ET_HARBOR) -> list[str]:
    return list_tasks(out_root)


# --------------------------------------------------------------------------- images
def _apptainer() -> str:
    return shutil.which("apptainer") or str(HOME / "apptainer-env" / "bin" / "apptainer")


def _build_env(sif_dir: Path) -> dict[str, str]:
    """apptainer needs the fakeroot-compat shim first on PATH (scripts/env.sh order); temp files stay node-local."""
    env = dict(os.environ)
    env["PATH"] = f"{HOME}/fakeroot-compat/bin:{HOME}/apptainer-env/bin:" + env.get("PATH", "")
    env.setdefault("APPTAINER_TMPDIR", str(Path(sif_dir) / ".tmp"))
    env.setdefault("APPTAINER_CACHEDIR", str(Path(sif_dir) / ".cache"))
    Path(env["APPTAINER_TMPDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["APPTAINER_CACHEDIR"]).mkdir(parents=True, exist_ok=True)
    return env


@contextmanager
def _lock(target: Path):
    lock = Path(str(target) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def ensure_base_sif(sif_dir: Path = ET_SIF, *, timeout_s: int = 900) -> Path:
    sif_dir = Path(sif_dir)
    base = sif_dir / BASE_SIF_NAME
    if base.exists():
        return base
    refuse_on_headnode("apptainer pull")
    sif_dir.mkdir(parents=True, exist_ok=True)
    with _lock(base):
        if base.exists():
            return base
        tmp = sif_dir / f"{BASE_SIF_NAME}.tmp.{os.getpid()}"
        r = subprocess.run(
            [_apptainer(), "pull", "--force", str(tmp), BASE_DOCKER],
            capture_output=True, text=True, timeout=timeout_s, env=_build_env(sif_dir),
        )
        if r.returncode != 0 or not tmp.exists():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"apptainer pull {BASE_DOCKER} failed (exit {r.returncode}): {r.stderr[-800:]}")
        tmp.rename(base)
    return base


def build_sif(
    name: str,
    root: Path = ET_ROOT,
    sif_dir: Path = ET_SIF,
    *,
    force: bool = False,
    fakeroot: bool = True,
    timeout_s: int = 1800,
) -> Path:
    """`apptainer build --fakeroot <sif_dir>/<name>.sif <rendered container.def>` (node-local, locked, atomic)."""
    out = sif_path(name, sif_dir)
    if out.exists() and not force:
        return out
    refuse_on_headnode("apptainer build")
    task = read_task(root, name)
    sif_dir = Path(sif_dir)
    sif_dir.mkdir(parents=True, exist_ok=True)
    def_text = task.container_def
    if is_localimage(def_text):
        def_text = render_def(def_text, ensure_base_sif(sif_dir))
    with _lock(out):
        if out.exists() and not force:
            return out
        def_path = sif_dir / f"{name}.def"
        def_path.write_text(def_text)
        tmp = sif_dir / f"{name}.sif.tmp.{os.getpid()}"
        cmd = [_apptainer(), "build", "--force", *(["--fakeroot"] if fakeroot else []), str(tmp), str(def_path)]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=_build_env(sif_dir))
        (sif_dir / f"{name}.build.log").write_text(
            f"$ {' '.join(cmd)}\n[{time.time() - t0:.0f}s exit {r.returncode}]\n--- stdout\n{r.stdout}\n--- stderr\n{r.stderr}"
        )
        if r.returncode != 0 or not tmp.exists():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"apptainer build failed for {name} (exit {r.returncode}): {r.stderr[-800:]}")
        tmp.rename(out)
    return out


def ensure_sifs(
    names: Sequence[str],
    root: Path = ET_ROOT,
    sif_dir: Path = ET_SIF,
    *,
    workers: int = 2,
    log: Callable[[str], None] | None = None,
    **build_kw,
) -> dict[str, Path]:
    """Build every missing image for `names` (parallel), return name -> .sif. Raises listing failures."""
    names = list(dict.fromkeys(names))
    out = {n: sif_path(n, sif_dir) for n in names if sif_path(n, sif_dir).exists()}
    missing = [n for n in names if n not in out]
    if not missing:
        return out
    if any(is_localimage(read_task(root, n).container_def) for n in missing):
        ensure_base_sif(sif_dir)  # once, before the pool
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(build_sif, n, root, sif_dir, **build_kw): n for n in missing}
        for f in as_completed(futs):
            n = futs[f]
            try:
                out[n] = f.result()
                if log:
                    log(f"sif ready: {out[n]} ({out[n].stat().st_size / 1e6:.0f} MB)")
            except Exception as e:  # noqa: BLE001
                errors[n] = str(e)[-300:]
    if errors:
        raise RuntimeError(f"{len(errors)} image build(s) failed: " + json.dumps(errors)[:2000])
    return out


def sif_usage(sif_dir: Path = ET_SIF) -> dict:
    sif_dir = Path(sif_dir)
    sifs = list(sif_dir.glob("task_*.sif")) if sif_dir.exists() else []
    sbs = list(sif_dir.glob("task_*.sandbox")) if sif_dir.exists() else []
    return {
        "n_sif": len(sifs),
        "sif_gb": sum(p.stat().st_size for p in sifs) / 1e9,
        "n_sandbox": len(sbs),
        "dir": str(sif_dir),
    }


# --------------------------------------------------------------------------- leakage guard
def leakage_guard(
    names: Sequence[str],
    root: Path = ET_ROOT,
    *,
    tb_root: Path,
    threshold: float = 0.85,
    review_band: tuple[float, float] = (0.75, 0.85),
    top_k: int = 3,
    embedder=None,
):
    """Nearest-neighbour check of the chosen training instructions against every TB2 instruction."""
    from tarl.embed.analysis import leakage_report
    from tarl.embed.model import get_embedder

    tb_root = Path(tb_root)
    tb_tasks = sorted(p.name for p in tb_root.iterdir() if p.is_dir() and (p / "instruction.md").exists())
    if not tb_tasks:
        raise FileNotFoundError(f"no Harbor tasks with instruction.md under {tb_root}")
    train_texts = [read_task(root, n).instruction for n in names]
    eval_texts = [(tb_root / t / "instruction.md").read_text(encoding="utf-8", errors="replace") for t in tb_tasks]
    return leakage_report(
        train_texts, eval_texts, top_k=top_k, threshold=threshold, review_band=review_band,
        train_labels=list(names), eval_labels=tb_tasks, embedder=embedder or get_embedder(),
    )
