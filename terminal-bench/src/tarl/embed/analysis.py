"""Analyses on top of embeddings: PCA map, k-means (numpy), nearest neighbours, the train/eval
leakage guard, failure-mode clusters of judged trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from tarl.common.paths import A0_JOB, JOBS_G4, PILOT_JSONL, REPO


# --------------------------------------------------------------------------- linear algebra
def pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(coords[n,2], explained_variance_ratio[2]) via SVD of the centred matrix."""
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T
    var = S**2 / max(1, len(X) - 1)
    ratio = var[:2] / var.sum() if var.sum() > 0 else np.zeros(2)
    return coords, ratio


def kmeans(X: np.ndarray, k: int, *, seed: int = 0, n_init: int = 5, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    """k-means++ with several restarts; returns (labels, centroids, inertia)."""
    n = len(X)
    if k > n or k < 1:
        raise ValueError(f"k={k} must be in [1, n={n}]")
    rng = np.random.default_rng(seed)
    best = (None, None, float("inf"))
    for _ in range(n_init):
        centers = [X[rng.integers(n)]]
        for _ in range(1, k):
            d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
            probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
            centers.append(X[rng.choice(n, p=probs)])
        C = np.array(centers)
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
            new = d.argmin(1)
            if np.array_equal(new, labels) and _ > 0:
                break
            labels = new
            for j in range(k):
                pts = X[labels == j]
                if len(pts):
                    C[j] = pts.mean(0)
        inertia = float(((X - C[labels]) ** 2).sum())
        if inertia < best[2]:
            best = (labels.copy(), C.copy(), inertia)
    return best  # type: ignore[return-value]


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    An = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-9, None)
    Bn = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-9, None)
    return An @ Bn.T


def nearest_neighbours(Q: np.ndarray, K: np.ndarray, k: int = 3, *, exclude_self: bool = False) -> list[list[tuple[int, float]]]:
    S = cosine_matrix(Q, K)
    if exclude_self:
        np.fill_diagonal(S, -np.inf)
    out = []
    for i in range(len(Q)):
        idx = np.argsort(-S[i])[:k]
        out.append([(int(j), float(S[i, j])) for j in idx])
    return out


# --------------------------------------------------------------------------- leakage guard
def leakage_report(
    train_texts: Sequence[str],
    eval_texts: Sequence[str],
    *,
    top_k: int = 3,
    threshold: float = 0.9,
    review_band: tuple[float, float] = (0.75, 0.9),
    train_labels: Sequence[str] | None = None,
    eval_labels: Sequence[str] | None = None,
    embedder=None,
    train_emb: np.ndarray | None = None,
    eval_emb: np.ndarray | None = None,
):
    """For every TRAIN item, its nearest EVAL items. A train task whose instruction is (nearly) an
    eval task's must be dropped (flagged) or reviewed (review band)."""
    import pandas as pd

    if train_emb is None or eval_emb is None:
        from tarl.embed.model import get_embedder

        embedder = embedder or get_embedder()
        if hasattr(embedder, "fit") and getattr(embedder, "idf", None) is None:
            embedder.fit(list(train_texts) + list(eval_texts))
        train_emb = embedder.encode(list(train_texts))
        eval_emb = embedder.encode(list(eval_texts))
    train_labels = list(train_labels) if train_labels is not None else [f"train_{i}" for i in range(len(train_texts))]
    eval_labels = list(eval_labels) if eval_labels is not None else [f"eval_{i}" for i in range(len(eval_texts))]
    rows = []
    for i, nns in enumerate(nearest_neighbours(train_emb, eval_emb, k=top_k)):
        for rank, (j, cos) in enumerate(nns, start=1):
            action = "drop" if cos >= threshold else ("review" if review_band[0] <= cos < review_band[1] else "keep")
            rows.append({"train_label": train_labels[i], "rank": rank, "eval_label": eval_labels[j], "cosine": cos, "flagged": cos >= threshold, "action": action})
    return pd.DataFrame(rows)


def leave_one_out_nn(labels: Sequence[str], X: np.ndarray, k: int = 1):
    """Within one set: each item's nearest other item (calibrates what 'near-duplicate' looks like)."""
    import pandas as pd

    rows = []
    for i, nns in enumerate(nearest_neighbours(X, X, k=k, exclude_self=True)):
        for rank, (j, cos) in enumerate(nns, start=1):
            rows.append({"label": labels[i], "rank": rank, "nn_label": labels[j], "cosine": cos})
    return pd.DataFrame(rows).sort_values("cosine", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- data adapters
def task_instructions(job_dirs: Sequence[str | Path] | str | Path = A0_JOB) -> tuple[list[str], list[str]]:
    """(task_names, instructions) deduplicated by task, from the first user step of any trajectory."""
    from tarl.data.instruction import task_instruction
    from tarl.data.loader import iter_trials
    from tarl.judge.compressor import load_trajectory

    if isinstance(job_dirs, (str, Path)):
        job_dirs = [job_dirs]
    seen: dict[str, str] = {}
    for jd in job_dirs:
        for t in iter_trials(jd, include_incomplete=True):
            if t.task in seen or not t.has_trajectory:
                continue
            try:
                inst = task_instruction(load_trajectory(t.trajectory_path))
            except Exception:
                inst = None
            if inst:
                seen[t.task] = inst
    names = sorted(seen)
    return names, [seen[n] for n in names]


def task_embeddings(job_dirs=A0_JOB, embedder=None) -> tuple[list[str], list[str], np.ndarray]:
    from tarl.embed.model import get_embedder

    names, texts = task_instructions(job_dirs)
    embedder = embedder or get_embedder()
    return names, texts, embedder.encode(texts)


def trajectory_texts(episodes, *, char_budget: int = 1500) -> list[str]:
    """Action-inclusive, task-only compressed transcript per episode (what the judge SHOULD see)."""
    from tarl.judge.compressor import compress

    out = []
    for ep in episodes:
        traj = ep.trajectory if hasattr(ep, "trajectory") else ep
        out.append(compress(traj or {"steps": []}, char_budget, include_actions=True, task_only=True))
    return out


def trajectory_embeddings(episodes, embedder=None, *, char_budget: int = 1500) -> np.ndarray:
    from tarl.embed.model import get_embedder

    embedder = embedder or get_embedder()
    return embedder.encode(trajectory_texts(episodes, char_budget=char_budget))


# --------------------------------------------------------------------------- clusters
@dataclass
class ClusterResult:
    labels: np.ndarray
    members: dict[int, list[int]]
    mean_score: dict[int, float | None]
    sizes: dict[int, int]
    centroids: np.ndarray
    inertia: float
    top_terms: dict[int, list[str]] = field(default_factory=dict)
    exemplar: dict[int, int] = field(default_factory=dict)

    def table(self, names: Sequence[str] | None = None):
        import pandas as pd

        rows = []
        for c in sorted(self.members):
            rows.append(
                {
                    "cluster": c,
                    "size": self.sizes[c],
                    "mean_judge": self.mean_score[c],
                    "top_terms": ", ".join(self.top_terms.get(c, [])),
                    "exemplar": names[self.exemplar[c]] if names is not None and c in self.exemplar else self.exemplar.get(c),
                    "members": [names[i] for i in self.members[c]] if names is not None else self.members[c],
                }
            )
        return pd.DataFrame(rows)


def cluster_failures(X: np.ndarray, scores: Sequence[float | None], k: int = 4, *, texts: Sequence[str] | None = None, seed: int = 0) -> ClusterResult:
    labels, C, inertia = kmeans(X, k, seed=seed)
    members = {c: [int(i) for i in np.where(labels == c)[0]] for c in range(k)}
    mean_score = {}
    for c, idx in members.items():
        vals = [float(scores[i]) for i in idx if scores[i] is not None]
        mean_score[c] = (sum(vals) / len(vals)) if vals else None
    sizes = {c: len(idx) for c, idx in members.items()}
    exemplar = {}
    for c, idx in members.items():
        if idx:
            d = ((X[idx] - C[c]) ** 2).sum(1)
            exemplar[c] = idx[int(np.argmin(d))]
    top_terms: dict[int, list[str]] = {}
    if texts is not None:
        from tarl.embed.model import TfidfEmbedder

        tf = TfidfEmbedder(min_df=1).fit(list(texts))
        T = tf.encode(list(texts))
        for c, idx in members.items():
            if idx:
                top_terms[c] = tf.top_terms(T[idx].mean(0), k=6)
    return ClusterResult(labels, members, mean_score, sizes, C, inertia, top_terms, exemplar)


# --------------------------------------------------------------------------- corpora
def proxy_train_set(pilot_jsonl: Path = PILOT_JSONL, jobs_g4: Path = JOBS_G4) -> tuple[list[str], list[str]]:
    """Stand-in for a training corpus while Endless Terminals / terminal-bench-rl are not on disk:
    the TB2 tasks used by the G4 micro-run and the judge pilot. Exact hits against the eval set
    are EXPECTED (positive control for the guard) — these are eval tasks, which is the point."""
    import json

    from tarl.data.instruction import task_instruction
    from tarl.judge.compressor import load_trajectory

    seen: dict[str, str] = {}
    for jd in sorted(Path(jobs_g4).glob("2*")):
        for tp in jd.glob("*/agent/trajectory.json"):
            task = tp.parent.parent.name.rsplit("__", 1)[0]
            if task in seen:
                continue
            inst = task_instruction(load_trajectory(tp))
            if inst:
                seen[task] = inst
    if Path(pilot_jsonl).exists():
        for line in Path(pilot_jsonl).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            p = REPO / rec["path"]
            if rec["task"] in seen or not p.exists():
                continue
            inst = task_instruction(load_trajectory(p))
            if inst:
                seen[rec["task"]] = inst
    names = sorted(seen)
    return [f"proxy-train:{n}" for n in names], [seen[n] for n in names]


def load_harbor_dataset_instructions(root: str | Path) -> tuple[list[str], list[str]]:
    """Instructions of a Harbor-format task dataset (<root>/<task>/instruction.md) — e.g. Endless
    Terminals once downloaded. Raises a clear error until then."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found — download the training corpus first (docs/02-architecture.md §3)")
    names, texts = [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        f = d / "instruction.md"
        if f.exists():
            names.append(d.name)
            texts.append(f.read_text(errors="ignore"))
    if not names:
        raise FileNotFoundError(f"no <task>/instruction.md under {root}")
    return names, texts
