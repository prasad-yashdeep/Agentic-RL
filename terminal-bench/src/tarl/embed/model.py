"""Text -> vector. MiniLM (all-MiniLM-L6-v2, 384-d) through plain `transformers` with masked mean
pooling — no sentence-transformers / scikit-learn install on a 96 %-full NFS. A TF-IDF embedder is
the dependency-free fallback and a useful second guard for near-duplicate detection."""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from tarl.common.paths import EMBED_CACHE, HF_CACHE

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbedModelUnavailable(RuntimeError):
    pass


def snapshot_dir(model: str = DEFAULT_MODEL, cache: Path = HF_CACHE) -> Path | None:
    base = Path(cache) / "hub" / f"models--{model.replace('/', '--')}" / "snapshots"
    if not base.exists():
        return None
    for snap in sorted(base.iterdir()):
        if (snap / "model.safetensors").exists() or (snap / "pytorch_model.bin").exists():
            return snap
    return None


def ensure_model(model: str = DEFAULT_MODEL) -> Path:
    snap = snapshot_dir(model)
    if snap is None:
        raise EmbedModelUnavailable(
            f"{model} is not in {HF_CACHE}/hub. Download once (headnode has internet): "
            f"HF_HOME={HF_CACHE} .venv/bin/python -c \"from huggingface_hub import snapshot_download as s; "
            f"s('{model}', allow_patterns=['*.json','*.txt','model.safetensors'])\"  (or: tarl embed download)"
        )
    return snap


class MiniLMEmbedder:
    """Masked-mean-pooled, L2-normalised sentence embeddings with a per-text disk cache."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str = "cpu",
        max_length: int = 256,
        batch_size: int = 16,
        cache_dir: Path | None = EMBED_CACHE,
        threads: int | None = None,
    ):
        self.model_name = model
        self.name = model.split("/")[-1]
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.threads = threads
        self._tok = self._model = None
        self.dim = 384

    def _load(self) -> None:
        if self._model is not None:
            return
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        snap = ensure_model(self.model_name)
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.threads:
            torch.set_num_threads(self.threads)
        self._tok = AutoTokenizer.from_pretrained(str(snap))
        self._model = AutoModel.from_pretrained(str(snap)).to(self.device).eval()
        self.dim = int(self._model.config.hidden_size)

    def _cache_path(self, text: str) -> Path | None:
        if self.cache_dir is None:
            return None
        h = hashlib.sha256((self.model_name + "\x00" + str(self.max_length) + "\x00" + text).encode()).hexdigest()[:32]
        return self.cache_dir / f"{h}.npy"

    def encode(self, texts: Sequence[str], *, normalize: bool = True, show_progress: bool = False) -> np.ndarray:
        texts = [str(t) for t in texts]
        out: list[np.ndarray | None] = [None] * len(texts)
        todo: list[int] = []
        for i, t in enumerate(texts):
            cp = self._cache_path(t)
            if cp is not None and cp.exists():
                try:
                    out[i] = np.load(cp)
                    continue
                except Exception:
                    pass
            todo.append(i)
        if todo:
            self._load()
            import torch

            for start in range(0, len(todo), self.batch_size):
                idx = todo[start : start + self.batch_size]
                batch = [texts[i] for i in idx]
                enc = self._tok(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    hidden = self._model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                vecs = pooled.float().cpu().numpy()
                for j, i in enumerate(idx):
                    v = vecs[j]
                    out[i] = v
                    cp = self._cache_path(texts[i])
                    if cp is not None:
                        cp.parent.mkdir(parents=True, exist_ok=True)
                        np.save(cp, v)
        X = np.stack([np.asarray(v, dtype=np.float32) for v in out])
        if normalize:
            X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
        return X


_TOKEN = re.compile(r"[a-z][a-z0-9_\-\.]{1,}")
STOPWORDS = set("the a an and or of to in on at for with by is are was were be been it its this that these those you your we our i my me they them their as from into than then so not no yes do does did done have has had will would can could should may might must if else when while where which who whom what how all any some such only also very just more most much many each other same own here there now".split())


class TfidfEmbedder:
    """Dependency-free TF-IDF (unigrams+bigrams), L2-normalised. `fit` on a corpus, `encode` anything."""

    name = "tfidf"

    def __init__(self, min_df: int = 1, max_features: int = 20_000):
        self.min_df = min_df
        self.max_features = max_features
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    @staticmethod
    def _terms(text: str) -> list[str]:
        toks = _TOKEN.findall(text.lower())
        return toks + [a + " " + b for a, b in zip(toks, toks[1:])]

    def fit(self, texts: Sequence[str]) -> "TfidfEmbedder":
        df: Counter = Counter()
        for t in texts:
            df.update(set(self._terms(t)))
        items = [(w, c) for w, c in df.items() if c >= self.min_df]
        items.sort(key=lambda x: (-x[1], x[0]))
        items = items[: self.max_features]
        self.vocab = {w: i for i, (w, _) in enumerate(items)}
        n = max(1, len(texts))
        self.idf = np.array([math.log((1 + n) / (1 + df[w])) + 1.0 for w, _ in items], dtype=np.float32)
        self.dim = len(self.vocab)
        return self

    def encode(self, texts: Sequence[str], *, normalize: bool = True) -> np.ndarray:
        if self.idf is None:
            self.fit(texts)
        X = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, t in enumerate(texts):
            for w, c in Counter(self._terms(t)).items():
                j = self.vocab.get(w)
                if j is not None:
                    X[i, j] = (1 + math.log(c)) * self.idf[j]
        if normalize:
            X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
        return X

    def top_terms(self, vector: np.ndarray, k: int = 6) -> list[str]:
        inv = {i: w for w, i in self.vocab.items()}
        out: list[str] = []
        for i in np.argsort(-vector):
            if vector[int(i)] <= 0 or len(out) >= k:
                break
            w = inv[int(i)]
            if any(part in STOPWORDS for part in w.split(" ")):
                continue
            out.append(w)
        return out


def get_embedder(prefer: str = "minilm", **kw):
    """MiniLM when its weights are cached, else TF-IDF (and say so)."""
    if prefer == "minilm" and snapshot_dir(DEFAULT_MODEL) is not None:
        return MiniLMEmbedder(**kw)
    return TfidfEmbedder()


def embed_texts(texts: Sequence[str], model: str = DEFAULT_MODEL, **kw) -> np.ndarray:
    return MiniLMEmbedder(model, **kw).encode(texts)
