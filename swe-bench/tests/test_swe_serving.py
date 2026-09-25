"""Serving layout helpers: server identity across nodes, per-server slots, and the Gemma PLE-offload launcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"sarl_script_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Resp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return {"data": self._data}


SNAP = "models--prithivMLmods--gemma-4-E4B-it-FP8/snapshots/e991fcb6c8b3871071c0ad7f7343dc5e0e7e5639"


def test_same_snapshot_under_different_cache_roots_is_the_same_server(monkeypatch):
    rp = _load("rolling_probe")
    roots = {"http://node08:8001": f"/home/yashp/hf-cache/hub/{SNAP}", "http://node04:8003": f"/var/tmp/swe/hf-cache/hub/{SNAP}/",
             "http://other:8001": "/home/yashp/hf-cache/hub/models--prithivMLmods--gemma-4-E4B-it-FP8/snapshots/0123abcd"}
    monkeypatch.setattr(rp.httpx, "get", lambda url, timeout: _Resp([{"id": "policy", "root": roots[url.rsplit("/v1", 1)[0]]},
                                                                      {"id": "run-a", "root": "/var/tmp/swe/adapters/run-a"}]))
    assert rp.served("http://node04:8003") == {"policy": SNAP, "run-a": "/var/tmp/swe/adapters/run-a"}
    ident = rp.check_servers(["http://node08:8001", "http://node04:8003"], {"gemma": "policy"})
    assert set(ident) == {"http://node08:8001", "http://node04:8003"}
    with pytest.raises(SystemExit, match="different weights"):
        rp.check_servers(["http://node08:8001", "http://other:8001"], {"gemma": "policy"})


def test_per_server_takes_one_value_or_one_per_server(tmp_path):
    rp = _load("rolling_probe")
    base = ["tasks.txt", "--sif-dir", str(tmp_path), "--arm", "g=policy", "--out", str(tmp_path / "o.json"),
            "--servers", "http://a:1", "http://b:2", "http://c:3"]
    with pytest.raises(SystemExit):
        rp.main(base + ["--per-server", "8", "4"])


def test_ple_offload_recognises_the_table_by_prefix():
    pytest.importorskip("vllm")
    mod = _load("vllm_ple_offload")
    mod._vpe.VocabParallelEmbedding.__init__ = mod._orig_init  # loading the launcher patches vLLM; undo it for other tests
    assert mod.is_ple((262144, 10752), {"prefix": "model.language_model.embed_tokens_per_layer"})
    assert not mod.is_ple((262144, 2560), {"prefix": "model.language_model.embed_tokens"})
    # ParallelLMHead calls super().__init__ with every argument positional, prefix last
    assert not mod.is_ple((262144, 2560, None, None, 64, None, "lm_head"), {})
    assert mod.is_ple((262144, 10752, None, None, 64, None, "x.embed_tokens_per_layer"), {})
