"""The Harbor command never carries an unexpanded shell variable (a quoted '$TARL_SIF_CACHE' once made harbor cache
images into ./$TARL_SIF_CACHE on the NFS disk)."""

from __future__ import annotations

import shlex

import pytest

from tarl.serving.harbor_cmd import build_harbor_cmd, format_cmd, resolve_sif_cache


def _cache_arg(cmd: list[str]) -> str:
    return next(a for a in cmd if a.startswith("singularity_image_cache_dir="))


def test_cache_dir_resolved_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TARL_SIF_CACHE", str(tmp_path / "sif"))
    cmd = build_harbor_cmd(tb_path="/tmp/tb2", tasks=["fix-git"], k=1, jobs_dir=tmp_path / "jobs")
    assert _cache_arg(cmd) == f"singularity_image_cache_dir={tmp_path / 'sif'}"
    assert not any("$" in a for a in cmd)
    # what `tarl run` shows and executes: the shell must see the same concrete path
    shown = format_cmd(cmd)
    assert f"singularity_image_cache_dir={tmp_path / 'sif'}" in shlex.split(shown)


def test_cache_dir_explicit_and_default(monkeypatch, tmp_path):
    assert resolve_sif_cache(tmp_path / "x") == str(tmp_path / "x")
    monkeypatch.delenv("TARL_SIF_CACHE", raising=False)
    assert "$" not in resolve_sif_cache() and resolve_sif_cache().startswith("/")


def test_cache_dir_rejects_shell_variables(monkeypatch):
    monkeypatch.setenv("TARL_SIF_CACHE", "$HOME/sif")
    with pytest.raises(ValueError, match="unexpanded"):
        resolve_sif_cache()
