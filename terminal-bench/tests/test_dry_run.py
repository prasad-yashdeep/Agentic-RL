import json

import pytest

from tarl.training.loop import run
from tarl.training.metrics import read_metrics
from tarl.training.run_config import RunConfig

pytestmark = pytest.mark.real_data


def test_dry_run_parity_no_tokenizer(g4_jobs, g4_log, tmp_path, monkeypatch):
    monkeypatch.setattr("tarl.training.loop.has_cuda", lambda: False)
    cfg = RunConfig(dry_run=True, replay_jobs=g4_jobs, g4_log=g4_log, judge_mode="replay", no_tokenize=True, runs_dir=tmp_path, tasks=["fix-git", "regex-log"], G=4, steps=5)
    run_dir = run(cfg, log_to_stdout=False)
    summ = json.loads((run_dir / "summary.json").read_text())
    assert summ["parity_checked"] and summ["parity_ok"] is True
    m = read_metrics(run_dir / "metrics.jsonl")
    assert (m["kind"] == "step_summary").sum() == 5 and (m["kind"] == "update").sum() == 5
    r = read_metrics(run_dir / "rollouts.jsonl")
    assert len(r) == 40 and r["judge"].notna().all() and (r["dropped_reason"] == "no_tokenizer").sum() > 0
    assert (run_dir / "manifest.json").exists() and (run_dir / "config.json").exists()
    assert "DRY_RUN_PARITY_OK" in (run_dir / "log.txt").read_text()


@pytest.mark.gemma_tok
def test_dry_run_with_tokenizer_one_step(g4_jobs, g4_log, tmp_path, monkeypatch):
    monkeypatch.setattr("tarl.training.loop.has_cuda", lambda: False)
    cfg = RunConfig(dry_run=True, replay_jobs=g4_jobs, g4_log=g4_log, judge_mode="replay", runs_dir=tmp_path, steps=1, render="legacy06")
    run_dir = run(cfg, log_to_stdout=False)
    r = read_metrics(run_dir / "rollouts.jsonl")
    assert r["n_ids"].notna().sum() > 0 and (r["included"] | r["dropped_reason"].notna()).all()
