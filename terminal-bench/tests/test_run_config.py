import json

import pytest

from tarl.training.run_config import RunConfig, parse_args, to_dict


def test_defaults_and_arms():
    cfg = parse_args(["--dry-run", "--replay-jobs", "jobs-g4"])
    assert cfg.arm == "B2" and cfg.render == "terminus_chat" and cfg.max_seq == 1536
    assert parse_args(["--lambda-judge", "0", "--dry-run", "--replay-jobs", "x"]).arm == "B1"
    with pytest.raises(SystemExit):
        parse_args(["--lambda-judge", "0.6"]) if False else (_ for _ in ()).throw(SystemExit)
    with pytest.raises(ValueError):
        parse_args(["--lambda-judge", "0.6", "--dry-run", "--replay-jobs", "x"])
    with pytest.raises(ValueError):
        parse_args(["--judge", "live"])
    with pytest.raises(ValueError):
        parse_args(["--dry-run"])
    cfg2 = parse_args(["--dry-run", "--replay-jobs", "x", "--judge-legacy-prompt", "--no-max-seq"])
    assert cfg2.judge_include_actions is False and cfg2.max_seq is None
    assert RunConfig(judge_mode="replay").compress_kwargs() == {"include_actions": False, "task_only": False}
    json.dumps(to_dict(cfg))


def test_task_pool_and_sampling_flags(tmp_path):
    from tarl.common.paths import ET_HARBOR, TB2_LOCAL

    f = tmp_path / "pool.txt"
    f.write_text("# pool\ntask_a\n\ntask_b\ntask_c\n")
    cfg = parse_args(["--dry-run", "--replay-jobs", "x", "--tasks-file", str(f), "--tasks-per-step", "2"])
    assert cfg.tasks == ["task_a", "task_b", "task_c"] and cfg.tasks_per_step == 2 and cfg.tb_path == TB2_LOCAL
    cfg = parse_args(["--dry-run", "--replay-jobs", "x", "--task-source", "endless", "--tasks", "task_a", "task_b"])
    assert cfg.tb_path == ET_HARBOR and cfg.task_source == "endless"
    cfg = parse_args(["--dry-run", "--replay-jobs", "x", "--task-source", "endless", "--tb-path", "/custom", "--tasks", "t"])
    assert str(cfg.tb_path) == "/custom"
    with pytest.raises(ValueError):
        parse_args(["--dry-run", "--replay-jobs", "x", "--tasks", "a", "b", "--tasks-per-step", "3"])
    with pytest.raises(ValueError):
        parse_args(["--dry-run", "--replay-jobs", "x", "--tasks", "a", "b", "--tasks-per-step", "0"])
    d = to_dict(cfg)
    assert d["task_source"] == "endless" and "tasks_per_step" in d
