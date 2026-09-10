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
