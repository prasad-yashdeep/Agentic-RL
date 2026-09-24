import json

import pytest

from tarl.cli import main


def test_help(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    assert "replay" in capsys.readouterr().out


def test_coldstart(capsys):
    assert main(["coldstart", "--p", "0.02", "--G", "8"]) == 0
    assert "G=8" in capsys.readouterr().out


def test_run_prints_command_without_live(capsys):
    assert main(["run", "--task", "fix-git"]) == 0
    out = capsys.readouterr().out
    assert "harbor run" in out and "-i fix-git" in out


def test_run_live_refuses_on_headnode(monkeypatch, capsys):
    monkeypatch.setattr("tarl.common.env.hostname", lambda: "bcm11-headnode")
    monkeypatch.delenv("TARL_ALLOW_HEADNODE", raising=False)
    assert main(["run", "--task", "fix-git", "--live"]) == 2


@pytest.mark.real_data
def test_replay_and_eval(capsys, a0_job, g4_log):
    assert main(["replay", str(a0_job / "build-pmars__iqVnGoH"), "--no-color", "--judge", "none"]) == 0
    out = capsys.readouterr().out
    assert "build-pmars" in out and "reward=1.0" in out and "apt-get source pmars" in out
    assert main(["eval", str(a0_job), "--json", "-"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["summary"]["n_pass"] == 5 and d["summary"]["n_clean"] == 150
    assert main(["g4log", str(g4_log)]) == 0
    assert "rubric 61083938f358" in capsys.readouterr().out
    assert main(["replay", str(a0_job / "build-pmars__iqVnGoH"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["n_agent_steps"] == 12
