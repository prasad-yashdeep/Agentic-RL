import pytest

from tarl.judge.g4log import parse_g4_log, steps_frame, to_frame

pytestmark = pytest.mark.real_data


def test_g4_log_structure(g4_log):
    run = parse_g4_log(g4_log)
    assert run.rubric_hash == "61083938f358" and run.judge_model == "grok-4.5" and run.G == 4 and run.done
    assert [s.step for s in run.steps] == [1, 2, 3, 4, 5]
    assert all(len(s.rollouts) == 8 for s in run.steps)
    s3 = run.step(3)
    fg = s3.group("fix-git")
    assert [r.judge for r in fg] == [0.3, 0.78, 0.65, 0.3]
    assert fg[1].adv == pytest.approx(0.136) and fg[1].R == pytest.approx(0.39)
    assert run.step(4).update.loss_tok == pytest.approx(-0.0226) and run.step(4).update.grad_norm == pytest.approx(0.1444)
    assert run.step(4).update.tokens == 1537 and run.step(4).update.seconds == 600
    hashes = [s.sync.adapter_hash for s in run.steps]
    assert len(set(hashes)) == 5 and all(not s.sync.probe_changed for s in run.steps)
    assert all(r.verifier == 0.0 for s in run.steps for r in s.rollouts)
    assert to_frame(run).shape[0] == 40 and len(steps_frame(run)) == 5
