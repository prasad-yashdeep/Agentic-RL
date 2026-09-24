"""Real SWE-bench sandbox + test reward (node08: apptainer image and grader venv present)."""

import re

import pytest

SMALL_TASK = "marshmallow-code__marshmallow-1359"  # the smallest pool image (also used by conftest)
from sarl.env.apptainer import ApptainerEnv, sif_for
from sarl.reward import grade
from sarl.tasks import get_task

pytestmark = [pytest.mark.sif, pytest.mark.swe_data, pytest.mark.slow]


def test_state_persists_timeout_kills_and_patch():
    t = get_task(SMALL_TASK)
    with ApptainerEnv(sif_for(t), command_timeout=20) as env:
        assert env.run("id -u").output.strip() == "0"
        assert "testbed" in env.run("which python").output
        env.run("echo probe > sarl_probe.txt")
        assert env.run("cat sarl_probe.txt").output.strip() == "probe"
        r = env.run("sleep 600", timeout=3)
        assert r.timed_out and r.seconds < 30
        assert "+probe" in env.patch()
        # an agent that commits must not hide its changes (patch is diffed against the HEAD pinned at start)
        env.run("git -c user.email=a@b -c user.name=a commit -qam wip && echo 'x = 1' > committed_probe.py && git add -A && git -c user.email=a@b -c user.name=a commit -qm more")
        assert "committed_probe.py" in env.patch()
        # no network by default: the host's vLLM on :8001 and the internet are unreachable
        r = env.run("python -c \"import urllib.request as u; u.urlopen('http://127.0.0.1:8001/v1/models', timeout=3)\"", timeout=15)
        assert r.returncode != 0


@pytest.mark.grader
def test_reward_gold_one_empty_zero_wrong_zero():
    t = get_task(SMALL_TASK)
    sif = sif_for(t)
    gold = grade(t, t.patch, sif=sif)
    assert gold.reward == 1.0 and gold.f2p_passed == gold.f2p_total and gold.p2p_passed == gold.p2p_total
    assert grade(t, "", sif=sif, run_empty=True).reward == 0.0
    # a valid patch to the right file that does not fix the bug
    target = re.search(r"^diff --git a/(\S+)", t.patch, re.M).group(1)
    with ApptainerEnv(sif) as env:
        env.run(f"echo '# unrelated edit' >> {target}")
        wrong = env.patch()
    r = grade(t, wrong, sif=sif)
    assert wrong and r.applied and r.reward == 0.0 and r.f2p_passed < r.f2p_total
