import pytest

from sarl.env import MockEnv

SRC = "def f(x):\n    return x + 1\n"


def test_state_and_patch():
    env = MockEnv({"pkg/mod.py": SRC}, ["tests/test_mod.py::test_f"])
    assert env.run("ls").output.startswith("pkg/mod.py")
    assert env.run("grep -n 'return' pkg/mod.py").output == "2:    return x + 1"
    assert env.run("python -m pytest").returncode == 1
    assert env.patch() == ""
    env.run("cat > pkg/mod.py <<'EOF'\ndef f(x):\n    return x + 2\nEOF")
    p = env.patch()
    assert "-    return x + 1" in p and "+    return x + 2" in p
    assert env.run("rm -rf /").returncode == 127


@pytest.mark.upstream
def test_matches_reference_semantics(upstream):
    """Same observations and the same patch as the reference MockEnv on the lecture's command sequence."""
    FIXED = upstream.MOCK_FILE.replace("= 1", "= right")
    cmds = ["ls", f"cat {upstream.TARGET}", "cat nope.py", f'grep -n "cright" {upstream.TARGET}', "python -m pytest",
            "pip install -e .", f"cat > {upstream.TARGET} <<'EOF'\n{FIXED}\nEOF", f"cat {upstream.TARGET}"]
    ref, ours = upstream.MockEnv(["t1", "t2"]), MockEnv({upstream.TARGET: upstream.MOCK_FILE}, ["t1", "t2"])
    for c in cmds:
        a, b = ref.run(c), ours.run(c).output
        if c.startswith("pip"):  # we append 'echo' to the implemented list
            assert a.split("\n")[0] == b.split("\n")[0]
        else:
            assert a == b, c
    assert ref.patch() == ours.patch()
    # the oracle's action sequence changes state only on the write: [F, T, F, F] (cat, write, pytest, ls)
    env = MockEnv({upstream.TARGET: upstream.MOCK_FILE})
    changed = []
    for c in [f"cat {upstream.TARGET}", f"cat > {upstream.TARGET} <<'EOF'\n{FIXED}\nEOF", "python -m pytest", "ls"]:
        before = dict(env.fs); env.run(c); changed.append(env.fs != before)
    assert changed == [False, True, False, False]
