"""SWE-smith support: the reversed diff is the gold fix, a row becomes a Task with hidden test files and a synthesized
eval script, the sandbox setup leaves no history, and the contamination guard maps mirrors back to their repos."""

from __future__ import annotations

import subprocess

import pytest

from sarl.smith import MARK_END, MARK_START, is_pr_mirror, reverse_unidiff, setup_script, strategy, upstream_repo
from sarl.tasks import Task, _row_to_task, contamination_report

BUG = """diff --git a/pkg/mod.py b/pkg/mod.py
index 1111111..2222222 100644
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,4 +1,4 @@
 def f(x):
-    return x + 1
+    return x - 1

 def g():
"""

ROW = {"instance_id": "owner__proj.abcdef12.func_basic__zz9y8x7w", "repo": "swesmith/owner__proj.abcdef12",
       "image_name": "jyangballin/swesmith.x86_64.owner_1776_proj.abcdef12", "problem_statement": "f is off by one",
       "patch": BUG, "FAIL_TO_PASS": ["tests/test_mod.py::test_f"],
       "PASS_TO_PASS": ["tests/test_mod.py::test_g", "tests/test_other.py::test_z"]}


def test_reverse_unidiff_is_the_fix_and_round_trips(tmp_path):
    fix = reverse_unidiff(BUG)
    assert "-    return x - 1" in fix and "+    return x + 1" in fix and "@@ -1,4 +1,4 @@" in fix
    assert reverse_unidiff(fix) == BUG
    # apply the bug to a real file, then the reverse: back to the original
    repo = tmp_path
    (repo / "pkg").mkdir()
    src = "def f(x):\n    return x + 1\n\ndef g():\n    return 2\n"
    (repo / "pkg" / "mod.py").write_text(src)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "apply", "-"], cwd=repo, input=BUG, text=True, check=True)
    assert "x - 1" in (repo / "pkg" / "mod.py").read_text()
    subprocess.run(["git", "apply", "-"], cwd=repo, input=fix, text=True, check=True)
    assert (repo / "pkg" / "mod.py").read_text() == src


def test_reverse_unidiff_new_file_becomes_deleted_file():
    new = "diff --git a/n.py b/n.py\nnew file mode 100644\nindex 0000000..e69de29\n--- /dev/null\n+++ b/n.py\n@@ -0,0 +1,2 @@\n+a = 1\n+b = 2\n"
    rev = reverse_unidiff(new)
    assert "deleted file mode 100644" in rev and "--- a/n.py\n+++ /dev/null" in rev and "@@ -1,2 +0,0 @@" in rev
    assert "-a = 1\n-b = 2\n" in rev and reverse_unidiff(rev) == new


def test_smith_row_becomes_a_task_with_hidden_tests_and_eval_script():
    t = _row_to_task(ROW, "smith")
    assert t.is_smith and t.bug_patch == BUG and t.patch == reverse_unidiff(BUG)
    assert t.image.endswith(":latest") and t.sif_name == "swesmith.x86_64.owner_1776_proj.abcdef12.sif"
    assert t.hidden_files == ("tests/test_mod.py",) and t.pass_to_pass == ("tests/test_mod.py::test_g",)  # file scope
    assert t.upstream_repo == "owner/proj" and t.base_commit == "abcdef12" and t.log_parser == "parse_log_pytest"
    assert MARK_START in t.eval_script and MARK_END in t.eval_script and "pytest -rA" in t.eval_script and "tests/test_mod.py" in t.eval_script
    row = t.grader_row()
    assert row["eval_script"] == t.eval_script and row["PASS_TO_PASS"] == ["tests/test_mod.py::test_g"]
    assert strategy(t.instance_id) == "func_basic" and not is_pr_mirror(t.instance_id) and is_pr_mirror("o__p.12345678.pr_1959__ab")


def test_setup_scripts_build_the_bug_state_and_scrub_history():
    t = _row_to_task(ROW, "smith")
    sb = t.sandbox_setup
    assert "git apply --verbose /tmp/.sarl_bug.diff" in sb and "rm -f -- 'tests/test_mod.py'" in sb
    assert "rm -rf .git" in sb and "git init -q" in sb and "commit -q -m 'task state'" in sb
    assert BUG.rstrip("\n") in sb  # the patch travels inline, inside a heredoc
    gr = t.grader_setup("/sarl_in/bug.diff")
    assert "git apply --verbose /sarl_in/bug.diff" in gr and "rm -f --" not in gr and "rm -rf .git" in gr
    plain = Task("x", "o/r", "abc", "issue", "img:latest", ("a",), ())
    assert plain.sandbox_setup is None and plain.grader_setup("/p") is None and not plain.is_smith and plain.sif_name == "x.sif"


def test_contamination_guard_sees_through_mirror_names_and_copied_issues():
    smith = _row_to_task(ROW, "smith")
    verified_like = Task("astropy__astropy-1", "astropy/astropy", "deadbeef", "unrelated", "img", ("t",), ())
    same_repo = _row_to_task({**ROW, "repo": "swesmith/astropy__astropy.12345678", "instance_id": "astropy__astropy.12345678.func_basic__q"}, "smith")
    copied = _row_to_task({**ROW, "instance_id": "owner__proj.abcdef12.pr_7__q", "problem_statement": "unrelated"}, "smith")
    rep = contamination_report([smith, same_repo, copied], [verified_like])
    assert rep["hits"]["repo"] == [same_repo.instance_id]
    assert rep["hits"]["problem_statement"] == [copied.instance_id]
    assert not rep["clean"] and same_repo.contaminates_eval and not smith.contaminates_eval
    assert upstream_repo("swesmith/pylint-dev__astroid.b114f6b5") == "pylint-dev/astroid"
