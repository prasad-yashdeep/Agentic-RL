"""SWE-smith (Yang et al. 2025) as a training-task source: synthetic bugs in real repositories, one Docker image per
repository, MIT. Scoped 2026-09-25 (experiments/EXPERIMENTS.md "SWE-smith as the next task source").

How a SWE-smith row differs from a SWE-bench row, and what this module does about it:
- `patch` INTRODUCES the bug into the repository's clean state (the image's `/testbed` at the mirror's single commit).
  The fix is the reversed diff: `reverse_unidiff` produces it, and it becomes `Task.patch` (the gold), while the
  original goes to `Task.bug_patch`.
- The agent must start from the buggy state with the FAIL_TO_PASS test files removed (the official setup), and the
  grader must rebuild the same buggy state with the tests present. `setup_script` does both, offline, from the dataset
  patch alone; the sandbox variant also wipes git history, so `git show`/`git log -p` cannot reveal the fix.
- There is no eval_script: `eval_script_for` synthesizes the official pytest command over the FAIL_TO_PASS test files
  with swebench's markers, parsed by swebench's `parse_log_pytest`. PASS_TO_PASS is restricted to tests in those files
  (the same file scope as SWE-bench dev), so grading a task costs one file's tests, not the whole suite.
- `repo` is the mirror name `swesmith/<owner>__<repo>.<sha8>`; `upstream_repo` recovers `owner/repo` for the
  contamination guard. Instances named `<repo>.pr_<n>__…` mirror real PRs, seven of which duplicate SWE-bench dev
  tasks byte for byte: never use them.
"""

from __future__ import annotations

import re
import secrets

MARK_START = ">>>>> Start Test Output"
MARK_END = ">>>>> End Test Output"
PYTEST_CMD = "pytest -rA --disable-warnings --color=no --tb=no -p no:cacheprovider"
_MIRROR = re.compile(r"^swesmith/(?P<owner>[^/]+?)__(?P<repo>.+)\.(?P<sha>[0-9a-f]{8})$")
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def is_smith_row(r: dict) -> bool:
    return "image_name" in r and "base_commit" not in r


def upstream_repo(repo: str) -> str:
    m = _MIRROR.match(repo)
    return f"{m['owner']}/{m['repo']}" if m else repo


def base_sha8(repo: str) -> str:
    m = _MIRROR.match(repo)
    return m["sha"] if m else ""


def strategy(instance_id: str) -> str:
    """`<owner>__<repo>.<sha8>.<strategy>__<id>` -> the bug-generation strategy, e.g. func_basic, lm_rewrite, pr_1959."""
    tail = instance_id.split(".", 2)[-1] if instance_id.count(".") >= 2 else instance_id
    return tail.rsplit("__", 1)[0]


def is_pr_mirror(instance_id: str) -> bool:
    return strategy(instance_id).startswith("pr_")


def test_files(test_ids) -> tuple[str, ...]:
    return tuple(sorted({t.split("::", 1)[0] for t in test_ids}))


def reverse_unidiff(patch: str) -> str:
    """The inverse of a unified diff (what `git apply --reverse` would apply): swap the file pair, the hunk ranges and
    the +/- of every body line; new files become deleted files and vice versa."""
    out = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            m = re.match(r"^diff --git a/(.*) b/(.*)$", line)
            out.append(f"diff --git a/{m[2]} b/{m[1]}" if m else line)
        elif line.startswith("--- ") and not line.startswith("--- a/") and line != "--- /dev/null":
            out.append(line)
        elif line.startswith("index "):
            m = re.match(r"^index ([0-9a-f]+)\.\.([0-9a-f]+)(.*)$", line)
            out.append(f"index {m[2]}..{m[1]}{m[3]}" if m else line)
        elif line.startswith("new file mode "):
            out.append("deleted file mode " + line[len("new file mode "):])
        elif line.startswith("deleted file mode "):
            out.append("new file mode " + line[len("deleted file mode "):])
        elif line.startswith("rename from "):
            out.append("rename to " + line[len("rename from "):])
        elif line.startswith("rename to "):
            out.append("rename from " + line[len("rename to "):])
        elif line.startswith("--- "):
            out.append(("+++ b/" + line[6:]) if line.startswith("--- a/") else "+++ /dev/null")
            out.append(None)  # placeholder: the following +++ line swaps in front
        elif line.startswith("+++ "):
            new = ("--- a/" + line[6:]) if line.startswith("+++ b/") else "--- /dev/null"
            i = len(out) - 1
            while out[i] is not None:
                i -= 1
            out[i] = out[i - 1]
            out[i - 1] = new
        elif line.startswith("@@ "):
            m = _HUNK.match(line)
            if not m:
                out.append(line)
                continue
            a, b, c, d, ctx = m.groups()
            lhs = f"-{c}" + (f",{d}" if d is not None else "")
            rhs = f"+{a}" + (f",{b}" if b is not None else "")
            out.append(f"@@ {lhs} {rhs} @@{ctx}")
        elif line.startswith("+"):
            out.append("-" + line[1:])
        elif line.startswith("-"):
            out.append("+" + line[1:])
        else:
            out.append(line)  # context, "\ No newline at end of file", blank
    assert None not in out, "unbalanced ---/+++ headers"
    return "\n".join(out) + "\n"


def eval_script_for(f2p_files) -> str:
    files = " ".join(f2p_files)
    return "\n".join(["#!/bin/bash", "set -uxo pipefail", "source /opt/miniconda3/bin/activate", "conda activate testbed",
                      "cd /testbed", f": '{MARK_START}'", f"{PYTEST_CMD} {files}", f": '{MARK_END}'"]) + "\n"


def setup_script(bug_patch: str, hidden_files, *, keep_tests: bool, patch_path: str | None = None) -> str:
    """Bash that turns the image's clean checkout into the task state. Without `patch_path` the patch travels inline
    (a heredoc, for `apptainer exec … bash -c`); with it, the grader reads it from its read-only bind. keep_tests=False
    (the agent's sandbox) deletes the FAIL_TO_PASS test files as the official setup does and leaves a one-commit
    repository with no trace of the bug diff; keep_tests=True (the grader) keeps the tests so they can run."""
    if not bug_patch.endswith("\n"):
        bug_patch += "\n"
    lines = ["set -e", "cd /testbed", "git config --global --add safe.directory /testbed >/dev/null 2>&1 || true"]
    if patch_path:
        lines.append(f"git apply --verbose {patch_path}")
    else:
        eof = "SARL_BUG_PATCH_" + secrets.token_hex(6)
        while eof in bug_patch:
            eof = "SARL_BUG_PATCH_" + secrets.token_hex(6)
        lines += [f"cat > /tmp/.sarl_bug.diff <<'{eof}'", bug_patch.rstrip("\n"), eof, "git apply --verbose /tmp/.sarl_bug.diff",
                  "rm -f /tmp/.sarl_bug.diff"]
    if not keep_tests and hidden_files:
        lines.append("rm -f -- " + " ".join(f"'{f}'" for f in hidden_files))
    # one commit, no remote, no reflog: nothing in the repository says what changed
    lines += ["rm -rf .git", "git init -q", "git add -A",
              "git -c user.email=sarl@local -c user.name=sarl -c commit.gpgsign=false commit -q -m 'task state'"]
    return "\n".join(lines) + "\n"
