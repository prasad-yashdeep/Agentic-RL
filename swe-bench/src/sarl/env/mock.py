"""An in-memory stand-in for a repository sandbox, with the same semantics as the reference's MockEnv
(abgoswam/swe_in_prod_vizuara_01 utils.py, which we do not vendor): a dict of files, a handful of commands
(ls, cat, grep, a canned pytest, a heredoc write), and the unified diff of the virtual FS as the patch.

The files are supplied by the caller (tests and the anchor notebook load the reference's trimmed astropy
`_cstack` from reference/_upstream when it has been fetched)."""

from __future__ import annotations

import difflib
import re
import shlex

from sarl.env.base import CommandResult

_HEREDOC = re.compile(r"cat\s*>\s*(\S+)\s*<<\s*'?(\w+)'?\n(.*?)\n\2", re.S)


class MockEnv:
    def __init__(self, files: dict[str, str], failing_tests: tuple[str, ...] | list[str] = (), extra_listing: str = "setup.py\nREADME.rst\ntests/"):
        self.fs = dict(files)
        self.orig = dict(files)
        self.failing_tests = list(failing_tests)
        self.extra_listing = extra_listing
        self.calls: list[str] = []

    # lifecycle (nothing to do in memory)
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def run(self, command: str, timeout: float | None = None) -> CommandResult:
        self.calls.append(command)
        cmd = command.strip()
        m = _HEREDOC.search(command)
        if m:  # the one state-changing action
            path, body = m.group(1), m.group(3)
            self.fs[path] = body
            return CommandResult(0, f"[mock env] wrote {len(body.splitlines())} lines to {path}")
        if cmd.startswith("ls"):
            return CommandResult(0, "\n".join(sorted(self.fs)) + ("\n" + self.extra_listing if self.extra_listing else ""))
        if cmd.startswith("cat "):
            path = cmd.split()[-1]
            if path in self.fs:
                return CommandResult(0, "[mock env] only the region near the bug is available\n" + self.fs[path])
            return CommandResult(1, f"cat: {path}: No such file or directory")
        if cmd.startswith("grep"):
            try:
                toks = shlex.split(cmd)
            except ValueError:
                toks = cmd.split()
            pats = [t for t in toks[1:-1] if not t.startswith("-")]
            pat = pats[0] if pats else ""
            src = self.fs.get(toks[-1] if toks else "", "")
            hits = [f"{i + 1}:{line}" for i, line in enumerate(src.split("\n")) if pat and pat in line]
            return CommandResult(0 if hits else 1, "\n".join(hits) if hits else "(no matches)")
        if "pytest" in cmd:
            body = "\n".join(f"FAILED {n}" for n in self.failing_tests)
            return CommandResult(1, f"{body}\n=== {len(self.failing_tests)} failed ===\n[mock env] test names are real, but nothing was executed")
        if cmd.startswith("echo "):
            return CommandResult(0, cmd[5:].strip().strip("'\""))
        name = cmd.split()[0] if cmd.split() else cmd
        return CommandResult(127, f"bash: {name}: command not found\n[mock env] implements only: ls, cat, grep, pytest, echo, cat > FILE <<'EOF'")

    def patch(self) -> str:
        out: list[str] = []
        for path, body in self.fs.items():
            before = self.orig.get(path, "")
            if body != before:
                out += list(difflib.unified_diff(before.split("\n"), body.split("\n"), fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""))
        return "\n".join(out)
