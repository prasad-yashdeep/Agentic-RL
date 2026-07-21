#!/usr/bin/env bash
# Re-apply the harbor singularity-server patch after any `uv tool upgrade harbor`.
# Bug: server.py exec_command() uses process.communicate(), which blocks on pipe
# EOF — a background child (apt-postinst daemons, asciinema, task servers) that
# inherits the stdout pipe keeps the request hung after the command exits, until
# the phase timeout kills the whole trial. Patch: file-based capture + wait().
# Reference diff: scripts/patches/harbor-server-pipe-eof.patch
set -euo pipefail
SERVER=$(ls "$HOME"/.local/share/uv/tools/harbor/lib/python3.*/site-packages/harbor/environments/singularity/server.py 2>/dev/null | head -1)
[ -n "$SERVER" ] || { echo "harbor singularity server.py not found"; exit 1; }
if grep -q "PATCHED (tarl)" "$SERVER"; then
    echo "already patched: $SERVER"
    exit 0
fi
cp "$SERVER" "$SERVER.orig"
python3 - "$SERVER" <<'EOF'
import sys

p = sys.argv[1]
src = open(p).read()
old = '''    process = subprocess.Popen(
        actual_command,
        shell=True,
        executable="/bin/bash",
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        env=env,
    )

    stdout, _ = process.communicate()
    actual_output = stdout.strip() if stdout else None'''
new = '''    # PATCHED (tarl): capture stdout via a temp file and wait() for the process
    # instead of communicate(), which blocks on pipe EOF -- a background child
    # (daemons from apt postinst, asciinema, task servers) inheriting the pipe
    # keeps it open forever after the foreground command exits, hanging the
    # request until the phase-level timeout kills the trial.
    import tempfile

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as _outf:
        process = subprocess.Popen(
            actual_command,
            shell=True,
            executable="/bin/bash",
            stdout=_outf,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=env,
        )
        process.wait()
        _outf.seek(0)
        stdout = _outf.read()
    actual_output = stdout.strip() if stdout else None'''
assert src.count(old) == 1, f"expected exactly 1 match, got {src.count(old)} (harbor version changed?)"
open(p, "w").write(src.replace(old, new))
print(f"patched: {p}")
EOF
python3 -m py_compile "$SERVER" && echo "syntax OK"
