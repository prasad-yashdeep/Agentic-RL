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

# --- patch 2: bootstrap.sh get-pip for containers with python < 3.10 ---------
# Generic https://bootstrap.pypa.io/get-pip.py refuses py<=3.9 (qemu-* images
# ship 3.9) -> "FATAL: cannot bootstrap pip". Use the versioned URL instead.
BOOTSTRAP="$(dirname "$SERVER")/bootstrap.sh"
if grep -q "PATCHED (tarl)" "$BOOTSTRAP"; then
    echo "already patched: $BOOTSTRAP"
else
    cp "$BOOTSTRAP" "$BOOTSTRAP.orig"
    python3 - "$BOOTSTRAP" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = """urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '/tmp/get-pip.py')"""
new = """# PATCHED (tarl): generic get-pip.py requires py>=3.10; use versioned URL for older pythons
import sys as _s
_v = '%d.%d' % _s.version_info[:2]
_url = 'https://bootstrap.pypa.io/get-pip.py' if _s.version_info[:2] >= (3, 10) else ('https://bootstrap.pypa.io/pip/%s/get-pip.py' % _v)
urllib.request.urlretrieve(_url, '/tmp/get-pip.py')"""
assert s.count(old) == 1, f"expected exactly 1 match, got {s.count(old)} (harbor version changed?)"
s = s.replace(old, new)
# 2b: get-pip still fails on old debian pythons missing distutils -> apt fallback
old2 = '''      && "$_HARBOR_PY" /tmp/get-pip.py --quiet 2>/dev/null \\
      || { echo "[harbor] FATAL: cannot bootstrap pip" >&2; exit 1; }
  fi'''
new2 = '''      && "$_HARBOR_PY" /tmp/get-pip.py --quiet 2>/dev/null \\
      || { echo "[harbor] get-pip failed; trying apt python3-pip (PATCHED tarl)..." >&2; \\
           apt-get update >/dev/null 2>&1; \\
           apt-get install -y python3-pip python3-distutils >/dev/null 2>&1; }
  fi
  if ! "$_HARBOR_PY" -m pip --version >/dev/null 2>&1; then
    echo "[harbor] FATAL: cannot bootstrap pip" >&2; exit 1
  fi'''
assert s.count(old2) == 1, f"2b matches: {s.count(old2)}"
open(p, "w").write(s.replace(old2, new2))
print(f"patched: {p}")
EOF
fi

# --- patch 3: env-configurable context-limit fallback ------------------------
# Unknown models (our vLLM endpoint name) fall back to a 1M context limit;
# Terminus then overruns the server's real window until vLLM 400s with
# "max_tokens must be at least 1, got 0". Set HARBOR_FALLBACK_CONTEXT_LIMIT
# (env.sh pins it to the served --max-model-len).
LITE="$(dirname "$SERVER")/../../llms/lite_llm.py"
LITE="$(readlink -f "$LITE")"
if grep -q "HARBOR_FALLBACK_CONTEXT_LIMIT" "$LITE"; then
    echo "already patched: $LITE"
else
    cp "$LITE" "$LITE.orig"
    python3 - "$LITE" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = "        fallback_context_limit = 1000000"
new = """        # PATCHED (tarl): unknown models (our vLLM endpoint) fell back to 1M and
        # blew past the server's real window -> vLLM 400 "max_tokens ... got 0".
        import os as _os
        fallback_context_limit = int(_os.environ.get("HARBOR_FALLBACK_CONTEXT_LIMIT", "1000000"))"""
assert s.count(old) == 1, f"expected exactly 1 match, got {s.count(old)} (harbor version changed?)"
open(p, "w").write(s.replace(old, new))
print(f"patched: {p}")
EOF
    python3 -m py_compile "$LITE" && echo "syntax OK"
fi
