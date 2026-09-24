#!/usr/bin/env bash
# Agentic-RL monorepo health check. Light (file reads + one JSON parse), safe on the headnode.
# Run: make doctor (root) or make -C terminal-bench doctor. `make exp` runs it first. Exit 1 on any FAIL.
#
# Why it exists: on 2026-09-24 terminal-agent-rl moved to /home/yashp/Agentic-RL/terminal-bench with NO
# compatibility symlink. Anything still pointing at the old path (a script, the venv, the Jupyter kernel,
# or Harbor's saved jobs_dir which `harbor jobs resume` trusts) would break or, worse, silently recreate
# the old tree. This script checks every one of those places.
set -u
ARL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TB="$ARL/terminal-bench"; SWE="$ARL/swe-bench"
VENV="${ARL_VENV:-$ARL/.venv}"
OLD="/home/yashp/terminal-agent""-rl"          # split so this file never matches its own scan
fail=0
ok()   { printf 'ok    %s\n' "$*"; }
bad()  { printf 'FAIL  %s\n' "$*"; fail=1; }
warn() { printf 'warn  %s\n' "$*"; }

# 1. live code must not reference the old path (history/docs and two frozen records are allowed)
OLD_RE='(/home/yashp|\$HOME|~)/terminal-agent''-rl'   # every spelling a script might use
hits=$(git -C "$ARL" grep -l -E "$OLD_RE" -- \
        '*.sh' '*Makefile' '*.mk' '*.py' '*.toml' '*.yaml' '*.yml' \
        ':!terminal-bench/scripts/06_grpo_micro.py' ':!terminal-bench/configs/lockfile.txt' 2>/dev/null)
[ -z "$hits" ] && ok "no old path in tracked code" || bad "old path in: $(echo $hits | tr '\n' ' ')"

# 2. the shared venv
if [ -x "$VENV/bin/python" ]; then ok "venv python: $VENV/bin/python"; else bad "no venv python at $VENV/bin/python"; fi
n=$(grep -rlI -F "$OLD" "$VENV/bin" 2>/dev/null | wc -l)
[ "$n" = 0 ] && ok "venv bin/ has no old path" || bad "$n venv bin/ files still reference the old path"
for pth in "$VENV"/lib/python3*/site-packages/__editable__*.pth; do
    [ -e "$pth" ] || continue
    while IFS= read -r line; do
        case "$line" in /*) [ -d "$line" ] && ok "editable $(basename "$pth") -> $line" || bad "editable $(basename "$pth") -> missing $line";; esac
    done < "$pth"
done

# 3. Jupyter kernels
for k in tarl sarl; do
    kj="$HOME/.local/share/jupyter/kernels/$k/kernel.json"
    [ -f "$kj" ] || { [ "$k" = sarl ] && warn "kernel $k not registered yet" || bad "kernel $k missing"; continue; }
    py=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['argv'][0])" "$kj")
    [ -x "$py" ] && ok "kernel $k -> $py" || bad "kernel $k -> missing interpreter $py"
done

# 4. Harbor A0 job: `harbor jobs resume` rebuilds the job dir from config.jobs_dir/job_name
A0="$TB/jobs/2026-07-21__05-38-50"
if [ -f "$A0/config.json" ]; then
    python3 - "$A0" <<'PY' && ok "A0 resume target == real A0 dir (180 trials consistent)" || bad "A0 resume target mismatch (see above)"
import json, sys
from pathlib import Path
a0 = Path(sys.argv[1]).resolve()
c = json.loads((a0 / "config.json").read_text())
jd = (Path(c["jobs_dir"]) / c["job_name"])
if not jd.exists() or jd.resolve() != a0:
    print(f"  config.jobs_dir/job_name = {jd} (exists={jd.exists()})"); sys.exit(1)
bad = [p.parent.name for p in a0.glob("*/config.json") if json.loads(p.read_text()).get("trials_dir") != str(jd)]
if bad:
    print(f"  {len(bad)} trial configs disagree, e.g. {bad[:3]}"); sys.exit(1)
PY
else
    warn "A0 job dir not present (fresh checkout?)"
fi

# 5. runtime files the tests and notebook parse
for f in g4.log watch.log; do [ -f "$TB/$f" ] && ok "terminal-bench/$f present" || warn "terminal-bench/$f missing (real_data tests will skip)"; done

# 6. the old path itself: a symlink is not allowed; the read-only placeholder is temporary
if [ -L "$OLD" ]; then bad "$OLD is a symlink (the cut-over forbids one)"
elif [ -d "$OLD" ]; then
    extra=$(ls -A "$OLD" | grep -v '^MOVED.md$' | head -3)
    [ -z "$extra" ] && warn "$OLD placeholder still present (remove after reopening Cursor on $ARL)" || bad "$OLD contains files again: $extra"
else ok "old path absent"
fi

[ -d "$SWE" ] && ok "swe-bench/ present" || warn "swe-bench/ not created yet"
[ $fail = 0 ] && echo "DOCTOR_OK" || echo "DOCTOR_FAIL"
exit $fail
