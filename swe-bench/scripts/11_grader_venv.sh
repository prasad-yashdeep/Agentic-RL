#!/usr/bin/env bash
# Node-local grader venv: swebench==5.0.2 (TestSpec.eval_script + log parsers + get_eval_report). Kept OUT of the
# shared project venv (its dependency set would re-resolve the pinned CUDA stack) and off NFS. Run on node08.
set -euo pipefail
case "$(hostname)" in bcm11-headnode*) echo "refusing on the headnode"; exit 1;; esac
V="${SARL_GRADER_VENV:-/var/tmp/swe/grader-venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/var/tmp/swe/uv-cache}"
mkdir -p "$(dirname "$V")" "$UV_CACHE_DIR"
[ -x "$V/bin/python" ] || "$HOME/.local/bin/uv" venv "$V" --python /usr/bin/python3
"$HOME/.local/bin/uv" pip install --python "$V/bin/python" "swebench==5.0.2" >/dev/null
"$V/bin/python" -c "import swebench; from swebench.harness.grading import get_eval_report; from swebench.harness.utils import make_test_spec; print('GRADER_OK swebench', swebench.__version__)"
