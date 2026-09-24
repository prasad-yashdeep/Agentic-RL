import subprocess
import sys

from sarl.paths import ARL


def test_tarl_and_sarl_stay_independent():
    r = subprocess.run([sys.executable, str(ARL / "tools" / "check_layering.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
